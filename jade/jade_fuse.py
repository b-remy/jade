import jax
import jax.numpy as jnp
from flax import nnx

import numpy as np
from einops import repeat, rearrange
import math

# =============================================================================
# Reused components (inline to keep file self-contained)
# If you prefer, replace these with:
#   from jade import (BottleneckPatchEmbed, RMSNorm, rotate_half,
#                     scaled_dot_product_attention, VisionRotaryEmbeddingFast,
#                     SwiGLUFFN, get_2d_sincos_pos_embed, modulate)
# =============================================================================

class BottleneckPatchEmbed(nnx.Module):
    """Image to Patch Embedding with bottleneck."""
    def __init__(self, img_size=256, patch_size=16, in_chans=1,
                 pca_dim=768, embed_dim=768, bias=True, rngs=None):
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.proj1 = nnx.Conv(in_features=in_chans, out_features=pca_dim,
                              kernel_size=(patch_size, patch_size),
                              strides=(patch_size, patch_size),
                              use_bias=False, rngs=rngs)
        self.proj2 = nnx.Conv(in_features=pca_dim, out_features=embed_dim,
                              kernel_size=(1, 1), strides=(1, 1),
                              use_bias=bias, rngs=rngs)

    def __call__(self, x):
        H, W, C = x.shape
        assert H == self.img_size[0] and W == self.img_size[1]
        x = self.proj1(x)
        x = self.proj2(x)
        H_out, W_out, C_out = x.shape
        return x.reshape(H_out * W_out, C_out)


class RMSNorm(nnx.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, rngs=None):
        self.variance_epsilon = eps
        self.weight = nnx.Param(jnp.ones(hidden_size))

    def __call__(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.variance_epsilon)
        return (self.weight.value * hidden_states).astype(input_dtype)


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x[..., 0], x[..., 1]
    x = jnp.stack([-x2, x1], axis=-1)
    return rearrange(x, '... d r -> ... (d r)')


def scaled_dot_product_attention(query, key, value, dropout_p=0.0,
                                  train=True, rngs=None):
    scale_factor = 1.0 / math.sqrt(query.shape[-1])
    query_f32 = query.astype(jnp.float32)
    key_f32 = key.astype(jnp.float32)
    attn_weight = jnp.einsum('hld,hsd->hls', query_f32, key_f32) * scale_factor
    attn_weight = jax.nn.softmax(attn_weight, axis=-1)
    if train and dropout_p > 0.0:
        keep_prob = 1.0 - dropout_p
        mask = jax.random.bernoulli(rngs(), keep_prob, attn_weight.shape)
        attn_weight = jnp.where(mask, attn_weight / keep_prob, 0.0)
    return jnp.einsum('hls,hsd->hld', attn_weight, value)


class VisionRotaryEmbeddingFast(nnx.Module):
    def __init__(self, dim, pt_seq_len=16, ft_seq_len=None, custom_freqs=None,
                 freqs_for='lang', theta=10000, max_freq=10, num_freqs=1,
                 num_cls_token=0, rngs=None):
        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
        elif freqs_for == 'pixel':
            freqs = jnp.linspace(1.0, max_freq / 2, dim // 2) * jnp.pi
        elif freqs_for == 'constant':
            freqs = jnp.ones(num_freqs, dtype=jnp.float32)
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        ft_seq_len = ft_seq_len or pt_seq_len
        t = jnp.arange(ft_seq_len, dtype=jnp.float32) / ft_seq_len * pt_seq_len
        freqs = jnp.einsum('t,f->tf', t, freqs)
        freqs = repeat(freqs, 't f -> t (f r)', r=2)
        freqs_h = freqs[:, None, :]
        freqs_w = freqs[None, :, :]
        freqs_h = jnp.broadcast_to(freqs_h, (ft_seq_len, ft_seq_len, freqs.shape[-1]))
        freqs_w = jnp.broadcast_to(freqs_w, (ft_seq_len, ft_seq_len, freqs.shape[-1]))
        freqs = jnp.concatenate([freqs_h, freqs_w], axis=-1)
        freqs_flat = freqs.reshape(-1, freqs.shape[-1])

        if num_cls_token > 0:
            cos_img = jnp.cos(freqs_flat)
            sin_img = jnp.sin(freqs_flat)
            N_img, D = cos_img.shape
            self.freqs_cos = jnp.concatenate([jnp.ones((num_cls_token, D)), cos_img], axis=0)
            self.freqs_sin = jnp.concatenate([jnp.zeros((num_cls_token, D)), sin_img], axis=0)
        else:
            self.freqs_cos = jnp.cos(freqs_flat)
            self.freqs_sin = jnp.sin(freqs_flat)

    def __call__(self, t):
        seq_len = t.shape[-2]
        if seq_len != self.freqs_cos.shape[0]:
            freqs_cos = self.freqs_cos[:seq_len]
            freqs_sin = self.freqs_sin[:seq_len]
        else:
            freqs_cos = self.freqs_cos
            freqs_sin = self.freqs_sin
        return t * freqs_cos + rotate_half(t) * freqs_sin


class SwiGLUFFN(nnx.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0,
                 bias: bool = True, rngs=None):
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nnx.Linear(dim, 2 * hidden_dim, use_bias=bias, rngs=rngs)
        self.w3 = nnx.Linear(hidden_dim, dim, use_bias=bias, rngs=rngs)
        self.drop_rate = drop
        self.rngs = rngs

    def __call__(self, x, train: bool = False):
        x12 = self.w12(x)
        x1, x2 = jnp.split(x12, 2, axis=-1)
        hidden = nnx.silu(x1) * x2
        if train and self.drop_rate > 0:
            keep_prob = 1.0 - self.drop_rate
            mask = jax.random.bernoulli(self.rngs(), keep_prob, hidden.shape)
            hidden = jnp.where(mask, hidden / keep_prob, 0.0)
        return self.w3(hidden)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def _get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)

def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def modulate(x, shift, scale):
    """Apply adaptive modulation: x * (1 + scale) + shift."""
    return x * (1 + jnp.expand_dims(scale, -2)) + jnp.expand_dims(shift, -2)


# =============================================================================
# New JADE_FUSE components
# =============================================================================


class JointConditioner(nnx.Module):
    """
    Fuses timestep (scalar) and noisy cosmology (vector) into a single
    conditioning embedding c_emb used to drive AdaLN-Zero modulation.

    Input:  t (scalar), cosmo (cosmo_dim,)
    Output: c_emb (hidden_size,)
    """

    def __init__(self, cosmo_dim: int, hidden_size: int, rngs=None):
        self.lin1 = nnx.Linear(1 + cosmo_dim, hidden_size, use_bias=True, rngs=rngs)
        self.lin2 = nnx.Linear(hidden_size, hidden_size, use_bias=True, rngs=rngs)

    def __call__(self, t, cosmo):
        """
        Args:
            t: scalar timestep (unbatched)
            cosmo: (cosmo_dim,) noisy cosmology vector
        Returns:
            c_emb: (hidden_size,)
        """
        t = jnp.atleast_1d(t)  # ensure (1,)
        x = jnp.concatenate([t, cosmo], axis=-1)  # (1 + cosmo_dim,)
        return self.lin2(nnx.silu(self.lin1(x)))


class FuseAttention(nnx.Module):
    """Multi-head self-attention with RMSNorm on Q/K and RoPE."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True,
                 attn_drop=0., proj_drop=0., rngs=None):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, rngs=rngs)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, rngs=rngs)
            self.k_norm = RMSNorm(self.head_dim, rngs=rngs)
        else:
            self.q_norm = lambda x: x
            self.k_norm = lambda x: x

        self.attn_drop_p = attn_drop
        self.proj_drop_p = proj_drop
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        self.rngs = rngs

    def __call__(self, x, rope, train=False):
        """x: (N, C), returns (N, C)."""
        N, C = x.shape
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'N (three H D) -> three H N D',
                         three=3, H=self.num_heads, D=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = rope(q)
        k = rope(k)

        x = scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop_p, train=train, rngs=self.rngs
        )
        x = rearrange(x, 'H N D -> N (H D)')
        x = self.proj(x)

        if train and self.proj_drop_p > 0:
            keep_prob = 1.0 - self.proj_drop_p
            mask = jax.random.bernoulli(self.rngs(), keep_prob, x.shape)
            x = jnp.where(mask, x / keep_prob, 0.0)
        return x


class FuseBlock(nnx.Module):
    """
    Transformer block with AdaLN-Zero conditioning.

    c_emb → ada_linear → 6 * hidden_size → (shift_msa, scale_msa, gate_msa,
                                              shift_mlp, scale_mlp, gate_mlp)

    ada_linear is zero-initialized so the block starts as identity.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,
                 attn_drop=0.0, proj_drop=0.0, rngs=None):
        self.norm1 = RMSNorm(hidden_size, eps=1e-6, rngs=rngs)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6, rngs=rngs)

        self.attn = FuseAttention(
            hidden_size, num_heads=num_heads,
            qkv_bias=True, qk_norm=True,
            attn_drop=attn_drop, proj_drop=proj_drop, rngs=rngs
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop, rngs=rngs)

        # AdaLN-Zero projection: c_emb → 6 modulation vectors
        self.ada_linear = nnx.Linear(hidden_size, 6 * hidden_size, rngs=rngs)

    def __call__(self, x, c_emb, feat_rope=None, train=False):
        """
        Args:
            x: (N, hidden_size) — token sequence (no batch dim)
            c_emb: (hidden_size,) — joint conditioning vector
            feat_rope: RoPE callable
            train: training flag
        Returns:
            (N, hidden_size)
        """
        # Predict 6 modulation vectors
        ada = self.ada_linear(nnx.silu(c_emb))  # (6 * hidden_size,)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            jnp.split(ada, 6, axis=-1)

        # --- Attention path with AdaLN ---
        res = modulate(self.norm1(x), shift_msa, scale_msa)
        res = self.attn(res, rope=feat_rope, train=train)
        x = x + jnp.expand_dims(gate_msa, -2) * res

        # --- MLP path with AdaLN ---
        res = modulate(self.norm2(x), shift_mlp, scale_mlp)
        res = self.mlp(res, train=train)
        x = x + jnp.expand_dims(gate_mlp, -2) * res

        return x


class FuseFinalLayer(nnx.Module):
    """Final layer for field prediction, modulated by c_emb."""

    def __init__(self, hidden_size, patch_size, out_channels, rngs=None):
        self.norm_final = RMSNorm(hidden_size, rngs=rngs)
        self.linear = nnx.Linear(
            hidden_size, patch_size * patch_size * out_channels,
            use_bias=True, rngs=rngs
        )
        self.ada_linear = nnx.Linear(hidden_size, 2 * hidden_size, rngs=rngs)

    def __call__(self, x, c_emb):
        """x: (N, hidden_size), c_emb: (hidden_size,) → (N, p*p*C)"""
        ada = self.ada_linear(nnx.silu(c_emb))
        shift, scale = jnp.split(ada, 2, axis=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class CosmoReadoutHead(nnx.Module):
    """
    Cross-Attention readout head for cosmology prediction.

    A single learnable query vector attends over all backbone output tokens,
    then projects to cosmo_dim.
    """

    def __init__(self, hidden_size, cosmo_dim, num_heads=4, rngs=None):
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size

        # Learnable query: (1, hidden_size)
        self.cosmo_query = nnx.Param(
            jax.random.normal(rngs(), (1, hidden_size)) * 0.02
        )

        self.q_proj = nnx.Linear(hidden_size, hidden_size, use_bias=True, rngs=rngs)
        self.k_proj = nnx.Linear(hidden_size, hidden_size, use_bias=True, rngs=rngs)
        self.v_proj = nnx.Linear(hidden_size, hidden_size, use_bias=True, rngs=rngs)

        self.q_norm = RMSNorm(self.head_dim, rngs=rngs)
        self.k_norm = RMSNorm(self.head_dim, rngs=rngs)

        self.out_norm = RMSNorm(hidden_size, rngs=rngs)
        self.out_proj = nnx.Linear(hidden_size, cosmo_dim, use_bias=True, rngs=rngs)

    def __call__(self, tokens):
        """
        Args:
            tokens: (S, hidden_size) — all backbone output tokens
        Returns:
            cosmo_pred: (cosmo_dim,)
        """
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(self.cosmo_query.value)  # (1, hidden_size)
        k = self.k_proj(tokens)                   # (S, hidden_size)
        v = self.v_proj(tokens)                   # (S, hidden_size)

        q = rearrange(q, 'L (H D) -> H L D', H=H, D=D)
        k = rearrange(k, 'S (H D) -> H S D', H=H, D=D)
        v = rearrange(v, 'S (H D) -> H S D', H=H, D=D)

        q = self.q_norm(q)
        k = self.k_norm(k)

        scale = 1.0 / math.sqrt(D)
        attn = jnp.einsum('hld,hsd->hls',
                           q.astype(jnp.float32),
                           k.astype(jnp.float32)) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum('hls,hsd->hld', attn, v)  # (H, 1, D)

        out = rearrange(out, 'H L D -> L (H D)')    # (1, hidden_size)
        out = self.out_norm(out[0])                  # (hidden_size,)
        return self.out_proj(out)                    # (cosmo_dim,)


# =============================================================================
# JADE_FUSE — main model
# =============================================================================


class JADE_FUSE(nnx.Module):
    """
    JADE with Fused AdaLN-Zero conditioning.

    - Global context (time + cosmo) → c_emb → AdaLN-Zero modulation.
    - Spatial context (conditioning field) → tokens in sequence.
    - Cosmo prediction → Cross-Attention readout over all output tokens.
    - Field prediction → FinalLayer (AdaLN) + unpatchify.

    All operations are unbatched; use jax.vmap for batching.
    """

    def __init__(
        self,
        input_size=128,
        patch_size=8,
        cond_patch_size=16,
        in_channels=5,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        bottleneck_dim=128,
        cosmo_dim=6,
        cond_channels=5,
        enable_cond_image=True,
        cosmo_readout_heads=4,
        rngs=None,
    ):
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.cond_patch_size = cond_patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cosmo_dim = cosmo_dim
        self.cond_channels = cond_channels
        self.enable_cond_image = enable_cond_image
        self.depth = depth

        # --- RNG allocation ---
        num_rngs = depth + 6
        if enable_cond_image:
            num_rngs += 3
        rng_keys = jax.random.split(rngs(), num_rngs)

        idx = 0
        def next_rngs():
            nonlocal idx
            r = nnx.Rngs(rng_keys[idx]); idx += 1; return r

        rngs_conditioner  = next_rngs()
        rngs_x_embedder   = next_rngs()

        if enable_cond_image:
            rngs_cond_embedder = next_rngs()
            rngs_cond_pos      = next_rngs()

        rngs_rope_nocond = next_rngs()
        if enable_cond_image:
            rngs_rope_cond = next_rngs()

        rngs_final      = next_rngs()
        rngs_cosmo_head = next_rngs()
        rngs_blocks     = [next_rngs() for _ in range(depth)]

        # =================================================================
        # 1) Joint Conditioner:  (t, cosmo) → c_emb ∈ R^{hidden_size}
        # =================================================================
        self.conditioner = JointConditioner(
            cosmo_dim, hidden_size, rngs=rngs_conditioner
        )

        # =================================================================
        # 2) Patch embeddings
        # =================================================================
        # Target field: patch_size (e.g. 8) → 256 tokens for 128×128
        self.x_embedder = BottleneckPatchEmbed(
            input_size, patch_size, in_channels, bottleneck_dim, hidden_size,
            bias=True, rngs=rngs_x_embedder
        )

        # Conditioning image: cond_patch_size (e.g. 16) → 64 tokens for 128×128
        if self.enable_cond_image:
            self.cond_embedder = BottleneckPatchEmbed(
                input_size, cond_patch_size, cond_channels, bottleneck_dim,
                hidden_size, bias=True, rngs=rngs_cond_embedder
            )
            num_cond_patches = (input_size // cond_patch_size) ** 2
            self.cond_pos_embed = nnx.Param(
                jax.random.normal(rngs_cond_pos(), (num_cond_patches, hidden_size)) * 0.02
            )

        # Fixed sin-cos positional embedding for target field tokens
        num_field_patches = (input_size // patch_size) ** 2
        pos_embed = get_2d_sincos_pos_embed(
            hidden_size, int(num_field_patches ** 0.5),
            cls_token=False, extra_tokens=0
        )
        self.pos_embed = nnx.Param(jnp.array(pos_embed, dtype=jnp.float32))

        # =================================================================
        # 3) RoPE variants
        # =================================================================
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len_field = input_size // patch_size

        # Without conditioning image (field tokens only)
        self.feat_rope_nocond = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len_field,
            num_cls_token=0,
            rngs=rngs_rope_nocond
        )

        # With conditioning image (cond_tokens prepended to field tokens)
        if self.enable_cond_image:
            num_cond_patches = (input_size // cond_patch_size) ** 2
            self.feat_rope_cond = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len_field,
                num_cls_token=num_cond_patches,
                rngs=rngs_rope_cond
            )

        # =================================================================
        # 4) Transformer backbone — FuseBlocks with AdaLN-Zero
        # =================================================================
        self.blocks = nnx.List([
            FuseBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                rngs=rngs_blocks[i]
            )
            for i in range(depth)
        ])

        # =================================================================
        # 5) Output heads
        # =================================================================
        self.field_head = FuseFinalLayer(
            hidden_size, patch_size, self.out_channels, rngs=rngs_final
        )
        self.cosmo_head = CosmoReadoutHead(
            hidden_size, cosmo_dim,
            num_heads=cosmo_readout_heads, rngs=rngs_cosmo_head
        )

        # =================================================================
        # 6) Weight initialization
        # =================================================================
        self._initialize_weights()

    # -----------------------------------------------------------------
    def _initialize_weights(self):
        """Careful initialization; AdaLN projections set to zero."""

        def _xavier(module):
            w = module.kernel.value
            flat_shape = (w.shape[0], -1) if w.ndim > 1 else w.shape
            w_flat = w.reshape(w.shape[0], -1)
            w_init = jax.nn.initializers.xavier_uniform()(
                jax.random.PRNGKey(0), w_flat.shape
            )
            module.kernel.value = w_init.reshape(w.shape)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.value = jnp.zeros_like(module.bias.value)

        # Conditioner — small random init
        for i, lin in enumerate([self.conditioner.lin1, self.conditioner.lin2]):
            lin.kernel.value = jax.random.normal(
                jax.random.PRNGKey(100 + i), lin.kernel.value.shape
            ) * 0.02
            lin.bias.value = jnp.zeros_like(lin.bias.value)

        # Patch embedders — Xavier
        embedders = [self.x_embedder]
        if self.enable_cond_image:
            embedders.append(self.cond_embedder)
        for j, emb in enumerate(embedders):
            w1 = emb.proj1.kernel.value
            w1_init = jax.nn.initializers.xavier_uniform()(
                jax.random.PRNGKey(10 + j), w1.reshape(w1.shape[0], -1).shape
            )
            emb.proj1.kernel.value = w1_init.reshape(w1.shape)

            w2 = emb.proj2.kernel.value
            w2_init = jax.nn.initializers.xavier_uniform()(
                jax.random.PRNGKey(20 + j), w2.reshape(w2.shape[0], -1).shape
            )
            emb.proj2.kernel.value = w2_init.reshape(w2.shape)
            emb.proj2.bias.value = jnp.zeros_like(emb.proj2.bias.value)

        # Transformer blocks
        for block in self.blocks:
            _xavier(block.attn.qkv)
            _xavier(block.attn.proj)
            _xavier(block.mlp.w12)
            _xavier(block.mlp.w3)

            # *** AdaLN-Zero: zero init ***
            block.ada_linear.kernel.value = jnp.zeros_like(block.ada_linear.kernel.value)
            block.ada_linear.bias.value   = jnp.zeros_like(block.ada_linear.bias.value)

        # Field head — zero init for ada + output
        _xavier(self.field_head.linear)
        self.field_head.ada_linear.kernel.value = jnp.zeros_like(self.field_head.ada_linear.kernel.value)
        self.field_head.ada_linear.bias.value   = jnp.zeros_like(self.field_head.ada_linear.bias.value)
        self.field_head.linear.kernel.value     = jnp.zeros_like(self.field_head.linear.kernel.value)
        self.field_head.linear.bias.value       = jnp.zeros_like(self.field_head.linear.bias.value)

        # Cosmo readout — Xavier for attention, zero for output
        _xavier(self.cosmo_head.q_proj)
        _xavier(self.cosmo_head.k_proj)
        _xavier(self.cosmo_head.v_proj)
        self.cosmo_head.out_proj.kernel.value = jnp.zeros_like(self.cosmo_head.out_proj.kernel.value)
        self.cosmo_head.out_proj.bias.value   = jnp.zeros_like(self.cosmo_head.out_proj.bias.value)

    # -----------------------------------------------------------------
    def unpatchify(self, x, p):
        """(num_patches, p*p*C) → (H, W, C)"""
        c = self.out_channels
        N = x.shape[0]
        h = w = int(N ** 0.5)
        assert h * w == N
        x = x.reshape(h, w, p, p, c)
        x = jnp.einsum('hwpqc->hpwqc', x)
        return x.reshape(h * p, w * p, c)

    # -----------------------------------------------------------------
    def __call__(self, field, cosmo, t, cond=None, train=False):
        """
        Forward pass (unbatched — use jax.vmap for batching).

        Args:
            field: (H, W, in_channels) — noisy target field
            cosmo: (cosmo_dim,) — noisy cosmology parameters
            t: scalar — diffusion timestep
            cond: (H, W, cond_channels) | None — optional conditioning image
            train: bool

        Returns:
            field_pred: (H, W, in_channels) — denoised field
            cosmo_pred: (cosmo_dim,) — denoised cosmology
        """
        # ---- 1. Fuse conditioning ----
        c_emb = self.conditioner(t, cosmo)  # (hidden_size,)

        # ---- 2. Embed spatial data ----
        using_cond = self.enable_cond_image and cond is not None
        token_list = []

        if using_cond:
            cond_tokens = self.cond_embedder(cond)                # (N_cond, D)
            cond_tokens = cond_tokens + self.cond_pos_embed.value
            token_list.append(cond_tokens)
            num_cond_patches = cond_tokens.shape[0]

        field_tokens = self.x_embedder(field)                     # (N_field, D)
        field_tokens = field_tokens + self.pos_embed.value
        token_list.append(field_tokens)

        x = jnp.concatenate(token_list, axis=0)  # (S, D)

        # ---- 3. Select RoPE ----
        feat_rope = self.feat_rope_cond if using_cond else self.feat_rope_nocond

        # ---- 4. Transformer backbone (each block modulated by c_emb) ----
        for block in self.blocks:
            x = block(x, c_emb, feat_rope=feat_rope, train=train)

        # ---- 5. Field prediction (last N_field tokens) ----
        if using_cond:
            field_tokens_out = x[num_cond_patches:]
        else:
            field_tokens_out = x

        field_pred = self.field_head(field_tokens_out, c_emb)
        field_pred = self.unpatchify(field_pred, self.patch_size)

        # ---- 6. Cosmo prediction (cross-attention over ALL tokens) ----
        cosmo_pred = self.cosmo_head(x)

        return field_pred, cosmo_pred


# =============================================================================
# Model variant constructors
# =============================================================================

def JADE_FUSE_B_16_mixed(
    rngs,
    cosmo_dim=6,
    enable_cond_image=True,
    cond_channels=5,
    **kwargs,
):
    """
    Base JADE_FUSE model.
    patch_size=8 for target field (256 tokens at 128×128).
    cond_patch_size=16 for conditioning image (64 tokens at 128×128).
    Total sequence: 320 tokens (with cond) or 256 tokens (without).
    """
    return JADE_FUSE(
        depth=12,
        hidden_size=768,
        num_heads=12,
        bottleneck_dim=128,
        cosmo_dim=cosmo_dim,
        enable_cond_image=enable_cond_image,
        cond_channels=cond_channels,
        patch_size=8,
        cond_patch_size=16,
        rngs=rngs,
        **kwargs,
    )
