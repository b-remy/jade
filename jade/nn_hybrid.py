import jax
import jax.numpy as jnp
from flax import nnx

import numpy as np

from einops import repeat, rearrange

import math

# =============================================================================
# Unchanged utility modules (BottleneckPatchEmbed, RMSNorm, etc.)
# =============================================================================

class BottleneckPatchEmbed(nnx.Module):
    """Image to Patch Embedding with bottleneck."""
    
    def __init__(
        self, 
        img_size=256, 
        patch_size=16, 
        in_chans=1, 
        pca_dim=768, 
        embed_dim=768, 
        bias=True, 
        rngs=None
    ):
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj1 = nnx.Conv(
            in_features=in_chans,
            out_features=pca_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            use_bias=False,
            rngs=rngs
        )
        
        self.proj2 = nnx.Conv(
            in_features=pca_dim,
            out_features=embed_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=bias,
            rngs=rngs
        )
    
    def __call__(self, x):
        H, W, C = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        x = self.proj1(x)
        x = self.proj2(x)
        
        H_out, W_out, C_out = x.shape
        x = x.reshape(H_out * W_out, C_out)
        
        return x
    
class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6, rngs: nnx.Rngs = None):
        self.variance_epsilon = eps
        self.weight = nnx.Param(jnp.ones(hidden_size))
    
    def __call__(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.variance_epsilon)
        return (self.weight.value * hidden_states).astype(input_dtype)
    
def rotate_half(x):
    """Rotate half the hidden dims."""
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x[..., 0], x[..., 1]
    x = jnp.stack([-x2, x1], axis=-1)
    return rearrange(x, '... d r -> ... (d r)')


def scaled_dot_product_attention(query, key, value, dropout_p=0.0, dropout_key=None, attn_mask=None):
    """
    Scaled dot-product attention WITHOUT batch dimension.
    query/key/value: (H, L/S, D)
    dropout_key: if not None and dropout_p > 0, apply dropout to attention weights
    attn_mask: optional additive (L, S) mask (broadcasts over heads); -inf
        entries are zeroed out by the softmax. Used to block θ→κ in stage 2.
    """
    scale_factor = 1.0 / math.sqrt(query.shape[-1])

    query_f32 = query.astype(jnp.float32)
    key_f32 = key.astype(jnp.float32)

    attn_weight = jnp.einsum('hld,hsd->hls', query_f32, key_f32) * scale_factor
    if attn_mask is not None:
        attn_weight = attn_weight + attn_mask
    attn_weight = jax.nn.softmax(attn_weight, axis=-1)
    
    if dropout_key is not None and dropout_p > 0.0:
        keep_prob = 1.0 - dropout_p
        mask = jax.random.bernoulli(dropout_key, keep_prob, attn_weight.shape)
        attn_weight = jnp.where(mask, attn_weight / keep_prob, 0.0)
    
    output = jnp.einsum('hls,hsd->hld', attn_weight, value)
    return output


class LabelEmbedder(nnx.Module):
    """Embeds class labels into vector representations."""
    
    def __init__(self, num_classes, hidden_size, rngs=None):
        self.num_classes = num_classes
        self.embedding_table = nnx.Embed(num_classes + 1, hidden_size, rngs=rngs)
    
    def __call__(self, labels):
        return self.embedding_table(labels)
    

class Attention(nnx.Module):
    """Multi-head attention with RMSNorm and RoPE support.

    If ``num_theta_tokens > 0``, the QKV projection is split per-modality:
    the first ``num_theta_tokens`` positions are projected by ``qkv_theta`` and
    the remaining positions by ``qkv_kg``. The two projections are initialized
    independently but produce the same output as a single shared ``qkv`` when
    their weights are equal — used for two-stage training (stage 2 freezes
    ``qkv_kg`` so cosmology gradients only flow through ``qkv_theta``).
    """

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True,
                 attn_drop=0., proj_drop=0., num_theta_tokens=0, rngs=None):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_theta_tokens = num_theta_tokens
        self.split_qkv = num_theta_tokens > 0

        if self.split_qkv:
            rng_keys = jax.random.split(rngs(), 2)
            rngs_theta = nnx.Rngs(rng_keys[0])
            rngs_kg = nnx.Rngs(rng_keys[1])
            self.qkv_theta = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, rngs=rngs_theta)
            self.qkv_kg = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, rngs=rngs_kg)
        else:
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

    def __call__(self, x, rope, train=False, key=None, attn_mask=None):
        """
        Args:
            x: (N, C)
            rope: RoPE function
            train: training mode
            key: JAX random key for dropout (None = no dropout)
            attn_mask: optional additive (N, N) mask passed to softmax
        """
        N, C = x.shape

        if self.split_qkv:
            n_theta = self.num_theta_tokens
            qkv_t = self.qkv_theta(x[:n_theta])
            qkv_kg = self.qkv_kg(x[n_theta:])
            qkv = jnp.concatenate([qkv_t, qkv_kg], axis=0)
        else:
            qkv = self.qkv(x)
        qkv = rearrange(qkv, 'N (three H D) -> three H N D',
                       three=3, H=self.num_heads, D=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        # Split key for attn dropout and proj dropout
        attn_key = None
        proj_key = None
        if train and key is not None:
            attn_key, proj_key = jax.random.split(key)

        x = scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_p if train else 0.0,
            dropout_key=attn_key,
            attn_mask=attn_mask,
        )
        
        x = rearrange(x, 'H N D -> N (H D)')
        x = self.proj(x)
        
        # Projection dropout
        if train and proj_key is not None and self.proj_drop_p > 0:
            keep_prob = 1.0 - self.proj_drop_p
            mask = jax.random.bernoulli(proj_key, keep_prob, x.shape)
            x = jnp.where(mask, x / keep_prob, 0.0)
        
        return x


class TimestepEmbedder(nnx.Module):
    """Embeds scalar timesteps into vector representations."""
    
    def __init__(self, hidden_size, frequency_embedding_size=256, rngs=None):
        self.frequency_embedding_size = frequency_embedding_size
        self.linear1 = nnx.Linear(frequency_embedding_size, hidden_size, use_bias=True, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_size, hidden_size, use_bias=True, rngs=rngs)
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = jnp.exp(
            -math.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half
        )
        args = t[:, None] * freqs[None, :]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        return embedding
    
    def __call__(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.linear1(t_freq)
        t_emb = nnx.silu(t_emb)
        t_emb = self.linear2(t_emb)
        return t_emb
    

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class SwiGLUFFN(nnx.Module):
    """SwiGLU Feed-Forward Network."""
    
    def __init__(self, dim, hidden_dim, drop=0.0, bias=True, rngs=None):
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nnx.Linear(dim, 2 * hidden_dim, use_bias=bias, rngs=rngs)
        self.w3 = nnx.Linear(hidden_dim, dim, use_bias=bias, rngs=rngs)
        self.drop_rate = drop
    
    def __call__(self, x, train=False, key=None):
        x12 = self.w12(x)
        x1, x2 = jnp.split(x12, 2, axis=-1)
        hidden = nnx.silu(x1) * x2
        
        if train and key is not None and self.drop_rate > 0:
            keep_prob = 1.0 - self.drop_rate
            mask = jax.random.bernoulli(key, keep_prob, hidden.shape)
            hidden = jnp.where(mask, hidden / keep_prob, 0.0)
        
        return self.w3(hidden)
    

def modulate(x, shift, scale):
    """Apply adaptive modulation to normalized features."""
    return x * (1 + jnp.expand_dims(scale, -2)) + jnp.expand_dims(shift, -2)


class JiTBlock(nnx.Module):
    """JiT Transformer Block with Adaptive Layer Normalization."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,
                 attn_drop=0.0, proj_drop=0.0, num_theta_tokens=0, rngs=None):
        self.norm1 = RMSNorm(hidden_size, eps=1e-6, rngs=rngs)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6, rngs=rngs)

        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
            attn_drop=attn_drop, proj_drop=proj_drop,
            num_theta_tokens=num_theta_tokens, rngs=rngs
        )
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop, rngs=rngs)
        
        self.ada_linear = nnx.Linear(hidden_size, 6 * hidden_size, rngs=rngs)
    
    def __call__(self, x, c, feat_rope=None, train=False, key=None, attn_mask=None):
        ada = self.ada_linear(nnx.silu(c))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(ada, 6, axis=-1)

        # Split key for attn and mlp dropout
        attn_key = None
        mlp_key = None
        if train and key is not None:
            attn_key, mlp_key = jax.random.split(key)

        x = x + jnp.expand_dims(gate_msa, -2) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            rope=feat_rope, train=train, key=attn_key, attn_mask=attn_mask,
        )
        
        x = x + jnp.expand_dims(gate_mlp, -2) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp), train=train, key=mlp_key
        )
        
        return x
    

class FinalLayer(nnx.Module):
    """The final layer of JiT."""
    
    def __init__(self, hidden_size, patch_size, out_channels, rngs=None):
        self.norm_final = RMSNorm(hidden_size, rngs=rngs)
        self.linear = nnx.Linear(
            hidden_size, patch_size * patch_size * out_channels,
            use_bias=True, rngs=rngs
        )
        self.ada_linear = nnx.Linear(hidden_size, 2 * hidden_size, rngs=rngs)
    
    def __call__(self, x, c):
        ada = self.ada_linear(nnx.silu(c))
        shift, scale = jnp.split(ada, 2, axis=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# =============================================================================
# NEW: MultiResolutionRoPE (replaces VisionRotaryEmbeddingFast in JADE)
# =============================================================================

class MultiResolutionRoPE(nnx.Module):
    """
    RoPE that handles tokens at multiple spatial resolutions.
    
    Builds a unified cos/sin frequency table for:
      1. Non-spatial tokens (cosmo) → identity rotation (cos=1, sin=0)
      2. Conditioning patches (coarse grid) → positions at patch centers in field coordinates
      3. Target field patches (fine grid) → standard integer grid positions
    
    This ensures that attention between conditioning and target patches
    is informed by their true spatial correspondence.
    """
    
    def __init__(
        self,
        dim,
        field_seq_len,
        cond_seq_len=0,
        num_nonspatial_tokens=0,
        theta=10000,
        rngs=None
    ):
        """
        Args:
            dim: half of head_dim (RoPE dimension per spatial axis)
            field_seq_len: side length of target field patch grid (e.g. 16)
            cond_seq_len: side length of conditioning patch grid (e.g. 8), 0 to disable
            num_nonspatial_tokens: number of tokens with no spatial position (cosmo tokens)
            theta: RoPE base frequency
        """
        # Base frequencies: same for both resolutions to ensure consistency
        freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
        
        # --- Target field: standard integer positions on [0, field_seq_len) ---
        t_field = jnp.arange(field_seq_len, dtype=jnp.float32)
        freqs_field = jnp.einsum('t,f->tf', t_field, freqs)
        freqs_field = repeat(freqs_field, 't f -> t (f r)', r=2)
        
        # Build 2D frequencies for field
        fh = jnp.broadcast_to(freqs_field[:, None, :], (field_seq_len, field_seq_len, freqs_field.shape[-1]))
        fw = jnp.broadcast_to(freqs_field[None, :, :], (field_seq_len, field_seq_len, freqs_field.shape[-1]))
        freqs_2d_field = jnp.concatenate([fh, fw], axis=-1)
        freqs_flat_field = freqs_2d_field.reshape(-1, freqs_2d_field.shape[-1])
        
        D = freqs_flat_field.shape[-1]  # = head_dim
        
        # Collect parts in order: [nonspatial | cond | field]
        parts_cos = []
        parts_sin = []
        
        # 1) Non-spatial tokens (cosmo): identity rotation
        if num_nonspatial_tokens > 0:
            parts_cos.append(jnp.ones((num_nonspatial_tokens, D)))
            parts_sin.append(jnp.zeros((num_nonspatial_tokens, D)))
        
        # 2) Conditioning patches: centered positions in field coordinate space
        if cond_seq_len > 0:
            scale = field_seq_len / cond_seq_len  # e.g. 16/8 = 2.0
            
            # Each cond patch (i) covers field positions [i*scale, (i+1)*scale).
            # We place RoPE at the center: i*scale + scale/2 - 0.5
            # This maps cond patch centers to field coordinate space.
            # For scale=2: cond patch 0 -> field pos 0.5, patch 1 -> 2.5, ...
            # For scale=1 (same resolution): cond patch i -> field pos i (exact match)
            t_cond = jnp.arange(cond_seq_len, dtype=jnp.float32) * scale + (scale - 1) / 2.0
            
            freqs_cond = jnp.einsum('t,f->tf', t_cond, freqs)
            freqs_cond = repeat(freqs_cond, 't f -> t (f r)', r=2)
            
            ch = jnp.broadcast_to(freqs_cond[:, None, :], (cond_seq_len, cond_seq_len, freqs_cond.shape[-1]))
            cw = jnp.broadcast_to(freqs_cond[None, :, :], (cond_seq_len, cond_seq_len, freqs_cond.shape[-1]))
            freqs_2d_cond = jnp.concatenate([ch, cw], axis=-1)
            freqs_flat_cond = freqs_2d_cond.reshape(-1, freqs_2d_cond.shape[-1])
            
            parts_cos.append(jnp.cos(freqs_flat_cond))
            parts_sin.append(jnp.sin(freqs_flat_cond))
        
        # 3) Target field patches
        parts_cos.append(jnp.cos(freqs_flat_field))
        parts_sin.append(jnp.sin(freqs_flat_field))
        
        self.freqs_cos = jnp.concatenate(parts_cos, axis=0)
        self.freqs_sin = jnp.concatenate(parts_sin, axis=0)
    
    def __call__(self, t):
        """Apply rotary embeddings. t: (..., seq_len, head_dim)"""
        seq_len = t.shape[-2]
        freqs_cos = self.freqs_cos[:seq_len]
        freqs_sin = self.freqs_sin[:seq_len]
        return t * freqs_cos + rotate_half(t) * freqs_sin


# =============================================================================
# Keep VisionRotaryEmbeddingFast for JiT (unchanged)
# =============================================================================

class VisionRotaryEmbeddingFast(nnx.Module):
    """Fast vision rotary positional embeddings (used by JiT, not JADE)."""
    
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
            cos_pad = jnp.ones((num_cls_token, D))
            sin_pad = jnp.zeros((num_cls_token, D))
            self.freqs_cos = jnp.concatenate([cos_pad, cos_img], axis=0)
            self.freqs_sin = jnp.concatenate([sin_pad, sin_img], axis=0)
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


# =============================================================================
# Cosmology modules (unchanged)
# =============================================================================

class CosmologyEmbedder(nnx.Module):
    """Embeds cosmological parameters using Token Inflation (N tokens)."""
    
    def __init__(self, cosmo_dim, hidden_size, num_tokens=16, rngs=None):
        self.cosmo_dim = cosmo_dim
        self.num_tokens = num_tokens
        self.proj = nnx.Linear(cosmo_dim, num_tokens * hidden_size, rngs=rngs)
        self.token_pos_embed = nnx.Param(
            jax.random.normal(rngs(), (num_tokens, hidden_size)) * 0.02
        )
    
    def __call__(self, cosmo):
        tokens = self.proj(cosmo)
        tokens = tokens.reshape(self.num_tokens, -1)
        return tokens + self.token_pos_embed.value


class CosmologyPredictor(nnx.Module):
    """Predicts denoised cosmology by pooling information from N tokens."""
    
    def __init__(self, hidden_size, cosmo_dim, num_tokens=16, rngs=None):
        self.cosmo_dim = cosmo_dim
        self.num_tokens = num_tokens
        self.norm = RMSNorm(hidden_size, rngs=rngs)
        self.proj = nnx.Linear(hidden_size, cosmo_dim, rngs=rngs)
    
    def __call__(self, cosmo_tokens):
        pooled = jnp.mean(cosmo_tokens, axis=0)
        pooled = self.norm(pooled)
        return self.proj(pooled)


# =============================================================================
# JADE with MultiResolutionRoPE and Staged Conditioning Injection
# =============================================================================

class JADE(nnx.Module):
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
        num_cosmo_tokens=16,
        cond_channels=5,
        enable_cond_image=True,
        cond_start=None,        # ← NEW: block index where conditioning is injected
        split_qkv=False,        # ← NEW: per-modality QKV (stage 2)
        mask_theta_to_field=False,  # ← NEW: block θ→κ in attention (stage 2 obs-only)
        rngs=None
    ):
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.cond_patch_size = cond_patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cosmo_dim = cosmo_dim
        self.num_cosmo_tokens = num_cosmo_tokens
        self.cond_channels = cond_channels
        self.enable_cond_image = enable_cond_image
        self.split_qkv = split_qkv
        self.mask_theta_to_field = mask_theta_to_field

        # Default: inject conditioning at 1/3 of depth (early layers process cosmo+field only)
        self.cond_start = cond_start if cond_start is not None else depth // 3
        
        # --- RNG management ---
        num_components = depth + 7
        if enable_cond_image:
            num_components += 1  # only cond_embedder, no cond_pos_embed
        
        rng_keys = jax.random.split(rngs(), num_components)
        
        idx = 0
        rngs_t_embedder = nnx.Rngs(rng_keys[idx]); idx += 1
        rngs_x_embedder = nnx.Rngs(rng_keys[idx]); idx += 1
        rngs_cosmo_embedder = nnx.Rngs(rng_keys[idx]); idx += 1
        
        if enable_cond_image:
            rngs_cond_embedder = nnx.Rngs(rng_keys[idx]); idx += 1
        
        rngs_rope = nnx.Rngs(rng_keys[idx]); idx += 1
        rngs_rope_nocond = nnx.Rngs(rng_keys[idx]); idx += 1
        rngs_final = nnx.Rngs(rng_keys[idx]); idx += 1
        rngs_cosmo_head = nnx.Rngs(rng_keys[idx]); idx += 1
        rngs_blocks = [nnx.Rngs(rng_keys[idx + i]) for i in range(depth)]
        
        # --- Embedders ---
        self.t_embedder = TimestepEmbedder(hidden_size, rngs=rngs_t_embedder)
        
        self.cosmo_embedder = CosmologyEmbedder(
            cosmo_dim, hidden_size, num_tokens=num_cosmo_tokens, rngs=rngs_cosmo_embedder
        )

        self.x_embedder = BottleneckPatchEmbed(
            input_size, patch_size, in_channels, bottleneck_dim, hidden_size,
            bias=True, rngs=rngs_x_embedder
        )
        
        if self.enable_cond_image:
            self.cond_embedder = BottleneckPatchEmbed(
                input_size, cond_patch_size, cond_channels, bottleneck_dim, hidden_size,
                bias=True, rngs=rngs_cond_embedder
            )
        
        # --- Positional embeddings for target field ---
        num_field_patches = (input_size // patch_size) ** 2
        pos_embed = get_2d_sincos_pos_embed(
            hidden_size, int(num_field_patches ** 0.5),
            cls_token=False, extra_tokens=0
        )
        self.pos_embed = nnx.Param(jnp.array(pos_embed, dtype=jnp.float32))
        
        # --- RoPE: MultiResolutionRoPE ---
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len_field = input_size // patch_size       # e.g. 16
        hw_seq_len_cond = input_size // cond_patch_size   # e.g. 8
        
        # RoPE for blocks BEFORE cond_start: [cosmo | field]
        self.feat_rope_nocond = MultiResolutionRoPE(
            dim=half_head_dim,
            field_seq_len=hw_seq_len_field,
            cond_seq_len=0,                            # No conditioning yet
            num_nonspatial_tokens=num_cosmo_tokens,
            rngs=rngs_rope_nocond
        )

        # RoPE for blocks FROM cond_start onward: [cosmo | cond | field]
        if self.enable_cond_image:
            self.feat_rope_cond = MultiResolutionRoPE(
                dim=half_head_dim,
                field_seq_len=hw_seq_len_field,        # 16
                cond_seq_len=hw_seq_len_cond,          # 8 (gets proper 2D positions!)
                num_nonspatial_tokens=num_cosmo_tokens, # 16 (identity rotation)
                rngs=rngs_rope
            )

        # --- Transformer blocks ---
        # When split_qkv is on, the first num_cosmo_tokens positions of each
        # block input are projected by qkv_theta and the rest by qkv_kg.
        num_theta_tokens = num_cosmo_tokens if split_qkv else 0
        self.blocks = nnx.List([
            JiTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                num_theta_tokens=num_theta_tokens,
                rngs=rngs_blocks[i]
            )
            for i in range(depth)
        ])
        
        # --- Output heads ---
        self.field_head = FinalLayer(hidden_size, patch_size, self.out_channels, rngs=rngs_final)
        self.cosmo_head = CosmologyPredictor(
            hidden_size, cosmo_dim, num_tokens=num_cosmo_tokens, rngs=rngs_cosmo_head
        )

        # --- Static θ→κ attention masks (stage-2 Level A "obs-only") ---
        # The sequence layout is:
        #   pre-cond layers  : [cosmo (n_θ) | field (n_κ)]
        #   post-cond layers : [cosmo (n_θ) | cond (n_γ) | field (n_κ)]
        # In both cases field tokens are the LAST n_κ positions, so the mask
        # is built as: rows [0:n_θ] (θ queries) × cols [N - n_κ : N] (field
        # keys) → -inf, everything else 0. The softmax then drops θ→κ scores.
        self._theta_mask_nocond = None
        self._theta_mask_cond = None
        if mask_theta_to_field:
            n_theta = num_cosmo_tokens
            n_field = num_field_patches

            N_nc = n_theta + n_field
            mask_nc = jnp.zeros((N_nc, N_nc), dtype=jnp.float32)
            mask_nc = mask_nc.at[:n_theta, n_theta:].set(-jnp.inf)
            self._theta_mask_nocond = nnx.data(mask_nc)

            if enable_cond_image:
                n_cond = (input_size // cond_patch_size) ** 2
                N_c = n_theta + n_cond + n_field
                mask_c = jnp.zeros((N_c, N_c), dtype=jnp.float32)
                mask_c = mask_c.at[:n_theta, N_c - n_field:].set(-jnp.inf)
                self._theta_mask_cond = nnx.data(mask_c)

        self.initialize_weights()

    def initialize_weights(self):
        def init_linear_xavier(module):
            if isinstance(module, nnx.Linear):
                w = module.kernel.value
                w_flat = w.reshape(w.shape[0], -1)
                w_init = jax.nn.initializers.xavier_uniform()(
                    jax.random.PRNGKey(0), w_flat.shape
                )
                module.kernel.value = w_init.reshape(w.shape)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.value = jnp.zeros_like(module.bias.value)
        
        init_linear_xavier(self.t_embedder.linear1)
        init_linear_xavier(self.t_embedder.linear2)
        init_linear_xavier(self.cosmo_embedder.proj)
        init_linear_xavier(self.cosmo_head.proj)
        
        for block in self.blocks:
            if block.attn.split_qkv:
                init_linear_xavier(block.attn.qkv_theta)
                init_linear_xavier(block.attn.qkv_kg)
            else:
                init_linear_xavier(block.attn.qkv)
            init_linear_xavier(block.attn.proj)
            init_linear_xavier(block.mlp.w12)
            init_linear_xavier(block.mlp.w3)
            init_linear_xavier(block.ada_linear)
        
        init_linear_xavier(self.field_head.linear)
        init_linear_xavier(self.field_head.ada_linear)
        
        # Patch embedding init
        w1 = self.x_embedder.proj1.kernel.value
        w1_init = jax.nn.initializers.xavier_uniform()(jax.random.PRNGKey(1), w1.reshape(w1.shape[0], -1).shape)
        self.x_embedder.proj1.kernel.value = w1_init.reshape(w1.shape)
        
        w2 = self.x_embedder.proj2.kernel.value
        w2_init = jax.nn.initializers.xavier_uniform()(jax.random.PRNGKey(2), w2.reshape(w2.shape[0], -1).shape)
        self.x_embedder.proj2.kernel.value = w2_init.reshape(w2.shape)
        self.x_embedder.proj2.bias.value = jnp.zeros_like(self.x_embedder.proj2.bias.value)
        
        if self.enable_cond_image:
            w1_c = self.cond_embedder.proj1.kernel.value
            w1_c_init = jax.nn.initializers.xavier_uniform()(jax.random.PRNGKey(10), w1_c.reshape(w1_c.shape[0], -1).shape)
            self.cond_embedder.proj1.kernel.value = w1_c_init.reshape(w1_c.shape)
            w2_c = self.cond_embedder.proj2.kernel.value
            w2_c_init = jax.nn.initializers.xavier_uniform()(jax.random.PRNGKey(11), w2_c.reshape(w2_c.shape[0], -1).shape)
            self.cond_embedder.proj2.kernel.value = w2_c_init.reshape(w2_c.shape)
            self.cond_embedder.proj2.bias.value = jnp.zeros_like(self.cond_embedder.proj2.bias.value)
        
        # Timestep embedder small init
        self.t_embedder.linear1.kernel.value = jax.random.normal(jax.random.PRNGKey(3), self.t_embedder.linear1.kernel.value.shape) * 0.02
        self.t_embedder.linear2.kernel.value = jax.random.normal(jax.random.PRNGKey(4), self.t_embedder.linear2.kernel.value.shape) * 0.02
        
        # Cosmo embedder init
        self.cosmo_embedder.proj.kernel.value = jax.random.normal(jax.random.PRNGKey(5), self.cosmo_embedder.proj.kernel.value.shape) * 0.02
        self.cosmo_embedder.proj.bias.value = jnp.zeros_like(self.cosmo_embedder.proj.bias.value)
        self.cosmo_embedder.token_pos_embed.value = jax.random.normal(jax.random.PRNGKey(6), self.cosmo_embedder.token_pos_embed.value.shape) * 0.02
        
        # Zero-init for AdaLN and output heads (diffusion standard)
        for block in self.blocks:
            block.ada_linear.kernel.value = jnp.zeros_like(block.ada_linear.kernel.value)
            block.ada_linear.bias.value = jnp.zeros_like(block.ada_linear.bias.value)
        
        self.field_head.ada_linear.kernel.value = jnp.zeros_like(self.field_head.ada_linear.kernel.value)
        self.field_head.ada_linear.bias.value = jnp.zeros_like(self.field_head.ada_linear.bias.value)
        self.field_head.linear.kernel.value = jnp.zeros_like(self.field_head.linear.kernel.value)
        self.field_head.linear.bias.value = jnp.zeros_like(self.field_head.linear.bias.value)
        
        self.cosmo_head.proj.kernel.value = jnp.zeros_like(self.cosmo_head.proj.kernel.value)
        self.cosmo_head.proj.bias.value = jnp.zeros_like(self.cosmo_head.proj.bias.value)

    def unpatchify(self, x, p):
        c = self.out_channels
        N = x.shape[0]
        h = w = int(N ** 0.5)
        assert h * w == N
        x = x.reshape(h, w, p, p, c)
        x = jnp.einsum('hwpqc->hpwqc', x)
        return x.reshape(h * p, w * p, c)
    
    def __call__(self, field, cosmo, t, cond=None, train=False, key=None):
        """
        Forward pass (no batch dim — use vmap for batching).
        
        Args:
            field: target field to denoise (H, W, C)
            cosmo: noisy cosmological parameters (cosmo_dim,)
            t: diffusion timestep, scalar or (1,)
            cond: conditioning field (H, W, C_cond) or None
            train: training mode
            key: JAX random key for dropout (None = no dropout)
        
        Returns:
            field_pred: denoised field prediction (H, W, C)
            cosmo_pred: denoised cosmology prediction (cosmo_dim,)
        """
        if jnp.ndim(t) == 0:
            t = jnp.array([t])
        
        # --- Timestep conditioning for AdaLN ---
        t_emb = self.t_embedder(t)[0]      # (hidden_size,)
        c = t_emb
        
        # --- Embed noisy cosmology into N tokens ---
        cosmo_tokens = self.cosmo_embedder(cosmo)   # (num_cosmo_tokens, hidden_size)
        
        # --- Embed target field ---
        field_tokens = self.x_embedder(field)        # (num_field_patches, hidden_size)
        field_tokens = field_tokens + self.pos_embed.value
        
        # --- Optionally prepare conditioning (but don't concatenate yet) ---
        using_cond = self.enable_cond_image and cond is not None
        if using_cond:
            cond_tokens = self.cond_embedder(cond)   # (num_cond_patches, hidden_size)
            num_cond_patches = cond_tokens.shape[0]
        
        # --- Pre-split keys for all blocks (one per block) ---
        depth = len(self.blocks)
        if train and key is not None:
            block_keys = jax.random.split(key, depth)
        else:
            block_keys = [None] * depth
        
        # --- Initial sequence: [cosmo | field] only ---
        x = jnp.concatenate([cosmo_tokens, field_tokens], axis=0)
        
        # --- Forward through transformer blocks ---
        for i, block in enumerate(self.blocks):
            
            # At cond_start: inject conditioning tokens between cosmo and field
            if using_cond and i == self.cond_start:
                # Split current sequence back into cosmo and field parts
                cosmo_part = x[:self.num_cosmo_tokens]
                field_part = x[self.num_cosmo_tokens:]
                
                # Reassemble with conditioning in the middle
                # Order: [cosmo | cond | field]
                x = jnp.concatenate([cosmo_part, cond_tokens, field_part], axis=0)
            
            # Select RoPE based on whether conditioning is present
            if using_cond and i >= self.cond_start:
                rope = self.feat_rope_cond
                attn_mask = self._theta_mask_cond
            else:
                rope = self.feat_rope_nocond
                attn_mask = self._theta_mask_nocond

            x = block(x, c, feat_rope=rope, train=train, key=block_keys[i],
                      attn_mask=attn_mask)
        
        # --- Split output tokens ---
        idx = 0
        cosmo_tokens_out = x[idx:idx + self.num_cosmo_tokens]
        idx += self.num_cosmo_tokens
        
        if using_cond:
            idx += num_cond_patches   # Skip conditioning tokens
        
        field_tokens_out = x[idx:]
        
        # --- Prediction heads ---
        cosmo_pred = self.cosmo_head(cosmo_tokens_out)
        field_tokens_pred = self.field_head(field_tokens_out, c)
        field_pred = self.unpatchify(field_tokens_pred, self.patch_size)
        
        return field_pred, cosmo_pred


# =============================================================================
# Model constructor
# =============================================================================

def convert_state_split_qkv(state):
    """Convert a shared-QKV stage-1 state to the split-QKV stage-2 layout.

    For every transformer block, replaces ``attn.qkv`` with two independent
    copies, ``attn.qkv_theta`` and ``attn.qkv_kg``, both initialised from the
    same shared weights. At t=0 of stage 2 this makes the split model
    bit-identical to the stage-1 model on the forward pass; freezing
    ``qkv_kg`` then keeps the κγ-token projections fixed while ``qkv_theta``
    is updated.

    Accepts state either at the JADE level or wrapped under a ``model`` key
    by the Denoiser.

    Args:
        state: nnx state from a stage-1 (shared-QKV) JADE model.

    Returns:
        A deep-copied state with ``qkv`` replaced by ``qkv_theta`` and
        ``qkv_kg`` in every transformer block. Non-attention entries pass
        through unchanged.
    """
    import copy as _copy

    new_state = _copy.deepcopy(state)
    blocks_state = (
        new_state['model']['blocks'] if 'model' in new_state else new_state['blocks']
    )

    for k in list(blocks_state.keys()):
        attn = blocks_state[k]['attn']
        if 'qkv' not in attn:
            continue
        qkv = attn['qkv']
        attn['qkv_theta'] = _copy.deepcopy(qkv)
        attn['qkv_kg'] = _copy.deepcopy(qkv)
        del attn['qkv']

    return new_state


def JADE_B_16(rngs, cosmo_dim=6, patch_size=8, enable_cond_image=True, cond_channels=5,
                    num_cosmo_tokens=16, cond_patch_size=16, cond_start=None,
                    split_qkv=False, mask_theta_to_field=False, **kwargs):
    """Base model with mixed patch sizes: 8 for target, 16 for conditioning."""
    return JADE(
        depth=12,
        hidden_size=768,
        num_heads=12,
        bottleneck_dim=128,
        cosmo_dim=cosmo_dim,
        enable_cond_image=enable_cond_image,
        cond_channels=cond_channels,
        patch_size=patch_size,
        cond_patch_size=cond_patch_size,
        num_cosmo_tokens=num_cosmo_tokens,
        cond_start=cond_start,                  # defaults to depth//3 = 4
        split_qkv=split_qkv,                    # per-modality QKV for stage 2
        mask_theta_to_field=mask_theta_to_field,  # block θ→κ in attention
        rngs=rngs,
        **kwargs
    )


# =============================================================================
# JiT (unchanged, kept for reference)
# =============================================================================

class JiT(nnx.Module):
    """Just image Transformer."""
    
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        rngs=None
    ):
        rng_keys = jax.random.split(rngs(), 6 + depth)
        rngs_t_embedder = nnx.Rngs(rng_keys[0])
        rngs_y_embedder = nnx.Rngs(rng_keys[1])
        rngs_x_embedder = nnx.Rngs(rng_keys[2])
        rngs_rope = nnx.Rngs(rng_keys[3])
        rngs_final = nnx.Rngs(rng_keys[4])
        rngs_blocks = [nnx.Rngs(rng_keys[5 + i]) for i in range(depth)]

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes
        
        self.t_embedder = TimestepEmbedder(hidden_size, rngs=rngs_t_embedder)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, rngs=rngs_y_embedder)
        
        self.x_embedder = BottleneckPatchEmbed(
            input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True, rngs=rngs_x_embedder
        )
        
        num_patches = self.x_embedder.num_patches
        pos_embed = get_2d_sincos_pos_embed(
            hidden_size, int(num_patches ** 0.5), cls_token=False, extra_tokens=0
        )
        self.pos_embed = nnx.Param(jnp.array(pos_embed, dtype=jnp.float32))
        
        if self.in_context_len > 0:
            self.in_context_posemb = nnx.Param(
                jax.random.normal(rngs(), (self.in_context_len, hidden_size)) * 0.02
            )
        
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=0, rngs=rngs_rope
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=in_context_len, rngs=rngs_rope
        )
        
        self.blocks = nnx.List([
            JiTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                rngs=rngs_blocks[i]
            )
            for i in range(depth)
        ])
        
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, rngs=rngs_final)
       
    def unpatchify(self, x, p):
        c = self.out_channels
        N = x.shape[0]
        h = w = int(N ** 0.5)
        assert h * w == N
        x = x.reshape(h, w, p, p, c)
        x = jnp.einsum('hwpqc->hpwqc', x)
        imgs = x.reshape(h * p, w * p, c)
        return imgs
    
    def __call__(self, x, t, y, train=False):
        if jnp.ndim(t) == 0:
            t = jnp.array([t])
        if jnp.ndim(y) == 0:
            y = jnp.array([y])
        
        t_emb = self.t_embedder(t)[0]
        y_emb = self.y_embedder(y)[0]
        c = t_emb + y_emb
        
        x = self.x_embedder(x)
        x = x + self.pos_embed.value
        
        for i, block in enumerate(self.blocks):
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = jnp.tile(
                    jnp.expand_dims(y_emb, 0), (self.in_context_len, 1)
                )
                in_context_tokens = in_context_tokens + self.in_context_posemb.value
                x = jnp.concatenate([in_context_tokens, x], axis=0)
            
            rope = self.feat_rope if i < self.in_context_start else self.feat_rope_incontext
            x = block(x, c, feat_rope=rope, train=train)
        
        x = x[self.in_context_len:]
        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size)
        return output


def JiT_B_16(rngs, **kwargs):
    return JiT(
        depth=12, hidden_size=768, num_heads=12,
        bottleneck_dim=128, num_classes=1, in_context_len=0, in_context_start=0,
        patch_size=16, rngs=rngs, **kwargs
    )