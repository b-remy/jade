import jax
import jax.numpy as jnp
from flax import nnx

import numpy as np

from einops import repeat, rearrange

import math

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
        """
        Args:
            img_size: input image size
            patch_size: patch size
            in_chans: number of input channels
            pca_dim: intermediate bottleneck dimension
            embed_dim: output embedding dimension
            bias: whether to use bias in proj2
            rngs: random number generator
        """
        # Store image and patch size as tuples
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patchify layer (no bias)
        self.proj1 = nnx.Conv(
            in_features=in_chans,
            out_features=pca_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            use_bias=False,
            rngs=rngs
        )
        
        # Linear predictor (patch to token)
        self.proj2 = nnx.Conv(
            in_features=pca_dim,
            out_features=embed_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=bias,
            rngs=rngs
        )
    
    def __call__(self, x):
        """
        Args:
            x: Input tensor of shape (H, W, C) in JAX/Flax convention
        
        Returns:
            Tensor of shape (num_patches, embed_dim)
        """
        H, W, C = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # Apply convolutions
        x = self.proj1(x)
        x = self.proj2(x)
        
        # Reshape from (H', W', C) to (num_patches, C)
        # where H' = H/patch_size, W' = W/patch_size
        H_out, W_out, C_out = x.shape
        x = x.reshape(H_out * W_out, C_out)
        
        return x
    
class RMSNorm(nnx.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Reference https://arxiv.org/abs/1910.07467
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6, rngs: nnx.Rngs = None):
        """
        Args:
            hidden_size: Size of the hidden dimension
            eps: Small constant for numerical stability
            rngs: Random number generator (not used but kept for API consistency)
        """
        self.variance_epsilon = eps
        self.weight = nnx.Param(jnp.ones(hidden_size))
    
    def __call__(self, hidden_states):
        """
        Args:
            hidden_states: Input tensor of any shape ending in hidden_size
        
        Returns:
            Normalized tensor with same shape as input
        """
        input_dtype = hidden_states.dtype
        
        # Cast to float32 for numerical stability
        hidden_states = hidden_states.astype(jnp.float32)
        
        # Compute variance over last dimension
        variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        
        # Normalize
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.variance_epsilon)
        
        # Scale by learned weight and cast back to original dtype
        return (self.weight.get_value() * hidden_states).astype(input_dtype)
    
def rotate_half(x):
    """Rotate half the hidden dims."""
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x[..., 0], x[..., 1]
    x = jnp.stack([-x2, x1], axis=-1)
    return rearrange(x, '... d r -> ... (d r)')


def scaled_dot_product_attention(query, key, value, dropout_p=0.0, train=True, rngs=None):
    """
    Scaled dot-product attention WITHOUT batch dimension.
    
    Args:
        query: (H, L, D) - heads, query_len, head_dim
        key: (H, S, D) - heads, key_len, head_dim  
        value: (H, S, D) - heads, value_len, head_dim
        dropout_p: dropout probability
        train: whether in training mode
        rngs: random number generator for dropout
    
    Returns:
        attention output: (H, L, D)
    """
    scale_factor = 1.0 / math.sqrt(query.shape[-1])
    
    # Compute attention weights in float32 for numerical stability
    query_f32 = query.astype(jnp.float32)
    key_f32 = key.astype(jnp.float32)
    
    # attn_weight = Q @ K^T * scale
    # Changed from 'bhld,bhsd->bhls' to 'hld,hsd->hls'
    attn_weight = jnp.einsum('hld,hsd->hls', query_f32, key_f32) * scale_factor
    
    # Softmax
    attn_weight = jax.nn.softmax(attn_weight, axis=-1)
    
    # Dropout
    if train and dropout_p > 0.0:
        keep_prob = 1.0 - dropout_p
        mask = jax.random.bernoulli(rngs(), keep_prob, attn_weight.shape)
        attn_weight = jnp.where(mask, attn_weight / keep_prob, 0.0)
    
    # Apply attention to values
    # Changed from 'bhls,bhsd->bhld' to 'hls,hsd->hld'
    output = jnp.einsum('hls,hsd->hld', attn_weight, value)
    
    return output

class LabelEmbedder(nnx.Module):
    """
    Embeds class labels into vector representations. 
    Also handles label dropout for classifier-free guidance.
    """
    
    def __init__(self, num_classes, hidden_size, rngs=None):
        """
        Args:
            num_classes: number of classes
            hidden_size: dimension of embeddings
            rngs: random number generator
        """
        self.num_classes = num_classes
        # num_classes + 1 to include an extra token for unconditional generation
        self.embedding_table = nnx.Embed(num_classes + 1, hidden_size, rngs=rngs)
    
    def __call__(self, labels):
        """
        Args:
            labels: class labels, can be (N,) or scalar
        
        Returns:
            embeddings of shape (N, hidden_size) or (hidden_size,)
        """
        embeddings = self.embedding_table(labels)
        return embeddings
    
class Attention(nnx.Module):
    """Multi-head attention with RMSNorm and RoPE support."""
    
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, 
                 attn_drop=0., proj_drop=0., rngs=None):
        """
        Args:
            dim: embedding dimension
            num_heads: number of attention heads
            qkv_bias: whether to use bias in qkv projection
            qk_norm: whether to apply RMSNorm to queries and keys
            attn_drop: attention dropout rate
            proj_drop: projection dropout rate
            rngs: random number generator
        """
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # QKV projection
        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, rngs=rngs)
        
        # Query/Key normalization
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, rngs=rngs)
            self.k_norm = RMSNorm(self.head_dim, rngs=rngs)
        else:
            self.q_norm = lambda x: x
            self.k_norm = lambda x: x
        
        # Dropout rates
        self.attn_drop_p = attn_drop
        self.proj_drop_p = proj_drop
        
        # Output projection
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        
        # Store rngs for dropout
        self.rngs = rngs
    
    def __call__(self, x, rope, train=False):
        """
        Args:
            x: input tensor of shape (N, C) - NO batch dimension
            rope: RoPE embedding function
            train: whether in training mode
        
        Returns:
            output tensor of shape (N, C)
        """
        N, C = x.shape
        
        # QKV projection and reshape
        # (N, C) -> (N, 3*C) -> (N, 3, num_heads, head_dim)
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'N (three H D) -> three H N D', 
                       three=3, H=self.num_heads, D=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply normalization to queries and keys
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Apply RoPE (Rotary Position Embedding)
        q = rope(q)
        k = rope(k)
        
        # Scaled dot-product attention
        x = scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.attn_drop_p,
            train=train,
            rngs=self.rngs
        )
        
        # Reshape back: (H, N, D) -> (N, H*D)
        x = rearrange(x, 'H N D -> N (H D)')
        
        # Output projection
        x = self.proj(x)
        
        # Projection dropout
        if train and self.proj_drop_p > 0:
            keep_prob = 1.0 - self.proj_drop_p
            mask = jax.random.bernoulli(self.rngs(), keep_prob, x.shape)
            x = jnp.where(mask, x / keep_prob, 0.0)
        
        return x


class VisionRotaryEmbeddingFast(nnx.Module):
    """Fast vision rotary positional embeddings."""
    
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for='lang',
        theta=10000,
        max_freq=10,
        num_freqs=1,
        num_cls_token=0,  # Add this parameter back!
        rngs=None
    ):
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
        
        # Add support for in-context tokens
        if num_cls_token > 0:
            cos_img = jnp.cos(freqs_flat)
            sin_img = jnp.sin(freqs_flat)
            
            # Prepend zeros for in-context/class tokens
            N_img, D = cos_img.shape
            cos_pad = jnp.ones((num_cls_token, D))
            sin_pad = jnp.zeros((num_cls_token, D))
            
            self.freqs_cos = jnp.concatenate([cos_pad, cos_img], axis=0)
            self.freqs_sin = jnp.concatenate([sin_pad, sin_img], axis=0)
        else:
            self.freqs_cos = jnp.cos(freqs_flat)
            self.freqs_sin = jnp.sin(freqs_flat)
    
    def __call__(self, t):
        """Apply rotary embeddings."""
        seq_len = t.shape[-2]
        
        # Slice to match input sequence length
        if seq_len != self.freqs_cos.shape[0]:
            freqs_cos = self.freqs_cos[:seq_len]
            freqs_sin = self.freqs_sin[:seq_len]
        else:
            freqs_cos = self.freqs_cos
            freqs_sin = self.freqs_sin
        
        return t * freqs_cos + rotate_half(t) * freqs_sin
    

class TimestepEmbedder(nnx.Module):
    """Embeds scalar timesteps into vector representations."""
    
    def __init__(self, hidden_size, frequency_embedding_size=256, rngs=None):
        """
        Args:
            hidden_size: dimension of output embeddings
            frequency_embedding_size: dimension of sinusoidal embeddings
            rngs: random number generator
        """
        self.frequency_embedding_size = frequency_embedding_size
        
        # MLP: frequency_embedding_size -> hidden_size -> hidden_size
        self.linear1 = nnx.Linear(frequency_embedding_size, hidden_size, use_bias=True, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_size, hidden_size, use_bias=True, rngs=rngs)
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            t: a 1-D array of N indices (timesteps), may be fractional
            dim: the dimension of the output
            max_period: controls the minimum frequency of the embeddings
        
        Returns:
            (N, dim) array of positional embeddings
        """
        half = dim // 2
        freqs = jnp.exp(
            -math.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half
        )
        args = t[:, None] * freqs[None, :]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        
        # Handle odd dimensions
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        
        return embedding
    
    def __call__(self, t):
        """
        Args:
            t: timestep array of shape (N,) - batch of timesteps
        
        Returns:
            timestep embeddings of shape (N, hidden_size)
        """
        # Create sinusoidal embeddings
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        
        # Pass through MLP
        t_emb = self.linear1(t_freq)
        t_emb = nnx.silu(t_emb)
        t_emb = self.linear2(t_emb)
        
        return t_emb
    
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    Generate 2D sin-cos positional embeddings.
    
    Args:
        embed_dim: embedding dimension
        grid_size: int of the grid height and width
        cls_token: if True, add cls token
        extra_tokens: number of extra tokens to prepend
    
    Returns:
        pos_embed: [grid_size*grid_size, embed_dim] or 
                   [extra_tokens+grid_size*grid_size, embed_dim] (with cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    Generate 2D sin-cos embeddings from grid.
    
    Args:
        embed_dim: embedding dimension
        grid: grid of shape [2, 1, H, W]
    
    Returns:
        embeddings of shape [H*W, embed_dim]
    """
    assert embed_dim % 2 == 0
    
    # Use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Generate 1D sin-cos embeddings from positions.
    
    Args:
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
    
    Returns:
        embeddings of shape (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)
    
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
    
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class SwiGLUFFN(nnx.Module):
    """SwiGLU Feed-Forward Network."""
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop: float = 0.0,
        bias: bool = True,
        rngs: nnx.Rngs = None
    ) -> None:
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nnx.Linear(dim, 2 * hidden_dim, use_bias=bias, rngs=rngs)
        self.w3 = nnx.Linear(hidden_dim, dim, use_bias=bias, rngs=rngs)
        self.drop_rate = drop
        self.rngs = rngs
    
    def __call__(self, x, train: bool = False):
        # Project to 2 * hidden_dim
        x12 = self.w12(x)
        
        # Split into two halves
        x1, x2 = jnp.split(x12, 2, axis=-1)
        
        # SwiGLU activation: silu(x1) * x2
        hidden = nnx.silu(x1) * x2
        
        # Apply dropout
        if train and self.drop_rate > 0:
            keep_prob = 1.0 - self.drop_rate
            mask = jax.random.bernoulli(self.rngs(), keep_prob, hidden.shape)
            hidden = jnp.where(mask, hidden / keep_prob, 0.0)
        
        # Project back to dim
        return self.w3(hidden)
    

def modulate(x, shift, scale):
    """Apply adaptive modulation to normalized features."""
    return x * (1 + jnp.expand_dims(scale, -2)) + jnp.expand_dims(shift, -2)

class JiTBlock(nnx.Module):
    """JiT Transformer Block with Adaptive Layer Normalization."""
    
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, 
                 attn_drop=0.0, proj_drop=0.0, rngs=None):
        """
        Args:
            hidden_size: dimension of hidden states
            num_heads: number of attention heads
            mlp_ratio: ratio of MLP hidden dim to embedding dim
            attn_drop: attention dropout rate
            proj_drop: projection dropout rate
            rngs: random number generator
        """
        # Layer norms
        self.norm1 = RMSNorm(hidden_size, eps=1e-6, rngs=rngs)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6, rngs=rngs)
        
        # Attention
        self.attn = Attention(
            hidden_size, 
            num_heads=num_heads, 
            qkv_bias=True, 
            qk_norm=True,
            attn_drop=attn_drop, 
            proj_drop=proj_drop, 
            rngs=rngs
        )
        
        # MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop, rngs=rngs)
        
        # Adaptive Layer Norm modulation
        self.ada_linear = nnx.Linear(hidden_size, 6 * hidden_size, rngs=rngs)
    
    def __call__(self, x, c, feat_rope=None, train=False):
        """
        Args:
            x: input features of shape (N, hidden_size) - no batch dimension
            c: conditioning signal of shape (hidden_size,)
            feat_rope: RoPE embedding function
            train: whether in training mode
        
        Returns:
            output features of shape (N, hidden_size)
        """
        # Compute adaptive modulation parameters
        ada = self.ada_linear(nnx.silu(c))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(ada, 6, axis=-1)
        
        # Attention block with adaptive modulation
        x = x + jnp.expand_dims(gate_msa, -2) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), 
            rope=feat_rope, 
            train=train
        )
        
        # MLP block with adaptive modulation
        x = x + jnp.expand_dims(gate_mlp, -2) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp), 
            train=train
        )
        
        return x
    
class FinalLayer(nnx.Module):
    """The final layer of JiT."""
    
    def __init__(self, hidden_size, patch_size, out_channels, rngs=None):
        """
        Args:
            hidden_size: dimension of hidden states
            patch_size: size of image patches
            out_channels: number of output channels
            rngs: random number generator
        """
        self.norm_final = RMSNorm(hidden_size, rngs=rngs)
        self.linear = nnx.Linear(
            hidden_size, 
            patch_size * patch_size * out_channels, 
            use_bias=True, 
            rngs=rngs
        )
        # AdaLN modulation (2 parameters: shift and scale)
        self.ada_linear = nnx.Linear(hidden_size, 2 * hidden_size, rngs=rngs)
    
    def __call__(self, x, c):
        """
        Args:
            x: input features of shape (N, hidden_size) - no batch dimension
            c: conditioning signal of shape (hidden_size,)
        
        Returns:
            output of shape (N, patch_size * patch_size * out_channels)
        """
        # Compute adaptive modulation parameters
        ada = self.ada_linear(nnx.silu(c))
        shift, scale = jnp.split(ada, 2, axis=-1)
        
        # Apply adaptive normalization and modulation
        x = modulate(self.norm_final(x), shift, scale)
        
        # Project to output space
        x = self.linear(x)
        
        return x
    
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
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes
        
        # Time and class embedders
        self.t_embedder = TimestepEmbedder(hidden_size, rngs=rngs)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, rngs=rngs)
        
        # Patch embedding
        self.x_embedder = BottleneckPatchEmbed(
            input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True, rngs=rngs
        )
        
        # Fixed sin-cos positional embeddings
        num_patches = self.x_embedder.num_patches
        pos_embed = get_2d_sincos_pos_embed(
            hidden_size, 
            int(num_patches ** 0.5),
            cls_token=False,
            extra_tokens=0
        )
        self.pos_embed = nnx.Param(jnp.array(pos_embed, dtype=jnp.float32))
        
        # In-context positional embeddings
        if self.in_context_len > 0:
            self.in_context_posemb = nnx.Param(
                jax.random.normal(rngs(), (self.in_context_len, hidden_size)) * 0.02
            )
        
        # RoPE
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        
        # RoPE without in-context tokens
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0,  # No extra tokens
            rngs=rngs
        )
        
        # RoPE WITH in-context tokens
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=in_context_len,  # Add in-context tokens!
            rngs=rngs
        )
        
        # Transformer blocks - USE nnx.List instead of regular list
        self.blocks = nnx.List([
            JiTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                rngs=rngs
            )
            for i in range(depth)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, rngs=rngs)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights following the PyTorch implementation."""
        # Zero-out adaLN modulation layers in blocks
        for block in self.blocks:
            block.ada_linear.kernel.value = jnp.zeros_like(block.ada_linear.kernel.value)
            block.ada_linear.bias.value = jnp.zeros_like(block.ada_linear.bias.value)
        
        # Zero-out final layer adaLN and linear
        self.final_layer.ada_linear.kernel.value = jnp.zeros_like(self.final_layer.ada_linear.kernel.value)
        self.final_layer.ada_linear.bias.value = jnp.zeros_like(self.final_layer.ada_linear.bias.value)
        self.final_layer.linear.kernel.value = jnp.zeros_like(self.final_layer.linear.kernel.value)
        self.final_layer.linear.bias.value = jnp.zeros_like(self.final_layer.linear.bias.value)
    
    def unpatchify(self, x, p):
        """
        Convert patches back to image.
        x: (N, patch_size**2 * C)
        returns: (H, W, C) in JAX format
        """
        c = self.out_channels
        N = x.shape[0]
        h = w = int(N ** 0.5)
        assert h * w == N
        
        x = x.reshape(h, w, p, p, c)
        x = jnp.einsum('hwpqc->hpwqc', x)
        imgs = x.reshape(h * p, w * p, c)
        return imgs
    
    def __call__(self, x, t, y, train=False):
        """
        Forward pass without batch dimension (use vmap for batching).
        
        Args:
            x: input image (H, W, C) in JAX format
            t: timestep scalar or (1,) array
            y: class label scalar or (1,) array
            train: training mode
        
        Returns:
            output image (H, W, C)
        """
        # Ensure t and y are arrays
        if jnp.ndim(t) == 0:
            t = jnp.array([t])
        if jnp.ndim(y) == 0:
            y = jnp.array([y])
        
        # Get embeddings and combine
        t_emb = self.t_embedder(t)[0]  # (hidden_size,)
        y_emb = self.y_embedder(y)[0]  # (hidden_size,)
        c = t_emb + y_emb
        
        # Patch embedding
        x = self.x_embedder(x)  # (num_patches, hidden_size)
        x = x + self.pos_embed.value
        
        # Forward through blocks
        for i, block in enumerate(self.blocks):
            # Add in-context tokens
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = jnp.tile(
                    jnp.expand_dims(y_emb, 0), 
                    (self.in_context_len, 1)
                )
                in_context_tokens = in_context_tokens + self.in_context_posemb.value
                x = jnp.concatenate([in_context_tokens, x], axis=0)
            
            # Select appropriate RoPE
            rope = self.feat_rope if i < self.in_context_start else self.feat_rope_incontext
            x = block(x, c, feat_rope=rope, train=train)
        
        # Remove in-context tokens
        x = x[self.in_context_len:]
        
        # Final layer
        x = self.final_layer(x, c)
        
        # Unpatchify to image
        output = self.unpatchify(x, self.patch_size)
        
        return output

# Model variants
def JiT_B_16(rngs, **kwargs):
    return JiT(
        depth=12, hidden_size=768, num_heads=12,
        bottleneck_dim=128, in_context_len=32, in_context_start=4,
        patch_size=16, rngs=rngs, **kwargs
    )
