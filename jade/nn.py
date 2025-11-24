import jax
import jax.numpy as jnp
from flax import nnx

from einops import repeat, rearrange

import math

class PatchEmbed(nnx.Module):
    """Image to Patch Embedding"""
    
    def __init__(
        self, 
        in_features: int = 1,
        pca_dim: int = 768, 
        embed_dim: int = 768, 
        patch_size: int = 16,
        rngs: nnx.Rngs = None
    ):
        self.patch_size = patch_size
        self.pca_dim = pca_dim
        self.embed_dim = embed_dim
        
        # Patchify layer
        self.layer1 = nnx.Conv(
            in_features=in_features,
            out_features=pca_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            use_bias=False,
            rngs=rngs
        )
        
        # Linear predictor (patch to token)
        self.layer2 = nnx.Conv(
            in_features=pca_dim,
            out_features=embed_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=False,
            rngs=rngs
        )
    
    def __call__(self, x):
        """
        Args:
            x: Input tensor of shape (H, W, C) in JAX/Flax convention
        
        Returns:
            Tensor of shape (num_patches, embed_dim)
        """
        # Apply convolutions
        x = self.layer1(x)
        x = self.layer2(x)
        x = rearrange(x, "H W C -> (H W) C")        
        
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


# def rotate_half(x):
#     """Rotate half the hidden dims."""
#     x = rearrange(x, '... (d r) -> ... d r', r=2)
#     x1, x2 = x[..., 0], x[..., 1]
#     x = jnp.stack([-x2, x1], axis=-1)
#     return rearrange(x, '... d r -> ... (d r)')


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
        rngs=None
    ):
        # Compute base frequencies
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
        
        # Create time steps
        ft_seq_len = ft_seq_len or pt_seq_len
        t = jnp.arange(ft_seq_len, dtype=jnp.float32) / ft_seq_len * pt_seq_len
        
        # Compute frequencies for each position
        freqs = jnp.einsum('t,f->tf', t, freqs)
        freqs = repeat(freqs, 't f -> t (f r)', r=2)
        
        # Broadcast to same shape, then concatenate
        freqs_h = freqs[:, None, :]  # (seq_len, 1, dim)
        freqs_w = freqs[None, :, :]  # (1, seq_len, dim)
        freqs_h = jnp.broadcast_to(freqs_h, (ft_seq_len, ft_seq_len, freqs.shape[-1]))
        freqs_w = jnp.broadcast_to(freqs_w, (ft_seq_len, ft_seq_len, freqs.shape[-1]))
        freqs = jnp.concatenate([freqs_h, freqs_w], axis=-1)

        # Flatten
        freqs_flat = freqs.reshape(-1, freqs.shape[-1])
        
        # Precompute cos and sin
        self.freqs_cos = jnp.cos(freqs_flat)
        self.freqs_sin = jnp.sin(freqs_flat)
    
    def __call__(self, t):
        """Apply rotary embeddings."""
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin