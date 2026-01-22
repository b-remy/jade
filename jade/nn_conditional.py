class JADE_conditional(nnx.Module):
    """
    Joint denoising transformer for dark matter fields and cosmology with optional image conditioning.
    JADE: Joint Analysis of Density and Expansion.
    """
    
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=1,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        bottleneck_dim=128,
        cosmo_dim=6,
        cond_channels=1,  # Conditioning image channels
        enable_cond_image=True,  # Whether to create conditioning components
        rngs=None
    ):
        """
        Args:
            input_size: size of input images (assumes square)
            patch_size: size of patches
            in_channels: number of input field channels
            hidden_size: transformer hidden dimension
            depth: number of transformer blocks
            num_heads: number of attention heads
            mlp_ratio: MLP hidden dim ratio
            attn_drop: attention dropout rate
            proj_drop: projection dropout rate
            bottleneck_dim: bottleneck dimension in patch embedding
            cosmo_dim: number of cosmological parameters to denoise
            cond_channels: number of conditioning image channels
            enable_cond_image: whether to create conditioning components (set False to disable entirely)
            rngs: random number generator (will be split for each component)
        """
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cosmo_dim = cosmo_dim
        self.cond_channels = cond_channels
        self.enable_cond_image = enable_cond_image
        
        # Calculate number of components for RNG splitting
        num_components = depth + 7
        if enable_cond_image:
            num_components += 2  # cond_embedder + cond_pos_embed
        
        rng_keys = jax.random.split(rngs(), num_components)
        
        idx = 0
        rngs_t_embedder = nnx.Rngs(rng_keys[idx]); idx += 1
        rngs_x_embedder = nnx.Rngs(rng_keys[idx]); idx += 1
        rngs_cosmo_embedder = nnx.Rngs(rng_keys[idx]); idx += 1
        
        # Conditionally allocate RNG for conditioning components
        if enable_cond_image:
            rngs_cond_embedder = nnx.Rngs(rng_keys[idx]); idx += 1
            rngs_cond_pos = nnx.Rngs(rng_keys[idx]); idx += 1
        
        rngs_rope = nnx.Rngs(rng_keys[idx]); idx += 1
        rngs_rope_nocond = nnx.Rngs(rng_keys[idx]); idx += 1  # Separate RoPE for no conditioning
        rngs_final = nnx.Rngs(rng_keys[idx]); idx += 1
        rngs_cosmo_head = nnx.Rngs(rng_keys[idx]); idx += 1
        rngs_blocks = [nnx.Rngs(rng_keys[idx + i]) for i in range(depth)]
        
        # Time embedder (diffusion timestep)
        self.t_embedder = TimestepEmbedder(hidden_size, rngs=rngs_t_embedder)
        
        # Cosmology embedder (cosmo_dim parameters → cosmo_dim tokens)
        self.cosmo_embedder = CosmologyEmbedder(
            cosmo_dim, hidden_size, rngs=rngs_cosmo_embedder
        )
        
        # Field patch embedding (for the noisy field to denoise)
        self.x_embedder = BottleneckPatchEmbed(
            input_size, patch_size, in_channels, bottleneck_dim, hidden_size,
            bias=True, rngs=rngs_x_embedder
        )
        
        # Conditioning image components (only created if enabled)
        if self.enable_cond_image:
            self.cond_embedder = BottleneckPatchEmbed(
                input_size, patch_size, cond_channels, bottleneck_dim, hidden_size,
                bias=True, rngs=rngs_cond_embedder
            )
            
            # Learnable positional embeddings for conditioning image
            num_patches = self.x_embedder.num_patches
            cond_pos_embed = jax.random.normal(
                rngs_cond_pos(), (num_patches, hidden_size)
            ) * 0.02
            self.cond_pos_embed = nnx.Param(cond_pos_embed)
        
        # Fixed sin-cos positional embeddings for field patches
        num_patches = self.x_embedder.num_patches
        pos_embed = get_2d_sincos_pos_embed(
            hidden_size, int(num_patches ** 0.5),
            cls_token=False, extra_tokens=0
        )
        self.pos_embed = nnx.Param(jnp.array(pos_embed, dtype=jnp.float32))
        
        # Create TWO RoPE instances: one with conditioning, one without
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        
        # RoPE WITHOUT conditioning (cosmo_dim tokens only as prefix)
        self.feat_rope_nocond = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=cosmo_dim,  # Only skip cosmo tokens
            rngs=rngs_rope_nocond
        )
        
        # RoPE WITH conditioning (cosmo_dim tokens as prefix, apply to both cond and field)
        # This applies spatial encoding to BOTH cond_patches and field_patches
        if self.enable_cond_image:
            self.feat_rope_cond = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
                num_cls_token=cosmo_dim,  # Only skip cosmo, apply RoPE to cond+field
                rngs=rngs_rope
            )
        
        # Transformer blocks
        self.blocks = nnx.List([
            JiTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                rngs=rngs_blocks[i]
            )
            for i in range(depth)
        ])
        
        # Output heads
        self.field_head = FinalLayer(hidden_size, patch_size, self.out_channels, rngs=rngs_final)
        self.cosmo_head = CosmologyPredictor(hidden_size, cosmo_dim, rngs=rngs_cosmo_head)
        
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize weights for JADE model following PyTorch JiT initialization pattern.
        """
        
        # ===================================================================
        # STEP 1: Basic initialization - Xavier uniform for ALL Linear layers
        # ===================================================================
        def init_linear_xavier(module):
            """Apply Xavier uniform initialization to a Linear module."""
            if isinstance(module, nnx.Linear):
                w = module.kernel.value
                w_flat = w.reshape(w.shape[0], -1)
                w_init = jax.nn.initializers.xavier_uniform()(
                    jax.random.PRNGKey(0), w_flat.shape
                )
                module.kernel.value = w_init.reshape(w.shape)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.value = jnp.zeros_like(module.bias.value)
        
        # Apply Xavier uniform to all Linear layers
        init_linear_xavier(self.t_embedder.linear1)
        init_linear_xavier(self.t_embedder.linear2)
        init_linear_xavier(self.cosmo_embedder.proj)
        init_linear_xavier(self.cosmo_head.proj)
        
        for block in self.blocks:
            init_linear_xavier(block.attn.qkv)
            init_linear_xavier(block.attn.proj)
            init_linear_xavier(block.mlp.w12)
            init_linear_xavier(block.mlp.w3)
            init_linear_xavier(block.ada_linear)
        
        init_linear_xavier(self.field_head.linear)
        init_linear_xavier(self.field_head.ada_linear)
        
        # ===================================================================
        # STEP 2: Initialize patch_embed like nn.Linear (Xavier uniform)
        # ===================================================================
        # Field embedder
        w1 = self.x_embedder.proj1.kernel.value
        w1_flat = w1.reshape(w1.shape[0], -1)
        w1_init = jax.nn.initializers.xavier_uniform()(
            jax.random.PRNGKey(1), w1_flat.shape
        )
        self.x_embedder.proj1.kernel.value = w1_init.reshape(w1.shape)
        
        w2 = self.x_embedder.proj2.kernel.value
        w2_flat = w2.reshape(w2.shape[0], -1)
        w2_init = jax.nn.initializers.xavier_uniform()(
            jax.random.PRNGKey(2), w2_flat.shape
        )
        self.x_embedder.proj2.kernel.value = w2_init.reshape(w2.shape)
        self.x_embedder.proj2.bias.value = jnp.zeros_like(self.x_embedder.proj2.bias.value)
        
        # Conditioning image embedder (only if enabled)
        if self.enable_cond_image:
            w1_cond = self.cond_embedder.proj1.kernel.value
            w1_cond_flat = w1_cond.reshape(w1_cond.shape[0], -1)
            w1_cond_init = jax.nn.initializers.xavier_uniform()(
                jax.random.PRNGKey(10), w1_cond_flat.shape
            )
            self.cond_embedder.proj1.kernel.value = w1_cond_init.reshape(w1_cond.shape)
            
            w2_cond = self.cond_embedder.proj2.kernel.value
            w2_cond_flat = w2_cond.reshape(w2_cond.shape[0], -1)
            w2_cond_init = jax.nn.initializers.xavier_uniform()(
                jax.random.PRNGKey(11), w2_cond_flat.shape
            )
            self.cond_embedder.proj2.kernel.value = w2_cond_init.reshape(w2_cond.shape)
            self.cond_embedder.proj2.bias.value = jnp.zeros_like(
                self.cond_embedder.proj2.bias.value
            )
        
        # ===================================================================
        # STEP 3: Initialize timestep embedder (normal, std=0.02)
        # ===================================================================
        key = jax.random.PRNGKey(3)
        self.t_embedder.linear1.kernel.value = jax.random.normal(
            key, self.t_embedder.linear1.kernel.value.shape
        ) * 0.02
        
        key = jax.random.PRNGKey(4)
        self.t_embedder.linear2.kernel.value = jax.random.normal(
            key, self.t_embedder.linear2.kernel.value.shape
        ) * 0.02
        
        # ===================================================================
        # STEP 4: Initialize cosmology embedder (normal, std=0.02)
        # ===================================================================
        key = jax.random.PRNGKey(5)
        self.cosmo_embedder.proj.kernel.value = jax.random.normal(
            key, self.cosmo_embedder.proj.kernel.value.shape
        ) * 0.02
        self.cosmo_embedder.proj.bias.value = jnp.zeros_like(
            self.cosmo_embedder.proj.bias.value
        )
        
        # ===================================================================
        # STEP 5: Zero-out adaLN modulation layers in transformer blocks
        # ===================================================================
        for block in self.blocks:
            block.ada_linear.kernel.value = jnp.zeros_like(block.ada_linear.kernel.value)
            block.ada_linear.bias.value = jnp.zeros_like(block.ada_linear.bias.value)
        
        # ===================================================================
        # STEP 6: Zero-out field output layers
        # ===================================================================
        self.field_head.ada_linear.kernel.value = jnp.zeros_like(
            self.field_head.ada_linear.kernel.value
        )
        self.field_head.ada_linear.bias.value = jnp.zeros_like(
            self.field_head.ada_linear.bias.value
        )
        self.field_head.linear.kernel.value = jnp.zeros_like(
            self.field_head.linear.kernel.value
        )
        self.field_head.linear.bias.value = jnp.zeros_like(
            self.field_head.linear.bias.value
        )
        
        # ===================================================================
        # STEP 7: Zero-out cosmology output head
        # ===================================================================
        self.cosmo_head.proj.kernel.value = jnp.zeros_like(
            self.cosmo_head.proj.kernel.value
        )
        self.cosmo_head.proj.bias.value = jnp.zeros_like(
            self.cosmo_head.proj.bias.value
        )

    def unpatchify(self, x, p):
        """
        Convert patches back to image.
        
        Args:
            x: (N, patch_size**2 * C)
            p: patch_size
        
        Returns:
            image (H, W, C) in JAX format
        """
        c = self.out_channels
        N = x.shape[0]
        h = w = int(N ** 0.5)
        assert h * w == N
        
        x = x.reshape(h, w, p, p, c)
        x = jnp.einsum('hwpqc->hpwqc', x)
        imgs = x.reshape(h * p, w * p, c)
        return imgs
    
    def __call__(self, field, cosmo, t, cond_image=None, train=False):
        """
        Joint forward pass: denoise both field and cosmology with optional image conditioning.
        
        Args:
            field: noisy dark matter field (H, W, C)
            cosmo: noisy cosmological parameters (cosmo_dim,)
            t: diffusion timestep (scalar or array)
            cond_image: conditioning image (H, W, C_cond), optional (can be None)
            train: training mode flag
        
        Returns:
            field_pred: denoised field prediction (H, W, C)
            cosmo_pred: denoised cosmology prediction (cosmo_dim,)
        """
        # Ensure t is array
        if jnp.ndim(t) == 0:
            t = jnp.array([t])
        
        # ===================================================================
        # STEP 1: Get time embedding for AdaLN conditioning
        # ===================================================================
        t_emb = self.t_embedder(t)[0]  # (hidden_size,)
        c = t_emb  # Just time conditioning
        
        # ===================================================================
        # STEP 2: Embed cosmology parameters as tokens
        # ===================================================================
        cosmo_tokens = self.cosmo_embedder(cosmo)  # (cosmo_dim, hidden_size)
        
        # ===================================================================
        # STEP 3: Build the token sequence (varies based on conditioning)
        # ===================================================================
        # Start with cosmology tokens
        token_list = [cosmo_tokens]
        
        # Track whether we're using conditioning for this call
        using_cond = self.enable_cond_image and cond_image is not None
        
        # Add conditioning image tokens if provided
        if using_cond:
            # Process conditioning image exactly like the input field
            cond_tokens = self.cond_embedder(cond_image)  # (num_patches, hidden_size)
            
            # Add learnable positional embeddings (different from field)
            cond_tokens = cond_tokens + self.cond_pos_embed.value
            
            token_list.append(cond_tokens)
            
            # Track number of conditioning patches for later extraction
            num_cond_patches = self.x_embedder.num_patches
        
        # Add noisy field tokens (to be denoised)
        field_tokens = self.x_embedder(field)  # (num_patches, hidden_size)
        field_tokens = field_tokens + self.pos_embed.value
        token_list.append(field_tokens)
        
        # Concatenate all tokens
        # Without cond: [cosmo_tokens, field_tokens]
        # With cond:    [cosmo_tokens, cond_tokens, field_tokens]
        x = jnp.concatenate(token_list, axis=0)
        
        # ===================================================================
        # STEP 4: Select appropriate RoPE based on whether conditioning is used
        # ===================================================================
        # Use feat_rope_cond if conditioning image is provided, else feat_rope_nocond
        feat_rope = self.feat_rope_cond if using_cond else self.feat_rope_nocond
        
        # ===================================================================
        # STEP 5: Forward through transformer blocks
        # ===================================================================
        for block in self.blocks:
            x = block(x, c, feat_rope=feat_rope, train=train)
        
        # ===================================================================
        # STEP 6: Split sequence back into components
        # ===================================================================
        idx = 0
        
        # Extract cosmology tokens
        cosmo_tokens_out = x[idx:idx + self.cosmo_dim]
        idx += self.cosmo_dim
        
        # Skip conditioning image tokens if present
        if using_cond:
            idx += num_cond_patches
        
        # Extract field tokens (rest of the sequence)
        field_tokens_out = x[idx:]
        
        # ===================================================================
        # STEP 7: Predict outputs
        # ===================================================================
        # Predict denoised cosmology
        cosmo_pred = self.cosmo_head(cosmo_tokens_out)
        
        # Predict denoised field
        field_tokens_pred = self.field_head(field_tokens_out, c)
        field_pred = self.unpatchify(field_tokens_pred, self.patch_size)
        
        return field_pred, cosmo_pred

def JADE_C_B_16(rngs, cosmo_dim=6, enable_cond_image=True, cond_channels=1, **kwargs):
    """Base model with 16x16 patches and optional image conditioning."""
    return JADE_conditional(
        depth=12, hidden_size=768, num_heads=12,
        bottleneck_dim=128,
        cosmo_dim=cosmo_dim,
        enable_cond_image=enable_cond_image,
        cond_channels=cond_channels,
        rngs=rngs, **kwargs
    )