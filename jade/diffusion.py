import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial


class VESDE:
    """Variance Exploding SDE.
    
    The forward SDE: dx = sigma(t) * dw
    where sigma(t) = sigma_min * (sigma_max / sigma_min)^t
    """
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def sigma(self, t):
        """Compute noise schedule at time t.
        
        Args:
            t: Time value(s) in [0, 1]
            
        Returns:
            Noise level sigma(t)
        """
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

class Denoiser(nnx.Module):
    def __init__(self, model: nnx.Module, cfg: dict):
        """Initialize the Denoiser with a model and config.
        
        Args:
            model: An NNX module that takes (x, cosmo, t, train) as inputs
            cfg: Configuration dictionary with diffusion parameters
        """
        self.model = model
        self.cfg = cfg
        
        # Sampling parameters
        self.steps = cfg.get('sampling', {}).get('steps', 50)
        self.method = cfg.get('sampling', {}).get('method', 'euler')  # 'euler', 'heun', or 'ddpm'
        self.t_eps = cfg.get('sampling', {}).get('t_eps', 0.05)
        self.noise_scale = cfg.get('sampling', {}).get('noise_scale', 1.0)
        
        # Initialize VESDE for variance exploding methods
        sigma_min = cfg.get('diffusion', {}).get('sigma_min', 0.01)
        sigma_max = cfg.get('diffusion', {}).get('sigma_max', 50.0)
        self.sde = VESDE(sigma_min=sigma_min, sigma_max=sigma_max)
    
    def forward_process(self, x, cosmo, key):
        """Forward corruption process using flow matching.
        
        Args:
            x: Clean field data [batch, H, W, channels]
            cosmo: Clean cosmological parameters [batch, n_params]
            key: Random key
            
        Returns:
            tuple: (xt, cosmot, t) - corrupted data and time
        """
        keys = jax.random.split(key, 3)
        
        # Sample time
        if self.cfg['diffusion']['time_distribution'] == "logit":
            mu = self.cfg['diffusion']['mu']
            sigma = self.cfg['diffusion']['sigma']
            s = (jax.random.normal(keys[0], shape=x.shape[:1]) + mu) * sigma
            t = jax.nn.sigmoid(s)
        elif self.cfg['diffusion']['time_distribution'] == "beta":
            t = jax.random.beta(
                keys[0], 
                a=self.cfg['diffusion']['beta_a'], 
                b=self.cfg['diffusion']['beta_b'], 
                shape=x.shape[:1]
            )
        else:  # uniform
            t = jax.random.uniform(keys[0], shape=x.shape[:1])
        
        # Flow matching interpolation: x_t = t*x + (1-t)*noise
        xt = (t[..., None, None, None] * x + 
              (1 - t[..., None, None, None]) * jax.random.normal(keys[1], shape=x.shape))
        
        cosmot = (t[..., None] * cosmo + 
                  (1 - t[..., None]) * jax.random.normal(keys[2], shape=cosmo.shape))
        
        return xt, cosmot, t
    
    def __call__(self, x, cosmo, sigma_t, train=True):
        """Forward pass through the denoiser model.
        
        Args:
            x: Input field data (possibly corrupted)
            cosmo: Input cosmological parameters (possibly corrupted)
            t: Time values
            train: Whether in training mode
            
        Returns:
            tuple: (x_pred, cosmo_pred) - predicted clean data
        """

        # Apply EDM preconditioning (sigma_data = 1.0 for standardized data)
        c_skip = jnp.atleast_1d(1.0 / (sigma_t**2 + 1.0))
        c_out = sigma_t / jnp.sqrt(sigma_t**2 + 1.0)
        c_in = 1.0 / jnp.sqrt(sigma_t**2 + 1.0)
        c_noise = 0.25 * jnp.log(sigma_t)
        
        # Forward with preconditioning
        x_net, cosmo_net = self.model(c_in * x, c_in * cosmo, c_noise, train)
        
        # Apply output scaling and skip connection
        x_pred = c_skip * x + c_out * x_net
        cosmo_pred = c_skip * cosmo + c_out * cosmo_net
        
        return x_pred, cosmo_pred
    
    def loss_fn(self, x, cosmo, key, train=True, return_components=False):
        """Compute denoising loss.
        
        Args:
            x: Clean field data [batch, H, W, channels]
            cosmo: Clean cosmological parameters [batch, n_params]
            key: Random key
            train: Whether in training mode
            return_components: If True, return (total_loss, (loss_x, loss_cosmo))
            
        Returns:
            Total loss, or (total_loss, (loss_x, loss_cosmo)) if return_components=True
        """
        keys = jax.random.split(key, 3)

        # Sample time
        if self.cfg['diffusion']['time_distribution'] == "logit":
            mu = self.cfg['diffusion']['mu']
            sigma = self.cfg['diffusion']['sigma']
            s = (jax.random.normal(keys[0], shape=x.shape[:1]) + mu) * sigma
            t = jax.nn.sigmoid(s)
        elif self.cfg['diffusion']['time_distribution'] == "beta":
            t = jax.random.beta(
                keys[0], 
                a=self.cfg['diffusion']['beta_a'], 
                b=self.cfg['diffusion']['beta_b'], 
                shape=x.shape[:1]
            )
        else:  # uniform
            t = jax.random.uniform(keys[0], shape=x.shape[:1])
        
        # Forward diffusion
        xt = (t[..., None, None, None] * x + 
              (1 - t[..., None, None, None]) * jax.random.normal(keys[1], shape=x.shape))
        cosmot = (t[..., None] * cosmo + 
                  (1 - t[..., None]) * jax.random.normal(keys[2], shape=cosmo.shape))

        # Predict clean data
        model_vmap = jax.vmap(self.model, in_axes=(0, 0, 0, None))
        x_pred, cosmo_pred = model_vmap(xt, cosmot, t, train)

        # Compute losses
        loss_x = jnp.mean((x - x_pred)**2, axis=(-1, -2, -3))
        loss_cosmo = jnp.mean((cosmo - cosmo_pred)**2, axis=-1)
        
        # Apply loss weighting
        if self.cfg["loss"]["type"] == "x-loss":
            total_loss = jnp.mean(
                loss_x + self.cfg['loss']['lambda_cosmo'] * loss_cosmo
            )
        elif self.cfg["loss"]["type"] == "v-loss":
            vx = (x - xt) / jnp.clip((1 - t[..., None, None, None]), a_min=self.t_eps)
            vx_pred = (x_pred - xt) / jnp.clip((1 - t[..., None, None, None]), a_min=self.t_eps)
            
            vcosmo = (cosmo - cosmot) / jnp.clip((1 - t[..., None]), a_min=self.t_eps)
            vcosmo_pred = (cosmo_pred - cosmot) / jnp.clip((1 - t[..., None]), a_min=self.t_eps)
            
            total_loss = (jnp.mean((vx - vx_pred)**2, (-1, -2, -3)) + 
                         self.cfg['loss']['lambda_cosmo'] * jnp.mean((vcosmo - vcosmo_pred)**2, (-1,)))
            total_loss = total_loss.mean()
        
        if return_components:
            return total_loss, (jnp.mean(loss_x), jnp.mean(loss_cosmo))
        else:
            return total_loss
    
    # ==================== Flow Matching Sampling Methods ====================
    
    def _forward_sample(self, z_x, z_cosmo, t):
        """Compute velocity predictions for both x and cosmo.
        
        Args:
            z_x: Noisy field data [batch, H, W, channels]
            z_cosmo: Noisy cosmological parameters [batch, n_params]
            t: Time values [batch,]
            
        Returns:
            tuple: (v_x, v_cosmo) - velocity predictions
        """
        # Predict clean data
        model_vmap = jax.vmap(self.model, in_axes=(0, 0, 0, None))
        x_pred, cosmo_pred = model_vmap(z_x, z_cosmo, t, False)
        
        # Compute velocity: v = (x - z) / (1 - t)
        t_broadcast_x = t[:, None, None, None]
        t_broadcast_cosmo = t[:, None]
        
        v_x = (x_pred - z_x) / jnp.clip(1.0 - t_broadcast_x, a_min=self.t_eps)
        v_cosmo = (cosmo_pred - z_cosmo) / jnp.clip(1.0 - t_broadcast_cosmo, a_min=self.t_eps)
        
        return v_x, v_cosmo
    
    def _euler_step(self, z_x, z_cosmo, t, t_next):
        """Single Euler step.
        
        Args:
            z_x: Current field state [batch, H, W, channels]
            z_cosmo: Current cosmological parameters state [batch, n_params]
            t: Current time [batch,]
            t_next: Next time [batch,]
            
        Returns:
            tuple: (z_x_next, z_cosmo_next) - next state
        """
        v_x, v_cosmo = self._forward_sample(z_x, z_cosmo, t)
        
        dt_x = (t_next - t)[:, None, None, None]
        dt_cosmo = (t_next - t)[:, None]
        
        z_x_next = z_x + dt_x * v_x
        z_cosmo_next = z_cosmo + dt_cosmo * v_cosmo
        
        return z_x_next, z_cosmo_next
    
    def _heun_step(self, z_x, z_cosmo, t, t_next):
        """Single Heun step (2nd order Runge-Kutta).
        
        Args:
            z_x: Current field state [batch, H, W, channels]
            z_cosmo: Current cosmological parameters state [batch, n_params]
            t: Current time [batch,]
            t_next: Next time [batch,]
            
        Returns:
            tuple: (z_x_next, z_cosmo_next) - next state
        """
        # First prediction at t
        v_x_t, v_cosmo_t = self._forward_sample(z_x, z_cosmo, t)
        
        dt_x = (t_next - t)[:, None, None, None]
        dt_cosmo = (t_next - t)[:, None]
        
        # Euler step to get tentative next state
        z_x_euler = z_x + dt_x * v_x_t
        z_cosmo_euler = z_cosmo + dt_cosmo * v_cosmo_t
        
        # Second prediction at t_next
        v_x_t_next, v_cosmo_t_next = self._forward_sample(z_x_euler, z_cosmo_euler, t_next)
        
        # Average the two predictions
        v_x = 0.5 * (v_x_t + v_x_t_next)
        v_cosmo = 0.5 * (v_cosmo_t + v_cosmo_t_next)
        
        z_x_next = z_x + dt_x * v_x
        z_cosmo_next = z_cosmo + dt_cosmo * v_cosmo
        
        return z_x_next, z_cosmo_next
    
    # ==================== Variance Exploding Sampling Methods ====================
    
    @nnx.jit
    def _ve_denoise_fn(self, xt, cosmot, sigma_t):
        """Model wrapper for VE sampling that takes sigma instead of t.
        
        Args:
            xt: Noisy field data [batch, H, W, channels]
            cosmot: Noisy cosmological parameters [batch, n_params]
            sigma_t: Noise level (can be scalar or array)
            
        Returns:
            tuple: (x_pred, cosmo_pred) - predicted clean data
        """
        # Convert sigma to time t if needed
        # For VE, we can pass sigma directly or convert to t space
        # Assuming model expects t in [0, 1], we can use sigma as a proxy
        # or define a mapping. Here we'll assume the model can handle it.
        
        # If sigma_t is scalar, broadcast to batch
        #if jnp.ndim(sigma_t) == 0:
        #    sigma_t = jnp.full((xt.shape[0],), sigma_t)
        
        model_vmap = jax.vmap(self.model, in_axes=(0, 0, 0, None))
        x_pred, cosmo_pred = model_vmap(xt, cosmot, sigma_t, False)
        
        return x_pred, cosmo_pred
    
    def _ddpm_step(self, xt, cosmot, t, s, key):
        """Single DDPM step for variance exploding SDE.
        
        Uses the reverse SDE:
        x_s = x_t - tau * (x_t - f(x_t)) + sigma_s * sqrt(tau) * epsilon
        where tau = 1 - (sigma_s / sigma_t)^2
        
        Args:
            xt: Current field state [batch, H, W, channels]
            cosmot: Current cosmological parameters state [batch, n_params]
            t: Current time [batch,] or scalar
            s: Next time [batch,] or scalar
            key: Random key for noise
            
        Returns:
            tuple: (x_next, cosmo_next) - next state
        """
        keys = jax.random.split(key, 2)
        
        # Get noise levels
        sigma_s = self.sde.sigma(s)
        sigma_t = self.sde.sigma(t)

        # Compute tau
        tau = 1 - (sigma_s / sigma_t) ** 2
        
        # Get denoised predictions
        x_pred, cosmo_pred = self._ve_denoise_fn(xt, cosmot, sigma_t)
        
        # Generate noise
        eps_x = jax.random.normal(keys[0], xt.shape)
        eps_cosmo = jax.random.normal(keys[1], cosmot.shape)
        
        # Broadcast tau if scalar
        if jnp.ndim(tau) == 0:
            tau_x = tau
            tau_cosmo = tau
            sigma_s_x = sigma_s
            sigma_s_cosmo = sigma_s
        else:
            tau_x = tau[:, None, None, None]
            tau_cosmo = tau[:, None]
            sigma_s_x = sigma_s[:, None, None, None]
            sigma_s_cosmo = sigma_s[:, None]
        
        # DDPM update
        x_next = xt - tau_x * (xt - x_pred) + sigma_s_x * jnp.sqrt(tau_x) * eps_x
        cosmo_next = cosmot - tau_cosmo * (cosmot - cosmo_pred) + sigma_s_cosmo * jnp.sqrt(tau_cosmo) * eps_cosmo
        
        return x_next, cosmo_next
    
    def generate_ve(self, key, batch_size=1, t_start=1.0):
        """Generate samples using variance exploding DDPM sampling.
        
        Args:
            key: Random key
            batch_size: Number of samples to generate
            t_start: Starting time (default 1.0, maximum noise)
            
        Returns:
            tuple: (x_generated, cosmo_generated) - generated samples
        """
        # Determine shapes
        x_shape = (
            self.cfg.get('data', {}).get('image_size', 128),
            self.cfg.get('data', {}).get('image_size', 128),
            self.cfg.get('data', {}).get('n_channels', 5)
        )
        cosmo_dim = 6
        
        # Initialize from noise scaled by sigma(t_start)
        keys = jax.random.split(key, 2 + self.steps)
        sigma_start = self.sde.sigma(t_start)
        
        xt = sigma_start * jax.random.normal(keys[0], shape=(batch_size, 128, 128, 5))
        cosmot = sigma_start * jax.random.normal(keys[1], shape=(batch_size, cosmo_dim))
        
        # Create timesteps from t_start to dt (near 0)
        dt = t_start / self.steps
        timesteps = jnp.linspace(t_start, dt, self.steps)
        
        # Reverse process: iterate from high noise to low noise
        for i in range(self.steps - 1):
            t = jnp.full((batch_size,), timesteps[i])
            s = jnp.full((batch_size,), timesteps[i + 1])
            xt, cosmot = self._ddpm_step(xt, cosmot, t, s, keys[2 + i])
        
        # Final denoising step at t=dt
        x_final, cosmo_final = self._ve_denoise_fn(xt, cosmot, 0.*jnp.ones((batch_size,)))
        #return xt, cosmot
        return x_final, cosmo_final
    
    def generate(self, key, batch_size=1, x_shape=None, cosmo_shape=None, use_ve=False):
        """Generate joint samples of x and cosmo.
        
        Args:
            key: Random key for initialization
            batch_size: Number of samples to generate
            x_shape: Shape of field data (H, W, channels). If None, inferred from config
            cosmo_shape: Shape of cosmo parameters. If None, uses default (6,)
            use_ve: If True, use variance exploding DDPM sampler. Otherwise use flow matching.
            
        Returns:
            tuple: (x_generated, cosmo_generated) - generated samples
        """
        if use_ve:
            return self.generate_ve(key, batch_size)
        
        # Original flow matching generation
        # Determine field shape
        if x_shape is None:
            x_shape = (
                self.cfg.get('data', {}).get('image_size', 128),
                self.cfg.get('data', {}).get('image_size', 128),
                self.cfg.get('data', {}).get('n_channels', 5)
            )
        
        # Get cosmological parameter dimension
        cosmo_dim = 6
        
        # Initialize from pure noise (t=0)
        keys = jax.random.split(key, 2)
        z_x = self.noise_scale * jax.random.normal(keys[0], shape=(batch_size, 128, 128, 5))
        z_cosmo = self.noise_scale * jax.random.normal(keys[1], shape=(batch_size, cosmo_dim))
        
        # Create timesteps from 0 to 1
        timesteps = jnp.linspace(0.0, 1.0, self.steps + 1)
        
        # Select stepper
        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError(f"Method {self.method} not implemented")
        
        # ODE integration from t=0 (noise) to t=1 (data)
        for i in range(self.steps - 1):
            t = jnp.full((batch_size,), timesteps[i])
            t_next = jnp.full((batch_size,), timesteps[i + 1])
            z_x, z_cosmo = stepper(z_x, z_cosmo, t, t_next)
        
        # Last step with Euler
        t = jnp.full((batch_size,), timesteps[-2])
        t_next = jnp.full((batch_size,), timesteps[-1])
        z_x, z_cosmo = self._euler_step(z_x, z_cosmo, t, t_next)
        
        return z_x, z_cosmo
