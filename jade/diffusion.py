import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial


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
        self.method = cfg.get('sampling', {}).get('method', 'euler')  # 'euler' or 'heun'
        self.t_eps = cfg.get('sampling', {}).get('t_eps', 0.05)
        self.noise_scale = cfg.get('sampling', {}).get('noise_scale', 1.0)
    
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
    
    def __call__(self, x, cosmo, t, train=True):
        """Forward pass through the denoiser model.
        
        Args:
            x: Input field data (possibly corrupted)
            cosmo: Input cosmological parameters (possibly corrupted)
            t: Time values
            train: Whether in training mode
            
        Returns:
            tuple: (x_pred, cosmo_pred) - predicted clean data
        """
        return self.model(x, cosmo, t, train)
    
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
    
    # ==================== Sampling Methods ====================
    
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
    
    def generate(self, key, batch_size=1, x_shape=None, cosmo_shape=None):
        """Generate joint samples of x and cosmo using ODE sampling.
        
        Args:
            key: Random key for initialization
            batch_size: Number of samples to generate
            x_shape: Shape of field data (H, W, channels). If None, inferred from config
            
        Returns:
            tuple: (x_generated, cosmo_generated) - generated samples
        """
        # Determine field shape
        if x_shape is None:
            x_shape = (
                self.cfg.get('data', {}).get('image_size', 64),
                self.cfg.get('data', {}).get('image_size', 64),
                self.cfg.get('data', {}).get('n_channels', 3)
            )
        
        # Get cosmological parameter dimension
        cosmo_dim = 6
        
        # Initialize from pure noise (t=0)
        keys = jax.random.split(key, 2)
        z_x = self.noise_scale * jax.random.normal(keys[0], shape=(batch_size, 128,128,5))
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