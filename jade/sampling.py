import jax
import jax.numpy as jnp

from flax import nnx

class Sampler(nnx.Module):

    def __init__(self, model, num_steps: int):
        """
        model: denoiser
        x0: initial state
        num_steps: number of integration steps
        """

        self.model = model
        self.num_steps = num_steps 

    def step(self, xt, cosmot, t, t_next, key=None):
        raise NotImplementedError()

    def __call__(self, x0, cosmo0, key=None):
        """
        x0: initial state
        """

        timesteps = jnp.linspace(0.0, 1.0, self.num_steps + 1)
        keys = jax.random.split(key, self.num_steps+1)

        def scan_body(carry, t_pair_key):
            z_x, z_cosmo = carry
            t, t_next, key = t_pair_key
            z_x_new, z_cosmo_new = self.step(z_x, z_cosmo, t, t_next, key=key)
            return (z_x_new, z_cosmo_new), None

        # Initial state
        init_carry = (x0, cosmo0)

        # Create pairs of (t, t_next)
        timestep_pairs_key = (timesteps[:-1], timesteps[1:], keys[:-1])

        # Run the ODE solver
        (z_x, z_cosmo), _ = jax.lax.scan(scan_body, init_carry, timestep_pairs_key)
        
        # Final Euler step
        t, t_next = timesteps[-2], timesteps[-1]

        v_x, v_cosmo = self.model.v_pred(z_x, z_cosmo, t, False)

        dt_x = (t_next - t)
        dt_cosmo = (t_next - t)
        
        z_x = z_x + dt_x * v_x
        z_cosmo = z_cosmo + dt_cosmo * v_cosmo
        
        return z_x, z_cosmo

class EulerSampler(Sampler):

    def step(self, xt, cosmot, t, t_next, **kwargs):
        """
        xt: current state
        cosmot: current cosmology
        t: current time
        t_next: next time
        """
        
        v_x, v_cosmo = self.model.v_pred(xt, cosmot, t, False)

        dt_x = (t_next - t)
        dt_cosmo = (t_next - t)
        
        x_next = xt + dt_x * v_x
        cosmo_next = cosmot + dt_cosmo * v_cosmo
        
        return x_next, cosmo_next
    
class HeunSampler(Sampler):

    def step(self, xt, cosmot, t, t_next, **kwargs):
        """
        xt: current state
        cosmot: current cosmology
        t: current time
        t_next: next time
        """

        # First prediction at t
        v_x_t, v_cosmo_t = self.model.v_pred(xt, cosmot, t, False)
        
        dt_x = (t_next - t)
        dt_cosmo = (t_next - t)
        
        # Euler step to get tentative next state
        x_euler = xt + dt_x * v_x_t
        cosmo_euler = cosmot + dt_cosmo * v_cosmo_t
        
        # Second prediction at t_next
        v_x_t_next, v_cosmo_t_next = self.model.v_pred(x_euler, cosmo_euler, t_next, False)
        
        # Average the two predictions
        v_x = 0.5 * (v_x_t + v_x_t_next)
        v_cosmo = 0.5 * (v_cosmo_t + v_cosmo_t_next)
        
        x_next = xt + dt_x * v_x
        cosmo_next = cosmot + dt_cosmo * v_cosmo

        return x_next, cosmo_next

class DDPM(Sampler):

    def step(self, xt, cosmot, t, t_next, key):
        """
        xt: current state
        cosmot: current cosmology
        t: current time
        t_next: next time
        """

        x_pred, cosmo_pred = self.model.x_pred(xt, cosmot, t, False)

        alpha_t = 1.0 - t
        sigma_t = jnp.clip(t, a_min=self.model.t_eps)

        alpha_t_next = jnp.clip(1.0 - t_next, a_min=self.model.t_eps)
        sigma_t_next = t_next

        # tau = 1 - (alpha_t / alpha_t_next * sigma_t_next / sigma_t)**2
        tau = 1 - (alpha_t_next / alpha_t * sigma_t / sigma_t_next)**2

        keys = jax.random.split(key, 2)
        eps_x = jax.random.normal(keys[0], shape=xt.shape)
        eps_cosmo = jax.random.normal(keys[1], shape=cosmot.shape)

        x_next = alpha_t_next * x_pred
        x_next = x_next + sigma_t_next * jnp.sqrt(1 - tau) / sigma_t * (xt - alpha_t * x_pred)
        x_next = x_next + sigma_t_next * jnp.sqrt(tau) * eps_x

        cosmo_next = alpha_t_next * cosmo_pred
        cosmo_next = cosmo_next + sigma_t_next * jnp.sqrt(1-tau) / sigma_t * (cosmot - alpha_t * cosmo_pred)
        cosmo_next = cosmo_next + sigma_t_next * jnp.sqrt(tau) * eps_cosmo

        return x_next, cosmo_next