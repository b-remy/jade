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

    def step(self, xt, cosmot, t, t_next):
        raise NotImplementedError()

    def __call__(self, x0, cosmo0):
        """
        x0: initial state
        """

        timesteps = jnp.linspace(0.0, 1.0, self.num_steps + 1)
        
        def scan_body(carry, timestep_pair):
            z_x, z_cosmo = carry
            t, t_next = timestep_pair
            z_x_new, z_cosmo_new = self.step(z_x, z_cosmo, t, t_next)
            return (z_x_new, z_cosmo_new), None

        # Initial state
        init_carry = (x0, cosmo0)

        # Create pairs of (t, t_next)
        timestep_pairs = (timesteps[:-1], timesteps[1:])

        # Run the scan
        final_carry, _ = jax.lax.scan(scan_body, init_carry, timestep_pairs)

        z_x, z_cosmo = final_carry

        return z_x, z_cosmo

class EulerSampler(Sampler):

    def step(self, xt, cosmot, t, t_next):
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

    def step(self, xt, cosmot, t, t_next):
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
