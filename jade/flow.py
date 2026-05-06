import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial

from jade.lensing import Operator


class Denoiser(nnx.Module):
    def __init__(self, model: nnx.Module, cfg: dict):
        self.model = model
        self.cfg = cfg

        self.t_eps = cfg.get('sampling', {}).get('t_eps', 0.05)
        self.noise_scale = cfg.get('sampling', {}).get('noise_scale', 1.0)

    def __call__(self, x, cosmo, t, cond=None, train: bool = False, key=None):
        return self.model(x, cosmo, t, cond=cond, train=train, key=key)

    def x_pred(self, xt, cosmot, t, cond=None, train: bool = False, key=None):
        return self(xt, cosmot, t, cond=cond, train=train, key=key)

    def v_pred(self, xt, cosmot, t, cond=None, train: bool = False, key=None):
        x_pred, cosmo_pred = self.x_pred(xt, cosmot, t, cond=cond, train=train, key=key)

        # Compute velocity: v = (x - z) / (1 - t)
        v_x = (x_pred - xt) / jnp.clip(1.0 - t, a_min=self.t_eps)
        v_cosmo = (cosmo_pred - cosmot) / jnp.clip(1.0 - t, a_min=self.t_eps)

        return v_x, v_cosmo

    def dynamics_with_divergence(self, xt, cosmot, t, cond, eps_x, eps_cosmo):
        """Compute velocity and Hutchinson divergence estimate in one forward+vjp pass.
        
        Uses the Hutchinson trace estimator:
            Tr(J) ≈ εᵀ J ε
        where ε are random probe vectors (Rademacher or Gaussian).
        """
        def v_fn(xt_, cosmot_):
            return self.v_pred(xt_, cosmot_, t, cond=cond, train=False)

        (v_x, v_cosmo), vjp_fn = jax.vjp(v_fn, xt, cosmot)

        # Hutchinson: Tr(dv/d[x,cosmo]) ≈ [eps_x, eps_cosmo]^T @ J^T @ [eps_x, eps_cosmo]
        # vjp computes J^T @ eps, so eps^T @ (J^T @ eps) = eps^T @ J^T @ eps
        # For symmetric J this equals eps^T J eps; in general this is still
        # an unbiased estimator of Tr(J) since E[eps^T J^T eps] = Tr(J^T) = Tr(J).
        eps_jac_x, eps_jac_cosmo = vjp_fn((eps_x, eps_cosmo))

        div_x = jnp.sum(eps_jac_x * eps_x)
        div_cosmo = jnp.sum(eps_jac_cosmo * eps_cosmo)
        divergence = div_x + div_cosmo

        return v_x, v_cosmo, divergence

    def log_likelihood(self, x, cosmo, cond=None, key=None,
                       num_steps=100, num_hutchinson=1):
        """
        Compute log p(x, cosmo) via the instantaneous change-of-variables formula.

        For a flow ODE dx/dt = v(x, t), integrating from t=1 (data) to t=0 (prior):
            log p₁(x) = log p₀(z) + ∫₁→₀ Tr(∂v/∂x) dt

        The divergence Tr(∂v/∂x) is estimated using Hutchinson's trace estimator
        with Rademacher random vectors.

        Args:
            x: data sample, shape [H, W, C] (single sample, no batch dim)
            cosmo: cosmological parameters, shape [D]
            cond: optional conditioning
            key: JAX PRNG key
            num_steps: number of Euler integration steps
            num_hutchinson: number of Hutchinson probe vectors to average per step
        Returns:
            log_prob: scalar log-likelihood estimate
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        ts = jnp.linspace(1.0 - self.t_eps, self.t_eps, num_steps)
        dt = ts[1] - ts[0]  # negative, since going from ~1 to ~0

        # Pre-generate all Hutchinson probe vectors for all steps
        key, key_eps = jax.random.split(key)
        # Rademacher random vectors (±1) have lower variance than Gaussian
        eps_x_all = jax.random.rademacher(
            key_eps, shape=(num_steps, num_hutchinson, *x.shape)
        ).astype(x.dtype)
        eps_cosmo_all = jax.random.rademacher(
            jax.random.fold_in(key_eps, 1),
            shape=(num_steps, num_hutchinson, *cosmo.shape)
        ).astype(cosmo.dtype)

        def euler_step(carry, inputs):
            xt, cosmot, log_prob = carry
            t, eps_x_probes, eps_cosmo_probes = inputs

            # Average Hutchinson estimate over multiple probes
            def single_probe(eps_x, eps_cosmo):
                return self.dynamics_with_divergence(
                    xt, cosmot, t, cond, eps_x, eps_cosmo
                )

            v_xs, v_cosmos, divs = jax.vmap(single_probe)(eps_x_probes, eps_cosmo_probes)
            
            # Velocity is independent of probe vector — take the first
            v_x = v_xs[0]
            v_cosmo = v_cosmos[0]
            div_mean = jnp.mean(divs)

            # Euler update
            xt = xt + v_x * dt
            cosmot = cosmot + v_cosmo * dt
            log_prob = log_prob + div_mean * dt

            return (xt, cosmot, log_prob), None

        init_carry = (x, cosmo, 0.0)
        scan_inputs = (ts, eps_x_all, eps_cosmo_all)
        (z_x, z_cosmo, delta_logp, ), _ = jax.lax.scan(euler_step, init_carry, scan_inputs)

        # Standard normal prior log probability
        log_prior_x = (
            -0.5 * jnp.sum(z_x ** 2) 
            - 0.5 * z_x.size * jnp.log(2.0 * jnp.pi)
        )
        log_prior_cosmo = (
            -0.5 * jnp.sum(z_cosmo ** 2) 
            - 0.5 * z_cosmo.size * jnp.log(2.0 * jnp.pi)
        )
        log_prior = log_prior_x + log_prior_cosmo

        return log_prior + delta_logp

    def sample_with_log_prob(self, x_shape, cosmo_shape, cond=None, key=None,
                         num_steps=100, num_hutchinson=1):
        if key is None:
            key = jax.random.PRNGKey(0)

        key, key_z, key_eps = jax.random.split(key, 3)

        z_x = jax.random.normal(key_z, shape=x_shape)
        z_cosmo = jax.random.normal(jax.random.fold_in(key_z, 1), shape=cosmo_shape)

        log_prior_x = -0.5 * jnp.sum(z_x ** 2) - 0.5 * z_x.size * jnp.log(2.0 * jnp.pi)
        log_prior_cosmo = -0.5 * jnp.sum(z_cosmo ** 2) - 0.5 * z_cosmo.size * jnp.log(2.0 * jnp.pi)
        log_prior = log_prior_x + log_prior_cosmo

        timesteps = jnp.linspace(0.0, 1.0, num_steps + 1)

        # num_steps + 1 probe sets: num_steps for scan + 1 for final step
        eps_x_all = jax.random.rademacher(
            key_eps, shape=(num_steps + 1, num_hutchinson, *x_shape)
        ).astype(z_x.dtype)
        eps_cosmo_all = jax.random.rademacher(
            jax.random.fold_in(key_eps, 1),
            shape=(num_steps + 1, num_hutchinson, *cosmo_shape)
        ).astype(z_cosmo.dtype)

        def euler_step(carry, inputs):
            xt, cosmot, delta_logp = carry
            t, t_next, eps_x_probes, eps_cosmo_probes = inputs

            def single_probe(eps_x, eps_cosmo):
                return self.dynamics_with_divergence(
                    xt, cosmot, t, cond, eps_x, eps_cosmo
                )

            v_xs, v_cosmos, divs = jax.vmap(single_probe)(eps_x_probes, eps_cosmo_probes)
            v_x = v_xs[0]
            v_cosmo = v_cosmos[0]
            div_mean = jnp.mean(divs)

            dt = t_next - t
            xt = xt + v_x * dt
            cosmot = cosmot + v_cosmo * dt
            delta_logp = delta_logp - div_mean * dt

            return (xt, cosmot, delta_logp), None

        # Scan over ALL num_steps intervals (matching reference sampler scan)
        init_carry = (z_x, z_cosmo, 0.0)
        scan_inputs = (
            timesteps[:-1],          # t:      num_steps
            timesteps[1:],           # t_next: num_steps
            eps_x_all[:num_steps],   # probes: num_steps
            eps_cosmo_all[:num_steps],
        )
        (xt, cosmot, delta_logp), _ = jax.lax.scan(euler_step, init_carry, scan_inputs)

        # Final step: repeat last interval (matching reference sampler quirk)
        t, t_next = timesteps[-2], timesteps[-1]

        def single_probe_final(eps_x, eps_cosmo):
            return self.dynamics_with_divergence(
                xt, cosmot, t, cond, eps_x, eps_cosmo
            )

        v_xs, v_cosmos, divs = jax.vmap(single_probe_final)(
            eps_x_all[num_steps], eps_cosmo_all[num_steps]
        )
        v_x = v_xs[0]
        v_cosmo = v_cosmos[0]
        div_mean = jnp.mean(divs)

        dt = t_next - t
        x = xt + v_x * dt
        cosmo = cosmot + v_cosmo * dt
        delta_logp = delta_logp - div_mean * dt

        log_prob = log_prior + delta_logp

        # return x, cosmo, log_prob

        return x, cosmo, log_prior, delta_logp
        
    def importance_sampling(self, log_q_fn, x_shape, cosmo_shape, cond=None,
                            key=None, num_samples=100, num_steps=100,
                            num_hutchinson=1):
        """
        Self-normalized importance sampling using the flow as proposal
        and an exact likelihood q as target.

        Draws samples x_i ~ p(x, cosmo | cond) from the flow, then computes:
            w_i = q(x_i, cosmo_i | cond) / p(x_i, cosmo_i | cond)
            w_normalized_i = w_i / sum(w_j)

        Args:
            log_q_fn: callable (x, cosmo, cond) -> scalar log q(x, cosmo | cond)
                      The exact target log-density to importance-sample against.
            x_shape: shape of a single spatial field sample
            cosmo_shape: shape of a single cosmo vector
            cond: conditioning information
            key: JAX PRNG key
            num_samples: number of importance samples
            num_steps: Euler steps for ODE integration
            num_hutchinson: Hutchinson probes per step
        Returns:
            samples_x: array of shape (num_samples, *x_shape)
            samples_cosmo: array of shape (num_samples, *cosmo_shape)
            log_weights: unnormalized log importance weights (num_samples,)
            weights_normalized: self-normalized importance weights (num_samples,)
            ess: effective sample size
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        keys = jax.random.split(key, num_samples)

        # Sample and evaluate log p simultaneously — single ODE pass per sample
        def sample_one(k):
            return self.sample_with_log_prob(
                x_shape, cosmo_shape, cond=cond, key=k,
                num_steps=num_steps, num_hutchinson=num_hutchinson,
            )

        samples_x, samples_cosmo, log_p = jax.vmap(sample_one)(keys)

        # Evaluate exact target log-density
        log_q = jax.vmap(lambda x, c: log_q_fn(x, c, cond))(samples_x, samples_cosmo)

        # Log importance weights: log w = log q - log p
        log_weights = log_q - log_p

        # Self-normalized weights via log-sum-exp for numerical stability
        log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
        weights_normalized = jnp.exp(log_weights_normalized)

        # Effective sample size: ESS = 1 / sum(w_i^2)
        ess = 1.0 / jnp.sum(weights_normalized ** 2)

        return samples_x, samples_cosmo, log_weights, weights_normalized, ess

    def forward_coupling(self, x, cosmo, t, key):
        alpha_t = t
        sigma_t = 1.0 - t

        noise_x = jax.random.normal(key, shape=x.shape)
        noise_cosmo = jax.random.normal(key, shape=cosmo.shape)

        xt = alpha_t * x + sigma_t * noise_x
        cosmot = alpha_t * cosmo + sigma_t * noise_cosmo

        return xt, cosmot

class PosteriorDenoiser(Denoiser):
    def __init__(self, model: nnx.Module, cfg: dict, gamma, sigma_gamma=1.0):
        self.model = model
        self.cfg = cfg

        self.t_eps = cfg.get('sampling', {}).get('t_eps', 0.001)
        self.noise_scale = cfg.get('sampling', {}).get('noise_scale', 1.0)

        # linear solver
        self.solve = jax.scipy.sparse.linalg.cg
        self.tol = 1e-3
        self.maxiter = 10

        self.gamma = gamma
        self.cov_y = sigma_gamma ** 2

    def __call__(self, xt, cosmot, t, train: bool = False, *args, **kwargs):

        alpha_t = t
        sigma_t = 1.0 - t

        cov_t = sigma_t**2 / jnp.clip(alpha_t, a_min=self.t_eps)

        (x, cosmo), vjp = jax.vjp(lambda x, cosmo: self.model(x, cosmo, t, None, False), xt, cosmot)
   
        y, A = jax.linearize(Operator, x)
 
        At_ = jax.linear_transpose(Operator, x)
        At = lambda x: next(iter(At_(x)))

        # MMPS
        cov_y_xt = lambda v: self.cov_y*v + cov_t*A(vjp((At(v), jnp.zeros(6)))[0])

        b = self.gamma - y
        
        v, _ = self.solve(
            A=cov_y_xt,
            b=b,
            tol=self.tol,
            maxiter=self.maxiter,
        )

        (score, _) = vjp((At(v), jnp.zeros_like(cosmo)))
        
        return x + cov_t * score, cosmo
        
        
class FlowLoss(nnx.Module):
    def __init__(self, cfg: dict):
        self.t_eps = cfg["diffusion"]["t_eps"]
        self.mu = cfg['diffusion']['mu']
        self.sigma = cfg['diffusion']['sigma']
        self.mixture = cfg['loss'].get('mixture', False)

    def __call__(self, model, x, cosmo, key, lambda_cosmo, train: bool = False, cond=None):

        mu = self.mu
        sigma = self.sigma
        s = (jax.random.normal(key, shape=x.shape[:1]) + mu) * sigma
        t = jax.nn.sigmoid(s)

        alpha_t = t
        sigma_t = 1 - t

        key, key_noise_x, key_noise_c, key_dropout = jax.random.split(key, 4)
        noise_x = jax.random.normal(key_noise_x, shape=x.shape)
        noise_cosmo = jax.random.normal(key_noise_c, shape=cosmo.shape)

        xt = alpha_t[:,None,None,None] * x + sigma_t[:,None,None,None] * noise_x
        cosmot = alpha_t[:,None] * cosmo + sigma_t[:,None] * noise_cosmo

        # Generate per-sample dropout keys for vmap
        batch_size = x.shape[0]
        if train:
            dropout_keys = jax.random.split(key_dropout, batch_size)
        else:
            dropout_keys = jnp.zeros((batch_size, 2), dtype=jnp.uint32)  # dummy, won't be used

        x_pred, cosmo_pred = jax.vmap(
            model.x_pred, in_axes=(0, 0, 0, 0, None, 0)
        )(xt, cosmot, t, cond, train, dropout_keys)
        
        vx = (x - xt) / jnp.clip((1 - t[:,None,None,None]), a_min=0.05)
        vx_pred = (x_pred - xt) / jnp.clip((1 - t[:,None,None,None]), a_min=0.05)

        vcosmo = (cosmo - cosmot) / jnp.clip((1 - t[:,None]), a_min=0.05)
        vcosmo_pred = (cosmo_pred - cosmot) / jnp.clip((1 - t[:,None]), a_min=0.05)

        loss_x = jnp.mean((vx - vx_pred)**2, axis=(-1,-2,-3))
        loss_cosmo = jnp.mean((vcosmo - vcosmo_pred)**2, axis=-1)

        if self.mixture:
            total_loss = (loss_x + lambda_cosmo * loss_cosmo).mean()
        else:
            total_loss = jnp.sum((vx - vx_pred)**2, axis=(-1,-2,-3)) + jnp.sum((vcosmo - vcosmo_pred)**2, axis=-1)
            total_loss = total_loss / (x.shape[-1]*x.shape[-2]*x.shape[-3] + cosmo.shape[-1])
            total_loss = total_loss.mean()

        return total_loss, (jnp.mean(loss_x), jnp.mean(loss_cosmo))