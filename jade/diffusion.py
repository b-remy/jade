"""EDM-style diffusion model for joint (field, cosmology) inference.

Implements the preconditioning, loss, and samplers from:
    Karras et al. (2022) "Elucidating the Design Space of Diffusion-Based
    Generative Models" (https://arxiv.org/abs/2206.00364)

The same network is used jointly for the lensing field x and the cosmological
parameters cosmo. Both are assumed to be standardized (sigma_data ~ 1).
"""

import jax
import jax.numpy as jnp
from flax import nnx


class VESDE(nnx.Module):
    r"""Variance-exploding noise schedule.

    .. math:: \sigma(t) = \exp\bigl(\log\sigma_{\min} + t(\log\sigma_{\max} - \log\sigma_{\min})\bigr)
    """

    def __init__(self, sigma_min: float = 2e-3, sigma_max: float = 80.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_sigma_min = jnp.log(sigma_min)
        self.log_sigma_max = jnp.log(sigma_max)

    def sigma(self, t):
        return jnp.exp(self.log_sigma_min + (self.log_sigma_max - self.log_sigma_min) * t)


class Preconditioning:
    r"""EDM (Karras et al. 2022) preconditioning coefficients.

    .. math::
        D_\theta(x; \sigma) = c_\text{skip}(\sigma)\, x
            + c_\text{out}(\sigma)\, F_\theta\bigl(c_\text{in}(\sigma)\, x;\, c_\text{noise}(\sigma)\bigr)
    """

    def __init__(self, sigma_data: float = 1.0):
        self.sigma_data = sigma_data

    def c_skip(self, sigma):
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / jnp.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def c_in(self, sigma):
        return 1.0 / jnp.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def c_noise(self, sigma):
        return 0.25 * jnp.log(sigma)

    def lambda_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2


class Denoiser(nnx.Module):
    r"""EDM-preconditioned joint denoiser for (field, cosmo).

    Wraps a raw network ``F_\theta(x, c, c_noise, cond)`` that returns
    ``(x_net, cosmo_net)`` and applies separate EDM preconditioning to each
    modality (allowing independent ``sigma_data`` for the field and the
    cosmological parameters if their post-normalization scales differ).
    """

    def __init__(self, model: nnx.Module, cfg: dict):
        self.model = model
        self.cfg = cfg

        sigma_data_x = cfg["diffusion"].get("sigma_data_x", 1.0)
        sigma_data_cosmo = cfg["diffusion"].get("sigma_data_cosmo", 1.0)
        self.precond_x = Preconditioning(sigma_data=sigma_data_x)
        self.precond_cosmo = Preconditioning(sigma_data=sigma_data_cosmo)

        self.sde = VESDE(
            sigma_min=cfg["diffusion"].get("sigma_min", 2e-3),
            sigma_max=cfg["diffusion"].get("sigma_max", 80.0),
        )

    def __call__(self, xt, cosmot, sigma, cond=None, train: bool = False, key=None):
        c_skip_x = self.precond_x.c_skip(sigma)
        c_out_x = self.precond_x.c_out(sigma)
        c_in_x = self.precond_x.c_in(sigma)

        c_skip_c = self.precond_cosmo.c_skip(sigma)
        c_out_c = self.precond_cosmo.c_out(sigma)
        c_in_c = self.precond_cosmo.c_in(sigma)

        c_noise = self.precond_x.c_noise(sigma)

        x_net, cosmo_net = self.model(
            c_in_x * xt, c_in_c * cosmot, c_noise,
            cond=cond, train=train, key=key,
        )

        x_pred = c_skip_x * xt + c_out_x * x_net
        cosmo_pred = c_skip_c * cosmot + c_out_c * cosmo_net
        return x_pred, cosmo_pred

    def x_pred(self, xt, cosmot, sigma, cond=None, train: bool = False, key=None):
        return self(xt, cosmot, sigma, cond=cond, train=train, key=key)

    def forward_diffusion(self, x, cosmo, sigma, key):
        """Add VE noise: xt = x + sigma * z, cosmot = cosmo + sigma * z."""
        key_x, key_c = jax.random.split(key, 2)
        noise_x = jax.random.normal(key_x, shape=x.shape)
        noise_cosmo = jax.random.normal(key_c, shape=cosmo.shape)
        xt = x + sigma * noise_x
        cosmot = cosmo + sigma * noise_cosmo
        return xt, cosmot


class DenoiserLoss(nnx.Module):
    r"""EDM denoiser-matching loss with log-normal :math:`\sigma` sampling.

    Per-sample :math:`\sigma \sim \exp(\mathcal{N}(P_\text{mean}, P_\text{std}^2))`.
    The loss on the denoiser output is weighted by ``lambda_weight`` so that the
    effective loss on the raw network is uniform across noise levels.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.P_mean = cfg["diffusion"].get("P_mean", -1.2)
        self.P_std = cfg["diffusion"].get("P_std", 1.2)
        self.sigma_data_x = cfg["diffusion"].get("sigma_data_x", 1.0)
        self.sigma_data_cosmo = cfg["diffusion"].get("sigma_data_cosmo", 1.0)

    def __call__(self, model, x, cosmo, key, train: bool = False, cond=None):
        key_sigma, key_nx, key_nc, key_dropout = jax.random.split(key, 4)

        # log-normal sigma per sample
        log_sigma = self.P_mean + self.P_std * jax.random.normal(key_sigma, shape=x.shape[:1])
        sigma = jnp.exp(log_sigma)

        sigma_b_x = sigma[:, None, None, None]
        sigma_b_c = sigma[:, None]

        noise_x = jax.random.normal(key_nx, shape=x.shape)
        noise_cosmo = jax.random.normal(key_nc, shape=cosmo.shape)
        xt = x + sigma_b_x * noise_x
        cosmot = cosmo + sigma_b_c * noise_cosmo

        batch_size = x.shape[0]
        if train:
            dropout_keys = jax.random.split(key_dropout, batch_size)
        else:
            dropout_keys = jnp.zeros((batch_size, 2), dtype=jnp.uint32)

        x_pred, cosmo_pred = jax.vmap(
            model.x_pred, in_axes=(0, 0, 0, 0, None, 0)
        )(xt, cosmot, sigma, cond, train, dropout_keys)

        # EDM weight: lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
        lambda_x = (sigma ** 2 + self.sigma_data_x ** 2) / (sigma * self.sigma_data_x) ** 2
        lambda_c = (sigma ** 2 + self.sigma_data_cosmo ** 2) / (sigma * self.sigma_data_cosmo) ** 2

        # Per-modality MSE (for logging — comparable across runs)
        loss_x = (lambda_x * jnp.mean((x_pred - x) ** 2, axis=(-1, -2, -3))).mean()
        loss_cosmo = (lambda_c * jnp.mean((cosmo_pred - cosmo) ** 2, axis=-1)).mean()

        # Total loss: sum of squared errors over ALL elements (field + cosmo),
        # normalized by the total element count. Matches FlowLoss(mixture=False)
        # so cosmo (6 dims) and field (128*128*5 dims) get equal per-element
        # weight rather than equal per-modality weight.
        n_field = x.shape[-1] * x.shape[-2] * x.shape[-3]
        n_cosmo = cosmo.shape[-1]
        se_x = lambda_x * jnp.sum((x_pred - x) ** 2, axis=(-1, -2, -3))
        se_c = lambda_c * jnp.sum((cosmo_pred - cosmo) ** 2, axis=-1)
        total_loss = ((se_x + se_c) / (n_field + n_cosmo)).mean()

        return total_loss, (loss_x, loss_cosmo)


def karras_sigma_schedule(num_steps: int, sigma_min: float, sigma_max: float, rho: float = 7.0):
    r"""Karras :math:`\rho`-discretized noise schedule (Eq. 5 in EDM).

    Returns an array of length ``num_steps + 1`` with ``sigmas[-1] = 0``.
    """
    i = jnp.arange(num_steps, dtype=jnp.float32)
    sigmas = (
        sigma_max ** (1.0 / rho)
        + i / (num_steps - 1) * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho
    return jnp.concatenate([sigmas, jnp.zeros(1, dtype=sigmas.dtype)])


class HeunSampler(nnx.Module):
    r"""Deterministic EDM Heun sampler (Algorithm 1 of Karras et al. 2022).

    Operates on a single (x, cosmo) sample; vmap to batch.
    Inputs ``x0`` and ``cosmo0`` are unit-Gaussian latents; they are scaled by
    ``sigma_max`` internally to initialize the reverse process.
    """

    def __init__(
        self,
        model: nnx.Module,
        num_steps: int = 18,
        sigma_min: float = 2e-3,
        sigma_max: float = 80.0,
        rho: float = 7.0,
    ):
        self.model = model
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigmas = karras_sigma_schedule(num_steps, sigma_min, sigma_max, rho)

    def _denoise(self, xt, cosmot, sigma, cond):
        return self.model.x_pred(xt, cosmot, sigma, cond=cond, train=False)

    def __call__(self, x0, cosmo0, cond=None, key=None):
        sigmas = self.sigmas

        xt = x0 * sigmas[0]
        cosmot = cosmo0 * sigmas[0]

        def heun_step(carry, idx):
            xt, cosmot = carry
            sigma = sigmas[idx]
            sigma_next = sigmas[idx + 1]
            dt = sigma_next - sigma

            x_pred, cosmo_pred = self._denoise(xt, cosmot, sigma, cond)
            d_x = (xt - x_pred) / sigma
            d_c = (cosmot - cosmo_pred) / sigma

            xt_e = xt + dt * d_x
            cosmot_e = cosmot + dt * d_c

            x_pred2, cosmo_pred2 = self._denoise(xt_e, cosmot_e, sigma_next, cond)
            d_x2 = (xt_e - x_pred2) / sigma_next
            d_c2 = (cosmot_e - cosmo_pred2) / sigma_next

            xt_n = xt + dt * 0.5 * (d_x + d_x2)
            cosmot_n = cosmot + dt * 0.5 * (d_c + d_c2)
            return (xt_n, cosmot_n), None

        # Heun steps from sigma_0 down to sigma_{N-1} (last with sigma_next > 0)
        (xt, cosmot), _ = jax.lax.scan(
            heun_step, (xt, cosmot), jnp.arange(self.num_steps - 1)
        )

        # Final Euler step into sigma = 0 (Heun is undefined there)
        sigma = sigmas[-2]
        sigma_next = sigmas[-1]
        x_pred, cosmo_pred = self._denoise(xt, cosmot, sigma, cond)
        d_x = (xt - x_pred) / sigma
        d_c = (cosmot - cosmo_pred) / sigma
        xt = xt + (sigma_next - sigma) * d_x
        cosmot = cosmot + (sigma_next - sigma) * d_c
        return xt, cosmot


class DDPMSampler(nnx.Module):
    r"""Stochastic DDPM-style sampler for the VE reverse SDE.

    .. math::
        x_s = x_t - \tau (x_t - D(x_t;\sigma_t)) + \sigma_s \sqrt{\tau}\, \epsilon,
        \quad \tau = 1 - (\sigma_s / \sigma_t)^2
    """

    def __init__(
        self,
        model: nnx.Module,
        num_steps: int = 64,
        sigma_min: float = 2e-3,
        sigma_max: float = 80.0,
        rho: float = 7.0,
    ):
        self.model = model
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigmas = karras_sigma_schedule(num_steps, sigma_min, sigma_max, rho)

    def _denoise(self, xt, cosmot, sigma, cond):
        return self.model.x_pred(xt, cosmot, sigma, cond=cond, train=False)

    def __call__(self, x0, cosmo0, cond=None, key=None):
        sigmas = self.sigmas
        keys = jax.random.split(key, self.num_steps)

        xt = x0 * sigmas[0]
        cosmot = cosmo0 * sigmas[0]

        def ddpm_step(carry, t_key):
            xt, cosmot = carry
            idx, k = t_key
            sigma_t = sigmas[idx]
            sigma_s = sigmas[idx + 1]
            tau = 1.0 - (sigma_s / sigma_t) ** 2

            x_pred, cosmo_pred = self._denoise(xt, cosmot, sigma_t, cond)

            k_x, k_c = jax.random.split(k, 2)
            eps_x = jax.random.normal(k_x, shape=xt.shape)
            eps_c = jax.random.normal(k_c, shape=cosmot.shape)

            xt_n = xt - tau * (xt - x_pred) + sigma_s * jnp.sqrt(tau) * eps_x
            cosmot_n = cosmot - tau * (cosmot - cosmo_pred) + sigma_s * jnp.sqrt(tau) * eps_c
            return (xt_n, cosmot_n), None

        # Run num_steps - 1 stochastic steps, then one final clean denoise at sigma=0
        (xt, cosmot), _ = jax.lax.scan(
            ddpm_step, (xt, cosmot), (jnp.arange(self.num_steps - 1), keys[:-1])
        )

        # Last step: denoiser at sigma_min (or smallest non-zero sigma)
        sigma_final = sigmas[-2]
        x_pred, cosmo_pred = self._denoise(xt, cosmot, sigma_final, cond)
        return x_pred, cosmo_pred
