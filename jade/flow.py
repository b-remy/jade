import jax
import jax.numpy as jnp
from flax import nnx


class Denoiser(nnx.Module):
    def __init__(self, model: nnx.Module, cfg: dict):
        self.model = model
        self.cfg = cfg

        self.t_eps = cfg.get("sampling", {}).get("t_eps", 0.05)

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

    def forward_coupling(self, x, cosmo, t, key):
        alpha_t = t
        sigma_t = 1.0 - t

        noise_x = jax.random.normal(key, shape=x.shape)
        noise_cosmo = jax.random.normal(key, shape=cosmo.shape)

        xt = alpha_t * x + sigma_t * noise_x
        cosmot = alpha_t * cosmo + sigma_t * noise_cosmo

        return xt, cosmot


class FlowLoss(nnx.Module):
    def __init__(self, cfg: dict):
        self.t_eps = cfg["diffusion"]["t_eps"]
        self.mu = cfg["diffusion"]["mu"]
        self.sigma = cfg["diffusion"]["sigma"]
        self.mixture = cfg["loss"].get("mixture", False)

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

        xt = alpha_t[:, None, None, None] * x + sigma_t[:, None, None, None] * noise_x
        cosmot = alpha_t[:, None] * cosmo + sigma_t[:, None] * noise_cosmo

        # Generate per-sample dropout keys for vmap
        batch_size = x.shape[0]
        if train:
            dropout_keys = jax.random.split(key_dropout, batch_size)
        else:
            dropout_keys = jnp.zeros((batch_size, 2), dtype=jnp.uint32)  # dummy, won't be used

        x_pred, cosmo_pred = jax.vmap(model.x_pred, in_axes=(0, 0, 0, 0, None, 0))(
            xt, cosmot, t, cond, train, dropout_keys
        )

        vx = (x - xt) / jnp.clip((1 - t[:, None, None, None]), a_min=0.05)
        vx_pred = (x_pred - xt) / jnp.clip((1 - t[:, None, None, None]), a_min=0.05)

        vcosmo = (cosmo - cosmot) / jnp.clip((1 - t[:, None]), a_min=0.05)
        vcosmo_pred = (cosmo_pred - cosmot) / jnp.clip((1 - t[:, None]), a_min=0.05)

        loss_x = jnp.mean((vx - vx_pred) ** 2, axis=(-1, -2, -3))
        loss_cosmo = jnp.mean((vcosmo - vcosmo_pred) ** 2, axis=-1)

        if self.mixture:
            total_loss = (loss_x + lambda_cosmo * loss_cosmo).mean()
        else:
            total_loss = jnp.sum((vx - vx_pred) ** 2, axis=(-1, -2, -3)) + jnp.sum((vcosmo - vcosmo_pred) ** 2, axis=-1)
            total_loss = total_loss / (x.shape[-1] * x.shape[-2] * x.shape[-3] + cosmo.shape[-1])
            total_loss = total_loss.mean()

        return total_loss, (jnp.mean(loss_x), jnp.mean(loss_cosmo))
