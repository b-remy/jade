import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial

from jade.lensing import Operator
from jade.diffusion import VESDE

class FlowDenoiser(nnx.Module):
    def __init__(self, model: nnx.Module, cfg: dict):
        self.model = model
        self.cfg = cfg

        self.t_eps = cfg.get('sampling', {}).get('t_eps', 0.05)
        self.noise_scale = cfg.get('sampling', {}).get('noise_scale', 1.0)

    def __call__(self, x,  cosmo, sigma_t, train: bool = False):
        return self.model(x, cosmo, sigma_t, train)

    def x_pred(self, xt, cosmot, t, train: bool = False):
        return self(xt, cosmot, t, train)

    def v_pred(self, xt, cosmot, t, train: bool = False):
        x_pred, cosmo_pred = self.x_pred(xt, cosmot, t, train)
        
        # Compute velocity: v = (x - z) / (1 - t)
        v_x = (x_pred - xt) / jnp.clip(1.0 - t, a_min=self.t_eps)
        v_cosmo = (cosmo_pred - cosmot) / jnp.clip(1.0 - t, a_min=self.t_eps)

        return v_x, v_cosmo

    def forward_coupling(self, x, cosmo, t, key):
        alpha_t = 1.0 - t
        sigma_t = t

        noise_x = jax.random.normal(key, shape=x.shape)
        noise_cosmo = jax.random.normal(key, shape=cosmo.shape)

        xt = alpha_t * x + sigma_t * noise_x
        cosmot = alpha_t * cosmo + sigma_t * noise_cosmo

        return xt, cosmot

class PosteriorDenoiser(FlowDenoiser):
    def __init__(self, model: nnx.Module, cfg: dict, gamma, sigma_gamma=1.0):
        self.model = model
        self.cfg = cfg

        self.t_eps = cfg.get('sampling', {}).get('t_eps', 0.05)
        self.noise_scale = cfg.get('sampling', {}).get('noise_scale', 1.0)

        # linear solver
        # self.solve = jax.scipy.sparse.linalg.cg
        self.solve = jax.scipy.sparse.linalg.gmres
        self.tol = 1e-3
        self.maxiter = 3

        self.gamma = gamma
        self.cov_y = sigma_gamma ** 2

    def __call__(self, xt, cosmot, t, train: bool = False):

        alpha_t = 1.0 - t
        sigma_t = t

        cov_t = sigma_t**2 / jnp.clip(alpha_t, a_min=self.t_eps)

        (x, cosmo), vjp = jax.vjp(lambda x, cosmo: self.model(x, cosmo, sigma_t, False), xt, cosmot)
   
        y, A = jax.linearize(Operator, x)
 
        At_ = jax.linear_transpose(Operator, x)
        At = lambda x: next(iter(At_(x)))

        # DPS
        # cov_y_xt = lambda v: (self.cov_y*v) + cov_t*A(At(v))

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
        self.sigma= cfg['diffusion']['sigma']

    def __call__(self, model, x, cosmo, key, lambda_cosmo, train: bool = False):

        mu = self.mu
        sigma = self.sigma
        s = (jax.random.normal(key, shape=x.shape[:1]) + mu) * sigma
        t = jax.nn.sigmoid(s)

        alpha_t = t
        sigma_t = 1 - t

        noise_x = jax.random.normal(key, shape=x.shape)
        noise_cosmo = jax.random.normal(key, shape=cosmo.shape)

        xt = alpha_t[:,None,None,None] * x + sigma_t[:,None,None,None] * noise_x
        cosmot = alpha_t[:,None] * cosmo + sigma_t[:,None] * noise_cosmo

        x_pred, cosmo_pred = jax.vmap(model.x_pred, in_axes=(0,0,0,None))(xt, cosmot, t, train)

        vx = (x - xt) / jnp.clip((1 - t[:,None,None,None]), a_min=0.05)
        vx_pred = (x_pred - xt) / jnp.clip((1 - t[:,None,None,None]), a_min=0.05) 

        vcosmo = (cosmo - cosmot) / jnp.clip((1 - t[:,None]), a_min=0.05)
        vcosmo_pred = (cosmo_pred - cosmot) / jnp.clip((1 - t[:,None]), a_min=0.05)
        
        loss_x = jnp.mean((x - x_pred)**2, axis=(-1,-2,-3))
        loss_cosmo = jnp.mean((cosmo - cosmo_pred)**2, axis=-1)

        total_loss = loss_x + lambda_cosmo * loss_cosmo
        
        total_loss = total_loss.mean()

        return total_loss, (jnp.mean(loss_x), jnp.mean(loss_cosmo))