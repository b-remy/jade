import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial

from jade.lensing import Operator
from jade.diffusion import VESDE

class Denoiser(nnx.Module):
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

class PosteriorDenoiser(Denoiser):
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