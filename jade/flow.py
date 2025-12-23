import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial

from jade.lensing import Operator

class Denoiser(nnx.Module):
    def __init__(self, model: nnx.Module, cfg: dict):
        self.model = model
        self.cfg = cfg

    def __call__(self, x,  cosmo, sigma_t, train: bool = False):
        return self.model(x, cosmo, sigma_t, train)

class PosteriorDenoiser(nnx.Module):
    def __init__(self, model: nnx.Module, cfg: dict, gamma):
        self.model = model
        self.cfg = cfg

        # linear solver
        self.solve = jax.scipy.sparse.linalg.cg
        self.tol = 1e-3
        self.maxiter = 1

        self.gamma = gamma
        self.cov_y = 1.

    def __call__(self, xt, cosmot, sigma_t, train: bool = False):

        cov_t = sigma_t ** 2
    
        (x, cosmo), vjp = jax.vjp(lambda x, cosmo: self.model(x, cosmo, sigma_t, False), xt, cosmot)
        
        y, A = jax.linearize(Operator, x)
        
        At_ = jax.linear_transpose(Operator, x)
        At = lambda x: next(iter(At_(x)))

        # DPS
        cov_y_xt = lambda v: (self.cov_y*v) + cov_t*A(At(v))

        b = self.gamma - y
        
        v, _ = self.solve(
            A=cov_y_xt,
            b=b,
            tol=self.tol,
            maxiter=self.maxiter,
        )

        (score, _) = vjp((At(v), jnp.zeros_like(cosmo)))
        
        return x + cov_t * score, cosmo