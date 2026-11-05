import jax
import jax.numpy as jnp
from flax import nnx

from jade.lensing import Operator


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


class PosteriorDenoiser(Denoiser):
    r"""Likelihood-guided denoiser: turns the *unconditional* joint model
    p(theta, kappa) into a posterior sampler p(theta, kappa | gamma_obs).

    The conditional model in ``train.py`` learns the posterior directly, by
    feeding the observation to the network. Here the network never sees the
    data. Instead the explicit Gaussian likelihood

        gamma_obs = A kappa + n,   n ~ N(0, Sigma_y)

    is folded into the sampler, where ``A`` is the Kaiser & Squires
    convergence-to-shear operator (:func:`jade.lensing.Operator`, which also
    undoes the field normalisation) and ``Sigma_y`` is the per-bin shape-noise
    variance. The likelihood has no explicit theta dependence, so cosmology is
    only ever reached through the denoiser Jacobian, which couples the cosmology
    tokens to the field tokens (see ``guide_cosmo``).

    For the linear-Gaussian path ``x_t = alpha_t x_0 + sigma_t eps`` with
    ``alpha_t = t`` and ``sigma_t = 1 - t``, Tweedie's identity gives the
    covariance of the denoising posterior in closed form,

        Cov[x_0 | x_t] = (sigma_t^2 / alpha_t) * J,    J = d x_hat_0 / d x_t

    so ``cov_t = sigma_t^2 / alpha_t`` below is the only scalar needed. The
    guided denoised estimate is one Gaussian update of the denoiser output,

        x_hat_0  <-  x_hat_0 + Cov[x_0|x_t] * grad_{x_t} log p(gamma_obs | x_t)
                  =  x_hat_0 + cov_t * J^T A^T (Sigma_y + cov_t A M A^T)^{-1} r

    with residual ``r = gamma_obs - A x_hat_0``. The linear system is solved
    with conjugate gradients; the two methods differ only in ``M``:

    ``method="mmps"``
        ``M = J``. The full moment-matching covariance
        ``Sigma_y + A Cov[x_0|x_t] A^T``, i.e. MMPS. Each CG iteration applies
        the denoiser VJP, so a solve costs ``maxiter`` backward passes.

    ``method="dps"``
        ``M = I``. The Jacobian is dropped from the middle of the likelihood
        covariance, which is the isotropic approximation used by DPS-style
        guidance. Only FFTs inside the CG loop, so roughly ``maxiter`` times
        cheaper, at the cost of a less faithful likelihood.

    Because :meth:`Denoiser.v_pred` routes through ``__call__``, any sampler in
    :mod:`jade.sampling` drives the guided dynamics unchanged.

    Args:
        model: a :class:`Denoiser` wrapping the trained unconditional network.
        cfg: the run config, read for ``sampling.t_eps``.
        gamma: the observed shear, shape ``(2, N, N, n_bins)``.
        sigma_gamma: shape-noise std, scalar or per-bin ``(n_bins,)``.
        method: ``"mmps"`` or ``"dps"``.
        solver: ``"auto"``, ``"cg"`` or ``"gmres"``. ``"auto"`` gives GMRES for
            MMPS and CG for DPS, and this choice matters a great deal. DPS's
            operator ``Sigma_y + cov_t A A^T`` is symmetric positive definite,
            exactly CG's assumption. The MMPS operator carries the learned
            denoiser Jacobian and is neither symmetric nor definite, so CG is
            outside its assumptions there and silently returns garbage on the
            draws where it breaks down: with CG at 512 steps, 20% of MMPS draws
            diverge past 5 prior sigma and the sampled field variance is ~28x
            the prior. GMRES solves the *same* linear system without the
            symmetry assumption and removes the failure completely (0/512
            divergent, field variance within 5% of the prior). It costs about
            4x more per solve, since it orthogonalises against a stored Krylov
            basis rather than using a two-term recurrence.
        tol, maxiter: linear-solver controls.
        restart: GMRES inner Krylov dimension. jax counts ``maxiter`` in
            restarts and runs a fixed-size Arnoldi of this length inside each,
            so this, not ``maxiter``, sets the cost floor: 20 matrix-vector
            products per solve at the default, each one a denoiser VJP. The
            solve converges to a relative residual of 1e-4 or better in under 5
            iterations everywhere on the trajectory (see
            ``diagnose_solver.py``), so lowering it trades no accuracy for
            close to a proportional speedup.
        alpha_eps: floor on ``alpha_t`` in ``cov_t``, so that ``t = 0`` stays
            finite. This is the only free numerical constant in the guidance and
            ``cov_t`` reaches ``1 / alpha_eps`` at ``t = 0``, so it sets how
            strong the guidance is allowed to get in the first steps -- where
            the denoiser Jacobian is also least reliable, since the
            flow-matching time distribution ``sigmoid(N(-0.64, 0.64^2))``
            almost never samples ``t`` near 0. The two ``exps``-branch
            prototypes disagreed on it (1e-3 in ``flow.py``, 5e-2 in
            ``flow_conditional.py``); 1e-3 here matches the ``flow.py`` one.
        guide_cosmo: whether to apply the guidance update to the cosmology
            output as well as the field. The likelihood has no explicit theta
            dependence, but the Jacobian block ``d kappa_hat_0 / d theta_t`` is
            non-zero, so the complete first-order update does move theta.
            ``False``, the default, drops that term and reproduces the
            ``exps``-branch prototype exactly: only the field is updated and
            theta is informed indirectly, through the guided field at
            subsequent steps.
    """

    def __init__(
        self,
        model: nnx.Module,
        cfg: dict,
        gamma,
        sigma_gamma=1.0,
        method: str = "mmps",
        solver: str = "auto",
        tol: float = 1e-3,
        maxiter: int = 10,
        restart: int = 20,
        alpha_eps: float = 1e-3,
        guide_cosmo: bool = False,
    ):
        super().__init__(model, cfg)

        method = method.lower()
        if method not in ("mmps", "dps"):
            raise ValueError(f"method must be 'mmps' or 'dps', got {method!r}")
        self.method = method

        solver = solver.lower()
        if solver == "auto":
            solver = "gmres" if method == "mmps" else "cg"
        if solver not in ("cg", "gmres"):
            raise ValueError(f"solver must be 'auto', 'cg' or 'gmres', got {solver!r}")
        self.solver = solver
        self.solve = getattr(jax.scipy.sparse.linalg, solver)

        self.tol = tol
        self.maxiter = maxiter
        # GMRES only. jax counts `maxiter` in restarts and runs a fixed-size
        # Arnoldi of dimension `restart` inside each, so the cost floor is one
        # `restart`-length inner loop even when the solve converges on the first
        # few iterations -- which here it does.
        self.solve_kwargs = {"restart": restart} if solver == "gmres" else {}
        self.alpha_eps = alpha_eps
        self.guide_cosmo = guide_cosmo

        self.gamma = gamma
        # Broadcasts over the trailing tomographic-bin axis of gamma.
        self.cov_y = jnp.asarray(sigma_gamma) ** 2

    def __call__(self, xt, cosmot, t, cond=None, train: bool = False, key=None):
        alpha_t = t
        sigma_t = 1.0 - t

        cov_t = sigma_t**2 / jnp.clip(alpha_t, min=self.alpha_eps)

        # x_hat_0 and the denoiser Jacobian, from a single forward pass. The
        # cosmology output is carried through the VJP so guidance can reach the
        # cosmology tokens.
        (x, cosmo), vjp = jax.vjp(
            lambda x_, cosmo_: self.model(x_, cosmo_, t, cond=cond, train=False),
            xt,
            cosmot,
        )

        # A: JVP of the Kaiser & Squires operator at x_hat_0; At: its adjoint.
        # Operator is linear (FIELD_MEAN is zero), so y == A(x).
        y, A = jax.linearize(Operator, x)
        At_ = jax.linear_transpose(Operator, x)
        At = lambda v: next(iter(At_(v)))  # noqa: E731

        zeros_cosmo = jnp.zeros_like(cosmo)

        def Jt(v_field):
            """The field-block Jacobian transpose, (d x_hat_0 / d x_t)^T v."""
            return vjp((v_field, zeros_cosmo))[0]

        if self.method == "mmps":
            # Sigma_y + cov_t A J^T A^T, standing in for Sigma_y + cov_t A J
            # A^T. Only the VJP is available from a single forward pass, and J
            # is the Jacobian of a score field, hence symmetric up to the
            # network's approximation error — which is also what lets CG,
            # a symmetric-operator solver, be used here at all.
            def cov_y_xt(v):
                return self.cov_y * v + cov_t * A(Jt(At(v)))
        else:
            # DPS: drop the Jacobian, Sigma_y + cov_t A A^T.
            def cov_y_xt(v):
                return self.cov_y * v + cov_t * A(At(v))

        r = self.gamma - y

        v, _ = self.solve(A=cov_y_xt, b=r, tol=self.tol, maxiter=self.maxiter, **self.solve_kwargs)

        score_x, score_cosmo = vjp((At(v), zeros_cosmo))

        x_guided = x + cov_t * score_x
        cosmo_guided = cosmo + cov_t * score_cosmo if self.guide_cosmo else cosmo

        return x_guided, cosmo_guided


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
