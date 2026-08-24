"""Differentiable TARP-CvM calibration regularizer for JADE.

Adds a *Test of Accuracy with Random Points* (TARP; Lemos, Coogan, Hezaveh &
Perreault-Levasseur, ICML 2023) calibration penalty to the JADE flow-matching
loss. The soft-credibility + closed-form Cramer-von Mises core is ported (pure
JAX, unchanged math) from ``../calibrated-sbi``:

    src/losses/cvm.py        -> cvm_uniform
    src/losses/tarp_cvm.py   -> soft credibility + reference sampling

The one substantive difference from ``calibrated-sbi`` is how posterior draws
are obtained. There a posterior sample is a single cheap reparameterized flow
pass. JADE is a *joint* flow-matching model over (field, cosmology): a
posterior draw of cosmology requires integrating the joint flow ODE with the
observation as conditioning. We do that with ``diffrax`` (Euler, configurable
step count) under a continuous adjoint so the calibration gradient
backpropagates through the whole sampler with O(1) memory
(``diffrax.BacksolveAdjoint``).

Everything operates in the *scaled/standardized* cosmology space used during
training (i.e. ``cosmo = theta_norm * SCALE_COSMO``); distances between the true
theta, posterior draws, and references are therefore all commensurate.
"""

from functools import partial

import jax
import jax.numpy as jnp
import diffrax
from flax import nnx


# Cosmology dimensionality for JADE (Omega_c, Omega_b, sigma_8, h_0, n_s, w_0).
_COSMO_DIM = 6


# ---------------------------------------------------------------------------
# Cramer-von Mises core (verbatim from calibrated-sbi/src/losses/cvm.py)
# ---------------------------------------------------------------------------
def cvm_uniform(r):
    r"""Cramer-von Mises distance from ``{r_i}`` to Uniform(0, 1).

    Closed form::

        CvM = (1/N) sum_i r_i^2  -  (1/N^2) sum_k (2k-1) r_{(k)}  +  1/3,

    where ``r_{(k)}`` is the k-th order statistic. Zero iff ``{r_i}`` is exactly
    uniform. O(N log N), differentiable through ``jnp.sort``.
    """
    N = r.shape[0]
    r_sorted = jnp.sort(r)
    k = jnp.arange(1, N + 1, dtype=r.dtype)
    return jnp.mean(r ** 2) - jnp.sum((2 * k - 1) * r_sorted) / N ** 2 + 1.0 / 3.0


# ---------------------------------------------------------------------------
# Reference distribution for TARP
# ---------------------------------------------------------------------------
def sample_references(key, R, theta_true, posterior_bank):
    """Stratified reference points for the TARP distance test.

    Mirrors the 1/3 + 1/3 + 1/3 mixture of ``calibrated-sbi`` but avoids needing
    explicit prior-box bounds by treating the batch's own true thetas as an
    empirical draw from the prior:

      * 1/3 resampled from the batch's true thetas (empirical prior),
      * 1/3 Gaussian at the batch theta mean with the batch marginal std,
      * 1/3 from the pooled posterior-sample bank.

    Args:
        key: PRNG key.
        R: number of reference points.
        theta_true: (n_cal, D) true cosmology (standardized space).
        posterior_bank: (n_cal * M, D) pooled posterior draws.

    Returns:
        (R, D) reference points.
    """
    D = theta_true.shape[-1]
    n3 = R // 3
    n_rest = R - 2 * n3
    k_p, k_g, k_q = jax.random.split(key, 3)

    # 1/3 empirical prior (resample true thetas).
    idx_p = jax.random.randint(k_p, (n3,), 0, theta_true.shape[0])
    refs_prior = theta_true[idx_p]

    # 1/3 Gaussian at batch mean with batch marginal std.
    mean = theta_true.mean(axis=0)
    std = theta_true.std(axis=0) + 1e-6
    refs_gauss = mean + std * jax.random.normal(k_g, (n3, D))

    # 1/3 posterior-sample bank.
    idx_q = jax.random.randint(k_q, (n_rest,), 0, posterior_bank.shape[0])
    refs_q = posterior_bank[idx_q]

    return jnp.concatenate([refs_prior, refs_gauss, refs_q], axis=0)


# ---------------------------------------------------------------------------
# Soft TARP credibility + CvM (ported from calibrated-sbi/src/losses/tarp_cvm.py)
# ---------------------------------------------------------------------------
def tarp_cvm_from_samples(theta_true, samples, refs, tau):
    """TARP-CvM value given precomputed posterior draws.

    Args:
        theta_true: (n, D) true cosmology.
        samples: (n, M, D) posterior draws per observation.
        refs: (R, D) reference points.
        tau: sigmoid temperature for the soft indicator.

    Returns:
        scalar TARP-CvM distance (mean over references of CvM-to-Uniform of the
        soft credibility values).
    """
    # ||.|| via sqrt(sum_sq + eps): jnp.linalg.norm has a NaN gradient at zero,
    # which can bite when a bank reference exactly equals the sample it came from.
    eps_sq = 1e-12

    # d_true[i, s]       = ||theta_i  - theta_r^(s)||
    d_true = jnp.sqrt(
        jnp.sum((theta_true[:, None, :] - refs[None, :, :]) ** 2, axis=-1) + eps_sq
    )  # (n, R)
    # d_samples[i, j, s] = ||theta_ij - theta_r^(s)||
    d_samples = jnp.sqrt(
        jnp.sum((samples[:, :, None, :] - refs[None, None, :, :]) ** 2, axis=-1) + eps_sq
    )  # (n, M, R)

    # Soft credibility r_i^(s) = mean_j sigma((d_true - d_samples) / tau).
    diff = (d_true[:, None, :] - d_samples) / tau
    r = jax.nn.sigmoid(diff).mean(axis=1)  # (n, R)

    # CvM along the n axis for each of the R references, then average.
    cvm_per_ref = jax.vmap(cvm_uniform, in_axes=1)(r)
    return cvm_per_ref.mean()


# ---------------------------------------------------------------------------
# Posterior cosmology sampling via the JADE flow ODE (diffrax)
# ---------------------------------------------------------------------------
def sample_cosmo_posterior(model, conds, key, *, M, num_steps, adjoint):
    """Draw ``M`` posterior cosmology samples per conditioning observation.

    Integrates the JADE joint flow ODE ``dz/dt = v(z, t | cond)`` from t=0 (unit
    Gaussian latent) to t=1 (data) with a diffrax Euler solver. Only the
    cosmology component of the terminal state is returned; the field is evolved
    jointly (required, since the network denoises both together) but discarded.

    Gradients w.r.t. the model parameters flow through the whole integration via
    ``adjoint`` (use ``diffrax.BacksolveAdjoint`` for O(1) memory).

    Args:
        model: a ``jade.flow.Denoiser`` exposing ``v_pred(x, cosmo, t, cond=...)``.
        conds: (n_cal, H, W, C) conditioning observations (one per posterior).
        key: PRNG key for the initial latents.
        M: posterior draws per observation.
        num_steps: number of Euler steps for the flow ODE.
        adjoint: a ``diffrax.AbstractAdjoint`` instance.

    Returns:
        (n_cal, M, D) posterior cosmology draws (standardized space).
    """
    # Split the model so its parameters travel through diffrax `args` (and thus
    # receive adjoint gradients); the graph definition and any non-Param state
    # are captured as constants.
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    n_cal = conds.shape[0]
    field_shape = conds.shape[1:]
    cosmo_dim = _COSMO_DIM  # fixed by the problem (6 cosmological parameters)

    # CRITICAL: run ONE diffeqsolve over the whole batch of draws and vmap
    # *inside* the vector field — do not vmap diffeqsolve itself.
    #
    # BacksolveAdjoint is O(1) in the number of ODE steps, but it implements that
    # by augmenting the ODE state with an adjoint accumulator for everything in
    # `args` — including `params`. vmapping the solver therefore gives every draw
    # its own private copy of a |params|-sized accumulator: with 256 draws and
    # 130M params that is ~133GB (~248GB with the fwd/bwd pair) and OOMs
    # instantly. Batching inside the vector field keeps a single shared param
    # adjoint (~0.5GB). The draws are independent, so the dynamics are identical.
    n_draws = n_cal * M
    key_x, key_c = jax.random.split(key)
    x0 = jax.random.normal(key_x, (n_draws, *field_shape))
    cosmo0 = jax.random.normal(key_c, (n_draws, cosmo_dim))
    # Repeat each observation M times: index i*M + j == observation i, draw j,
    # which is exactly the layout the final reshape to (n_cal, M, D) expects.
    conds_rep = jnp.repeat(conds, M, axis=0)  # (n_draws, H, W, C)

    # Every traced array the vector field touches (params, non-Param state, the
    # per-draw cond) travels through `args`; diffrax cannot handle tracers
    # captured in the term's closure. Only the static `graphdef` is closed over.
    # diffrax partitions `args` by dtype, so integer state (e.g. RNG keys) gets
    # no adjoint gradient while the float params do.
    def vector_field(t, y, args):
        p, r, c = args
        x, cosmo = y
        m = nnx.merge(graphdef, p, r)
        # t is a scalar shared by every draw; map over the batch dim only.
        v_x, v_cosmo = jax.vmap(
            lambda xi, ci, cond_i: m.v_pred(xi, ci, t, cond=cond_i, train=False)
        )(x, cosmo, c)
        return (v_x, v_cosmo)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Euler(),
        t0=0.0,
        t1=1.0,
        dt0=1.0 / num_steps,
        y0=(x0, cosmo0),
        args=(params, rest, conds_rep),
        adjoint=adjoint,
        saveat=diffrax.SaveAt(t1=True),
        max_steps=num_steps + 8,
    )
    # ys = (x_saved, cosmo_saved); leading dim is the single saved (t1) point.
    cosmo_draws = sol.ys[1][0]  # (n_draws, D)
    return cosmo_draws.reshape(n_cal, M, cosmo_dim)


def tarp_cvm_loss(model, cond, theta_true, key, *, M, R, tau, num_steps, adjoint):
    """Full differentiable TARP-CvM calibration loss for JADE.

    Args:
        model: ``jade.flow.Denoiser``.
        cond: (n_cal, H, W, C) conditioning observations.
        theta_true: (n_cal, D) true cosmology matching ``cond`` (standardized).
        key: PRNG key.
        M: posterior draws per observation.
        R: number of TARP reference points.
        tau: sigmoid temperature.
        num_steps: Euler steps for the posterior sampler.
        adjoint: ``diffrax.AbstractAdjoint`` instance.

    Returns:
        scalar TARP-CvM calibration loss.
    """
    key_s, key_r = jax.random.split(key)
    samples = sample_cosmo_posterior(
        model, cond, key_s, M=M, num_steps=num_steps, adjoint=adjoint
    )  # (n_cal, M, D)
    bank = samples.reshape(-1, theta_true.shape[-1])
    refs = sample_references(key_r, R, theta_true, bank)
    return tarp_cvm_from_samples(theta_true, samples, refs, tau)


def make_adjoint(name: str):
    """Build a diffrax adjoint from a config string."""
    name = (name or "backsolve").lower()
    if name in ("backsolve", "backsolve_o1", "o1"):
        return diffrax.BacksolveAdjoint()
    if name in ("recursive", "recursive_checkpoint", "checkpoint"):
        return diffrax.RecursiveCheckpointAdjoint()
    if name == "direct":
        return diffrax.DirectAdjoint()
    raise ValueError(f"Unknown adjoint '{name}'")
