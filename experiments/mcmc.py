import os
import argparse
import yaml

import jax
import jax.numpy as jnp

from flax import nnx
import orbax.checkpoint as ocp

import matplotlib.pyplot as plt
import numpy as np

from datasets import load_from_disk

import yaml
import pickle

import jax_cosmo as jc
import sbi_lens

import numpyro
import numpyro.distributions as dist
from numpyro import sample
from numpyro.handlers import condition, reparam, seed, trace
from numpyro.infer.reparam import LocScaleReparam, TransformReparam

from sbi_lens.config import config_lsst_y_10
from sbi_lens.simulator.LogNormal_field import lensingLogNormal

from functools import partial

def main():
    parser = argparse.ArgumentParser(
        description="NUTS reference chain for the Figure 5 comparison. Writes "
                    "to a fresh directory by default so the committed "
                    "mcmc_log_normal/ reference is not overwritten."
    )
    parser.add_argument("--out", default="./mcmc_log_normal_traced")
    parser.add_argument("--num-results", type=int, default=3_000)
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--num-chains", type=int, default=10)
    args = parser.parse_args()

    save_dir = args.out
    os.makedirs(save_dir, exist_ok=True)

    # sample at Planck15 fiducial cosmology
    cosmo = jc.parameters.Planck15()

    key = jax.random.key(0)    

    # generate mocked observation
    sigma_e = config_lsst_y_10.sigma_e
    gals_per_arcmin2 = config_lsst_y_10.gals_per_arcmin2
    nbins = config_lsst_y_10.nbins
    a = config_lsst_y_10.a
    b = config_lsst_y_10.b
    z0 = config_lsst_y_10.z0
    N = 128
    map_size = 5
    with_noise = True

    model_log_normal = partial(
        lensingLogNormal,
        N=N,
        map_size=map_size,
        gal_per_arcmin2=gals_per_arcmin2,
        sigma_e=sigma_e,
        nbins=nbins,
        a=a,
        b=b,
        z0=z0,
        model_type='lognormal',
        lognormal_shifts='LSSTY10',
        with_noise=with_noise,
        )

    cond_model = seed(model_log_normal, key)
    cond_model = condition(
        cond_model,
        {
            "omega_c": cosmo.Omega_c,
            "omega_b": cosmo.Omega_b,
            "sigma_8": cosmo.sigma8,
            "h_0": cosmo.h,
            "n_s": cosmo.n_s,
            "w_0": cosmo.w0,
        },
    )

    params_name = ["omega_c", "omega_b", "sigma_8", "h_0", "n_s", "w_0"]

    model_trace = trace(cond_model).get_trace()
    sample = {
        "theta": jnp.stack(
            [model_trace[name]["value"] for name in params_name], axis=-1
        ),
        "y": model_trace["y"]["value"],
    }

    with open(os.path.join(save_dir, "mcmc_log_obs_truth.pkl"), "wb") as f:
        pickle.dump(sample, f)
    
    obs = sample["y"]
    
    # get reference posterior samples

    # initialize from the truth
    init_values = {k: model_trace[k]['value'] for k in ['z', 'omega_c', 'sigma_8', 'omega_b', 'h_0', 'n_s', 'w_0']}

    num_results = args.num_results
    num_warmup = args.num_warmup
    num_chains = args.num_chains
    max_tree_depth = 6
    step_size = 1e-2

    # Build the NUTS sampler inline (mirrors
    # get_reference_sample_posterior_full_field) so we can collect the
    # `num_steps` diagnostic, i.e. the number of leapfrog steps per iteration.
    # Each leapfrog step is one gradient evaluation of the model = one
    # simulator call, so summing num_steps gives the exact simulator-call count.
    def config(x):
        if type(x["fn"]) is dist.TransformedDistribution:
            return TransformReparam()
        elif (
            type(x["fn"]) is dist.Normal or type(x["fn"]) is dist.TruncatedNormal
        ) and ("decentered" not in x["name"]):
            return LocScaleReparam(centered=0)
        else:
            return None

    observed_model = condition(model_log_normal, {"y": obs})
    observed_model_reparam = reparam(observed_model, config=config)

    nuts_kernel = numpyro.infer.NUTS(
        model=observed_model_reparam,
        init_strategy=numpyro.infer.init_to_value(values=init_values),
        max_tree_depth=max_tree_depth,
        step_size=step_size,
    )
    mcmc = numpyro.infer.MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_results,
        num_chains=num_chains,
        chain_method="vectorized",
        progress_bar=True,
    )

    # Warmup is run separately with collect_warmup=True because mcmc.run() does
    # not expose extra fields for the warmup phase (whose trajectories are often
    # the deepest, during step-size adaptation).
    mcmc.warmup(key, extra_fields=("num_steps",), collect_warmup=True)
    warmup_num_steps = np.asarray(mcmc.get_extra_fields()["num_steps"])

    mcmc.run(mcmc.post_warmup_state.rng_key, extra_fields=("num_steps",))
    sample_num_steps = np.asarray(mcmc.get_extra_fields()["num_steps"])

    n_warmup_calls = int(warmup_num_steps.sum())
    n_sample_calls = int(sample_num_steps.sum())
    n_total_calls = n_warmup_calls + n_sample_calls
    n_iters = num_chains * (num_warmup + num_results)

    print(
        f"[sim-call tracer] chains={num_chains} warmup={num_warmup} "
        f"results={num_results} max_tree_depth={max_tree_depth}"
    )
    print(f"[sim-call tracer] warmup   gradient evals: {n_warmup_calls:,}")
    print(f"[sim-call tracer] sampling gradient evals: {n_sample_calls:,}")
    print(f"[sim-call tracer] TOTAL    gradient evals: {n_total_calls:,}")
    print(f"[sim-call tracer] mean leapfrog / iter   : {n_total_calls / n_iters:.1f}")

    samples_ = mcmc.get_samples()
    samples_mcmc = jnp.stack(
        [samples_[name] for name in params_name], axis=-1
    )

    diagnostics = {
        "warmup_num_steps": warmup_num_steps,
        "sample_num_steps": sample_num_steps,
        "n_warmup_calls": n_warmup_calls,
        "n_sample_calls": n_sample_calls,
        "n_total_calls": n_total_calls,
    }
    with open(os.path.join(save_dir, "mcmc_log_num_steps.pkl"), "wb") as f:
        pickle.dump(diagnostics, f)

    with open(os.path.join(save_dir, "mcmc_log_posterior_samples.pkl"), "wb") as f:
        pickle.dump(samples_mcmc, f)

if __name__ == "__main__":
    main()
