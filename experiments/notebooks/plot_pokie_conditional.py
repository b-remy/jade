import argparse
import glob
import os
import re

import numpy as np
import torch
import matplotlib.pyplot as plt

from pokie import pokie, pokie_bootstrap, get_device


def load_samples(samples_dir):
    sample_files = sorted(
        glob.glob(os.path.join(samples_dir, "cosmo_samples_job_*.npy")),
        key=lambda p: int(re.search(r"_(\d+)\.npy$", p).group(1)),
    )
    if not sample_files:
        raise FileNotFoundError(
            f"No cosmo_samples_job_*.npy files found in {samples_dir}"
        )

    samples, truths = [], []
    for sf in sample_files:
        job_id = re.search(r"_(\d+)\.npy$", sf).group(1)
        tf = os.path.join(samples_dir, f"true_cosmo_job_{job_id}.npy")
        if not os.path.exists(tf):
            print(f"warning: skipping job {job_id}, missing {tf}")
            continue
        samples.append(np.load(sf))
        truths.append(np.load(tf))

    samples = np.concatenate(samples, axis=0)
    truths = np.concatenate(truths, axis=0)
    return samples, truths


def main():
    parser = argparse.ArgumentParser(
        description="Compute the cosmology-only Pokie score from saved "
                    "posterior samples (as written by tarp_conditional.py)."
    )
    parser.add_argument(
        "samples_dir",
        nargs="?",
        default="tarp_results/conditional",
        help="Folder containing cosmo_samples_job_*.npy and true_cosmo_job_*.npy",
    )
    parser.add_argument("--output", default="tarp_results/pokie_conditional.pdf",
                        help="Output figure path (pdf/png).")
    parser.add_argument("--num-bootstrap", type=int, default=100)
    parser.add_argument("--num-runs", type=int, default=100,
                        help="Number of Monte Carlo runs inside each pokie call.")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="Skip the bootstrap and just compute a point estimate.")
    args = parser.parse_args()

    samples, truths = load_samples(args.samples_dir)
    print(f"Loaded samples {samples.shape}, truths {truths.shape}")

    # pokie expects posterior shape (M, T, S, q) and truth shape (T, q).
    # samples is (T, S, q), truths is (T, q). Add a single model axis.
    truth_t = torch.as_tensor(truths, dtype=torch.float32)
    posterior_t = torch.as_tensor(samples, dtype=torch.float32).unsqueeze(0)

    device = get_device()
    print(f"Using device {device}")

    if args.no_bootstrap:
        score = pokie(truth_t, posterior_t,
                      num_runs=args.num_runs, device=device)
        score_np = score.detach().cpu().numpy()
        print(f"Pokie score: {score_np[0]:.4f}")
        boot = None
    else:
        boot = pokie_bootstrap(truth_t, posterior_t,
                               num_bootstrap=args.num_bootstrap,
                               num_runs=args.num_runs,
                               device=device)
        boot = boot.detach().cpu().numpy()[:, 0]  # single model
        print(f"Pokie score: {boot.mean():.4f} +/- {boot.std():.4f} "
              f"(over {args.num_bootstrap} bootstraps)")

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.axvline(2 / 3, ls="--", color="k", label="Well-calibrated (2/3)")
    ax.axvline(1 / 2, ls=":", color="gray", label="Misaligned (1/2)")

    if boot is not None:
        ax.hist(boot, bins=20, color="tab:blue", alpha=0.6,
                edgecolor="tab:blue", label="Bootstrap")
        mean = boot.mean()
        std = boot.std()
        ax.axvline(mean, color="tab:blue", lw=2, label=f"Mean = {mean:.3f}")
        for k in (1, 2, 3):
            ax.axvspan(mean - k * std, mean + k * std,
                       color="tab:blue", alpha=0.08)
    else:
        ax.axvline(score_np[0], color="tab:blue", lw=2,
                   label=f"Pokie = {score_np[0]:.3f}")

    ax.set_xlabel("Pokie score")
    ax.set_ylabel("Count")
    ax.set_xlim(0.45, 0.75)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output)
    print(f"Wrote {args.output}")
    base, _ = os.path.splitext(args.output)
    png_output = base + ".png"
    fig.savefig(png_output, dpi=200)
    print(f"Wrote {png_output}")


if __name__ == "__main__":
    main()
