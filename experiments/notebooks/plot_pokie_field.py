"""Run Pokie on the full κ-field posterior samples written by
tarp_conditional.py. Uses ``pokie.pokie_bootstrap_efficient`` so the
distance computation never materialises the (M, T, S, q) broadcast
intermediate — at q = 128 × 128 × 5 = 81920 that would be 82 GB on GPU
and impossible on a single device.

Memory strategy
---------------
* Pre-allocate one (1, T, S, 128, 128, 5) tensor on GPU, then for each
  per-job .npy file load it into host RAM, copy into the GPU buffer, and
  free the host copy. Host RAM peak: one file (16.4 GB) at a time.
* GPU peak: posterior (82 GB) + small per-MC-run scratch (~few MB).
  Fits in 96 GB with headroom.
"""

import argparse
import glob
import os
import re
import time

import numpy as np
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
})

from pokie import get_device, pokie_bootstrap_efficient


def _job_files(samples_dir):
    """Return list of (job_id, x_path, truth_path) for every present job."""
    x_files = sorted(
        glob.glob(os.path.join(samples_dir, "x_samples_job_*.npy")),
        key=lambda p: int(re.search(r"_(\d+)\.npy$", p).group(1)),
    )
    if not x_files:
        raise FileNotFoundError(
            f"No x_samples_job_*.npy files found in {samples_dir}"
        )
    out = []
    for xf in x_files:
        job_id = int(re.search(r"_(\d+)\.npy$", xf).group(1))
        tf = os.path.join(samples_dir, f"true_x_job_{job_id}.npy")
        if not os.path.exists(tf):
            print(f"warning: skipping job {job_id}, missing {tf}")
            continue
        out.append((job_id, xf, tf))
    return out


def load_to_gpu(samples_dir, device):
    """Stream-load per-job files into pre-allocated GPU tensors.

    Returns
    -------
    posterior_gpu : (M=1, T_total, S, q) torch.float32 on ``device``
    truth_gpu     : (T_total, q) torch.float32 on ``device``
    """
    jobs = _job_files(samples_dir)
    # Peek at the first file to learn shapes.
    head = np.load(jobs[0][1], mmap_mode="r")
    obs_per_job, S, *spatial = head.shape  # (n_obs_per_job, S, 128, 128, 5)
    q = int(np.prod(spatial))
    n_jobs = len(jobs)
    T_total = obs_per_job * n_jobs
    del head

    print(f"Allocating posterior on {device}: "
          f"(1, {T_total}, {S}, {q}) float32 = "
          f"{T_total * S * q * 4 / 2**30:.1f} GiB")
    posterior_gpu = torch.empty((1, T_total, S, q), dtype=torch.float32, device=device)
    truth_gpu = torch.empty((T_total, q), dtype=torch.float32, device=device)

    for job_id, xf, tf in jobs:
        t0 = time.time()
        # Load full file into host RAM. Reshape (n_obs, S, 128, 128, 5) ->
        # (n_obs, S, q) is contiguous so it's a view, no copy.
        x = np.load(xf)
        x_flat = x.reshape(obs_per_job, S, q)
        t = np.load(tf).reshape(obs_per_job, q)
        start = job_id * obs_per_job
        end = start + obs_per_job

        posterior_gpu[0, start:end].copy_(torch.from_numpy(x_flat))
        truth_gpu[start:end].copy_(torch.from_numpy(t))
        del x, x_flat, t
        print(f"  job {job_id}: {xf.split('/')[-1]} -> GPU "
              f"({time.time() - t0:.1f}s)")

    return posterior_gpu, truth_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "samples_dir",
        nargs="?",
        default="tarp_results/conditional",
        help="Folder with x_samples_job_*.npy and true_x_job_*.npy",
    )
    parser.add_argument("--output", default="tarp_results/pokie_field.pdf",
                        help="Output figure path (pdf/png).")
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--num-bootstrap", type=int, default=100)
    parser.add_argument("--save-npz",
                        default="tarp_results/pokie_field_scores.npz",
                        help="Path to dump boot_score, mean, std as .npz.")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="Run a single pokie_efficient point estimate.")
    parser.add_argument("--no-norm", action="store_true",
                        help="Skip the scalar normalisation of truth/posterior "
                             "to [0, 1]. With normalisation off and κ values "
                             "naturally near zero, Pokie's [0,1] random "
                             "centers sit far outside the data cloud and the "
                             "score is uninterpretable. Default: norm on.")
    args = parser.parse_args()
    use_norm = not args.no_norm

    device = get_device()
    print(f"device: {device}")
    if device.type != "cuda":
        print("WARNING: running on CPU; at q=81920 this will be slow.")

    t0 = time.time()
    posterior_gpu, truth_gpu = load_to_gpu(args.samples_dir, device)
    print(f"load done in {time.time() - t0:.1f}s. "
          f"posterior {tuple(posterior_gpu.shape)}, truth {tuple(truth_gpu.shape)}")
    if device.type == "cuda":
        print(f"GPU memory used: {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    torch.manual_seed(2024)
    t0 = time.time()
    print(f"normalisation: {'scalar (norm=True)' if use_norm else 'OFF'}")
    if args.no_bootstrap:
        from pokie import pokie_efficient
        score = pokie_efficient(truth_gpu, posterior_gpu,
                                num_runs=args.num_runs, device=device,
                                norm=use_norm)
        score_np = score.detach().cpu().numpy()
        print(f"pokie_efficient score: {score_np[0]:.6f}   "
              f"(wall {time.time() - t0:.1f}s)")
        np.savez(args.save_npz, score=score_np)
        return

    boot = pokie_bootstrap_efficient(
        truth_gpu, posterior_gpu,
        num_bootstrap=args.num_bootstrap,
        num_runs=args.num_runs, device=device,
        norm=use_norm,
    )
    t_compute = time.time() - t0
    boot_np = boot.detach().cpu().numpy()[:, 0]
    mean = boot_np.mean()
    std = boot_np.std()
    print(f"pokie_bootstrap_efficient: {mean:.6f} ± {std:.6f}   "
          f"(wall {t_compute:.1f}s for {args.num_runs} MC × "
          f"{args.num_bootstrap} bootstrap)")
    np.savez(args.save_npz, boot=boot_np, mean=mean, std=std)
    print(f"wrote {args.save_npz}")

    # Plot histogram, mirroring plot_pokie_conditional.py style.
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.axvline(2 / 3, ls="--", color="k", label="Well-calibrated (2/3)")
    ax.axvline(1 / 2, ls=":", color="gray", label="Misaligned (1/2)")
    ax.hist(boot_np, bins=20, color="tab:blue", alpha=0.6,
            edgecolor="tab:blue", label="Bootstrap")
    ax.axvline(mean, color="tab:blue", lw=2, label=f"Mean = {mean:.3f}")
    for k in (1, 2, 3):
        ax.axvspan(mean - k * std, mean + k * std, color="tab:blue", alpha=0.08)
    ax.set_xlabel("Pokie score")
    ax.set_ylabel("Count")
    ax.set_xlim(0.45, 0.75)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output)
    base, _ = os.path.splitext(args.output)
    fig.savefig(base + ".png", dpi=200)
    print(f"wrote {args.output} and {base}.png")


if __name__ == "__main__":
    main()
