"""
GPU batch-size stress test for the log-normal dataset generator.

Sweeps a list of batch sizes, and for each one measures:
  - JIT compile time (first call)
  - steady-state throughput (samples/sec, median over a few timed calls)
  - peak GPU memory used

OOM (or any failure) for a batch size is caught and reported so the sweep
continues to the next size.

Run on a GPU node, e.g.:
    srun --account=benb-dtai-gh --partition=ghx4 --gpus-per-node=1 \
         --mem=20G --cpus-per-task=4 -t 00:30:00 --pty \
         bash -c 'source ~/utils/activate_jade.sh && cd /u/bremy/repos/jade/experiments && \
                  python bench_batch.py --batch-sizes 100 250 500 1000 2000 4000'
"""

# Let JAX allocate on demand so peak_bytes_in_use reflects the real footprint.
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import time
from functools import partial

import jax

from lognormal import setup_model
from utils import get_samples


def bench_one(model, batch_size, with_noise, n_timed, base_seed=0):
    """Compile + time a single batch size. Returns a result dict."""
    dev = jax.devices()[0]

    fn = jax.jit(
        partial(get_samples, model=model, batch_size=batch_size, with_noise=with_noise)
    )

    key = jax.random.key(base_seed)

    # --- compile + first run ---
    key, sub = jax.random.split(key)
    t0 = time.perf_counter()
    out = fn(key=sub)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0

    # --- steady-state timed runs ---
    times = []
    for _ in range(n_timed):
        key, sub = jax.random.split(key)
        t0 = time.perf_counter()
        out = fn(key=sub)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    times.sort()
    median_s = times[len(times) // 2]

    peak_gb = None
    try:
        peak_gb = dev.memory_stats()["peak_bytes_in_use"] / 1e9
    except Exception:
        pass

    return {
        "batch_size": batch_size,
        "compile_s": compile_s,
        "median_s": median_s,
        "samples_per_s": batch_size / median_s,
        "peak_gb": peak_gb,
    }


def main():
    p = argparse.ArgumentParser(description="GPU batch-size stress test")
    p.add_argument("--batch-sizes", type=int, nargs="+",
                   default=[100, 250, 500, 1000, 2000, 4000])
    p.add_argument("--n-timed", type=int, default=5,
                   help="timed iterations per batch size (after compile)")
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--map-size", type=float, default=5.0)
    p.add_argument("--with-noise", action="store_true")
    args = p.parse_args()

    print(f"JAX devices: {jax.devices()}")
    model = setup_model(N=args.N, map_size=args.map_size, with_noise=args.with_noise)

    results = []
    for bs in args.batch_sizes:
        print(f"\n=== batch_size={bs} ===", flush=True)
        try:
            r = bench_one(model, bs, args.with_noise, args.n_timed)
            peak = f"{r['peak_gb']:.1f} GB" if r["peak_gb"] is not None else "n/a"
            print(
                f"  compile={r['compile_s']:.1f}s  "
                f"per-batch={r['median_s']*1e3:.0f}ms  "
                f"throughput={r['samples_per_s']:.0f} samples/s  "
                f"peak={peak}",
                flush=True,
            )
            results.append(r)
        except Exception as e:
            print(f"  FAILED ({type(e).__name__}): {str(e)[:200]}", flush=True)
            # Likely OOM -> larger sizes will also fail; stop the sweep.
            break

    if results:
        best = max(results, key=lambda r: r["samples_per_s"])
        print("\n================ summary ================")
        print(f"{'batch':>8} {'throughput (samp/s)':>20} {'peak (GB)':>12}")
        for r in results:
            peak = f"{r['peak_gb']:.1f}" if r["peak_gb"] is not None else "n/a"
            print(f"{r['batch_size']:>8} {r['samples_per_s']:>20.0f} {peak:>12}")
        # samples-per-job time estimate at the fastest size
        per_job = 250_000
        eta_h = per_job / best["samples_per_s"] / 3600
        print(
            f"\nFastest: batch_size={best['batch_size']} "
            f"(~{best['samples_per_s']:.0f} samples/s).\n"
            f"At that rate, 250k samples/job ≈ {eta_h:.1f} h "
            f"(4 jobs in parallel -> ~{eta_h:.1f} h wall for 1M)."
        )


if __name__ == "__main__":
    main()
