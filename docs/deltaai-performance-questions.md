# Speeding up JADE on NCSA DeltaAI: questions before we start

We have been asked to help JADE run faster on NCSA DeltaAI. Before scoping any
changes we need to know which workload matters and what "faster" means to you.
This note lists the questions, and what a first read of the code suggests. We
have measured nothing yet, so every observation below is a hypothesis to check.

## Questions for the developers

1. **Which stage should be faster?** Training (stage 1 `configs/hybrid.yaml`,
   stage 2 `configs/finetune.yaml`, or both), `sample_calibration.py`,
   `sample_posterior.py`, dataset generation (`lognormal_dataset.py`), the MCMC
   baseline, or the joint TARP/MIRA metrics? How long does it take today, and on
   how many DeltaAI nodes and GPUs?
2. **What does "faster" mean for you?** Shorter time to results, fewer allocation
   hours spent, or fitting inside the queue walltime limit?
3. **How closely must results match the paper?** Must the published numbers be
   reproduced exactly? Or are statistically equivalent changes acceptable, such as
   bf16 matmuls during sampling, a different solver or step count, or a
   different random-number stream?
4. **How do you run on DeltaAI today?** Could you share your SLURM scripts?
   `.gitignore` excludes `*.slurm`, so they are not in the repository. Which JAX,
   CUDA and torch versions do you use on the aarch64 nodes? Do you already stage
   the dataset on node-local NVMe for training, as `paper.md` suggests?
5. **Can we get access to benchmark?** We would need the allocation, the
   `sbi_lens_full` dataset and the paper checkpoint, so we can measure before we
   change anything.

## What the code suggests so far

The repository has several independent GPU workloads. Each can be optimized on
its own, so we would pick one first based on your answers.

### Training (`experiments/train.py`)

- `paper.md` already notes that training is IO-bound on the parallel filesystem,
  and that staging to node-local NVMe gives roughly a 3x throughput gain.
- A rough FLOP count for `JADE_B_16` (about 86M parameters, a few hundred tokens
  per sample) over both stages (about 750 epochs of about 95k samples) comes to
  tens of GPU-hours on GH200s at reasonable utilization. If wall-clock is much
  longer than that, the time is probably going somewhere other than the model
  math. Likely candidates:
  - reading from the shared filesystem;
  - the torch `DataLoader` handing numpy batches to JAX on the host, one step at
    a time, with no device prefetch;
  - `distributed.use_sharding: true` shards the parameters across GPUs (FSDP).
    For a model this small, plain data parallelism (replicated parameters,
    sharded batch) is usually cheaper, because FSDP adds collective
    communication at every layer.
- Neither config sets `model.attn_impl`, so training uses the `"dense"`
  attention path. `jade/nn.py` already supports `"xla"` and `"cudnn"`
  (flash attention) backends.

### Calibration sampling (`experiments/sample_calibration.py`)

- The paper workload is 500 observations x 500 draws x 128 Heun steps, and each
  step costs 2 network evaluations.
- Each array task loops over its 100 observations one at a time on a single GPU.
  A DeltaAI node has 4 GPUs, so each task probably uses one GPU and leaves the
  other three idle unless the launcher packs several tasks per node.
- Sampling never sets `jax_default_matmul_precision` to bfloat16, although
  training does. It also uses the dense attention path.
- The solver (Heun) and the step count (128) have not been tuned for cost
  against accuracy. Any change here is subject to question 3.

### Joint TARP/MIRA metrics (`plot_tarp.py`, `plot_mira.py`)

- The joint metrics build a single `(500, 500, 81926)` tensor of about 76 to
  82 GB. That only just fits in one GH200's 96 GB of HBM, so this stage is more
  likely limited by memory than by speed.

## Possible first step if the answer is "we are not sure"

If it is unclear which stage dominates, we could profile each stage on one
DeltaAI node first. That means wall-clock, GPU utilization, and time spent in
data loading versus compute. The measurements would then show where
optimization pays off.
