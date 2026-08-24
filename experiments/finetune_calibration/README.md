# Calibration finetuning (TARP-CvM)

Finetunes the trained JADE model with a **differentiable TARP calibration
regularizer** added to the flow-matching loss:

```
total_loss = flow_loss + calibration.lambda_tarp * tarp_cvm_loss
```

TARP (Lemos et al., ICML 2023) tests whether the model's posteriors are
statistically well-calibrated. The soft-credibility + closed-form Cramér–von
Mises core is ported from `../../../calibrated-sbi` (`src/losses/tarp_cvm.py`,
`src/losses/cvm.py`).

## Files
- `train_calibration.py` — copy of `experiments/train_sharding.py` with the
  calibration term wired in (a second, calibrated train step; base loop
  unchanged).
- `tarp_loss.py` — the calibration loss: posterior sampling + soft TARP + CvM.
- `configs/calibration.yaml` — config (finetune schedule + `calibration:` block).

## How the TARP term works here
`calibrated-sbi` gets a posterior sample from one cheap flow pass. JADE is a
**joint** flow-matching model over (field, cosmology), so a posterior draw of
cosmology requires integrating the joint flow ODE with the observation as
conditioning. We do that with **diffrax** (Euler, `num_steps`) under a
continuous adjoint (`BacksolveAdjoint`, O(1) memory) so the calibration gradient
backpropagates through the whole sampler.

Each training example `(cond_i = noisy field, θ_i = true cosmo)` is already a
valid TARP tuple. Per calibrated step we take the first `n_cal` examples of the
batch, draw `M` posterior cosmologies each, and compute the CvM distance of the
soft credibility values to Uniform(0,1) against `R` reference points.

## Cost
A calibrated step costs ~ `n_cal * M * num_steps` extra network evals **plus the
adjoint backward** — much more than a plain step. Control it with `n_cal`, `M`,
`num_steps`, and `every_n_steps` (apply the term only every N steps). Start
small (defaults: `n_cal=16, M=16, num_steps=50, every_n_steps=1`,
`lambda_tarp=1.0`) and tune.

## Run
Launch from `experiments/` — `data.dataset_path` (`sbi_lens_full`) and
`logging.mcmc_ref_dir` (`mcmc_log_normal`) are relative and resolve from there,
not from this folder:

```bash
cd /u/bremy/repos/jade/experiments
uv run python finetune_calibration/train_calibration.py \
    --config finetune_calibration/configs/calibration.yaml
```

New wandb metrics: `train/tarp`, `train/flow_loss` (logged on calibrated steps).

## Checkpoint safety
This run only ever **reads** the base model at `params_path`. Checkpoints are
written to `<wandb.run.dir>/checkpoints` — a fresh `wandb/run-<timestamp>-<id>/`
directory created per launch — so existing runs are never touched. The script
asserts `checkpoint_dir != params_path` at startup as a backstop, and prints
both paths before training.

## Key config knobs (`calibration:` block)
| key | meaning |
|-----|---------|
| `enabled` | turn the TARP term on/off (off ⇒ identical to `train_sharding.py`) |
| `lambda_tarp` | weight of the TARP term relative to the flow loss |
| `n_cal` | observations used for the CvM ECDF (≤ `training.batch_size`) |
| `M` | posterior draws per observation |
| `R` | TARP reference points |
| `tau` | sigmoid temperature for the soft indicator |
| `num_steps` | diffrax Euler steps for the posterior sampler |
| `every_n_steps` | apply the calibration term every N steps |
| `adjoint` | `backsolve` (O(1) memory) \| `recursive` \| `direct` |
