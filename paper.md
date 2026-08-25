# Reproducing arXiv:2606.31988

> Benjamin Remy, Chihway Chang, Rebecca Willett,
> *Joint inference of weak lensing convergence map and cosmology with diffusion models*,
> [arXiv:2606.31988](https://arxiv.org/abs/2606.31988)

## The paper model

Every result in the paper comes from **one training run**:

| | |
|---|---|
| wandb run | `run-20260507_170014-7hnur00g` (project `b-remy/jade`) |
| launched from | commit `d858997`, `train.py --config configs/finetune.yaml` |
| finetuned from | `run-20260414_232810-e06v6sdj` (`configs/hybrid.yaml`, from scratch) |
| architecture | `JADE_B_16` — 12 layers, 768 hidden, 12 heads, patch size 8, 16 cosmology tokens |
| objective | flow matching / v-prediction (`jade/flow.py`), *not* the score-matching formulation |
| training set | `sbi_lens_full` — 99,999 log-normal simulations, 128×128, 5×5 deg², 5 tomographic bins, wCDM |

Results use the `JADE_B_16_latest` checkpoint of that run.

## Artifact locations

Paths are resolved from the installed package rather than the working directory,
so scripts behave the same wherever they are launched. Each can be redirected
with an environment variable, which is how the datasets, checkpoints and sample
dumps — tens of GB — are kept outside the repository.

| variable | default | holds |
|---|---|---|
| `JADE_DATASET_PATH` | `experiments/sbi_lens_full` | training set |
| `JADE_WANDB_DIR` | `experiments/wandb` | run store, one directory per run |
| `JADE_RESULTS_DIR` | `experiments/results` | posterior dumps and calibration scores |
| `JADE_FIGURES_DIR` | `experiments/figures` | figures and the arrays behind them |
| `JADE_MCMC_REF_DIR` | `experiments/mcmc_log_normal` | reference chain |

## 1. Training set

Log-normal convergence fields from `sbi_lens`, generated in parallel shards:

```bash
python lognormal_dataset.py --total-samples 100000 --batch-size 100 \
    --job-id $SLURM_ARRAY_TASK_ID --output-dir $JADE_DATASET_PATH
```

## 2. Train

Stage 1 from scratch, then stage 2 finetuning from it. `configs/finetune.yaml`
carries a `params_path` pointing at the stage-1 checkpoint; update it to your own
stage-1 run.

```bash
python train.py --config configs/hybrid.yaml     # ran 547 epochs
python train.py --config configs/finetune.yaml   # ran 202 epochs
```

Training is IO-bound on a parallel filesystem. `JADE_DATASET_PATH` exists partly
so the dataset can be staged on node-local NVMe first, which is worth roughly a
3× throughput difference.

## 3. MCMC baseline

The NUTS reference for the Figure 5 contours. The chain used in the paper is
committed under `experiments/mcmc_log_normal/`, so this only needs rerunning to
regenerate it — and it writes elsewhere by default so it cannot clobber the
committed copy.

```bash
python mcmc.py --out ./mcmc_log_normal_traced
```

## 4. Figures 2, 3, 5, 6

Sampling and plotting are separate: draw once, then plot as often as you like
without a GPU.

```bash
python sample_posterior.py       # 512 draws -> posterior.npz
python plot_paper_figures.py     # -> contour_plot, joint_posterior_samples,
                                 #    power-spectra-posterior, one-point-function
```

`sample_posterior.py` reads its observation from `mcmc_log_normal/` rather than
drawing a fresh one. This is deliberate: the diffusion posterior and the NUTS
chain have to be conditioned on the *same* $\gamma_\mathrm{obs}$.

`--diagnostics` adds a relative-$C_\ell$ panel and a cross-correlation
coefficient panel, neither of which is in the paper.

## 5. Calibration — Figure 4 and the abstract's MIRA score

500 held-out observations with 500 posterior draws each, as 5 array tasks of 100:

```bash
sbatch --array=0-4 ...   # python sample_calibration.py
python plot_tarp.py --space cosmo            # Figure 4
python plot_mira.py --space joint            # 0.635 +/- 0.017
python plot_mira.py --space cosmo
python plot_tarp.py --space joint            # needs a GPU: the tensor is ~82 GB
```

`sample_calibration.py` evaluates on the held-out test split only, reproducing the
exact `train_test_split(test_size=val_split, seed=shuffle_seed)` used during
training.

The joint metrics build a single `(500, 500, 81926)` tensor — about 76–82 GB — so
they want a large-memory GPU.
