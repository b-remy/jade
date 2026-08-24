"""Artifact locations, resolved from the package rather than the cwd.

Each entry is overridable by an environment variable, so the heavy artifacts
can live outside the repo.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def _from_env(var, default):
    value = os.environ.get(var)
    return Path(value) if value else default


DATASET_DIR = _from_env("JADE_DATASET_PATH", EXPERIMENTS_DIR / "sbi_lens_full")
WANDB_DIR = _from_env("JADE_WANDB_DIR", EXPERIMENTS_DIR / "wandb")
RESULTS_DIR = _from_env("JADE_RESULTS_DIR", EXPERIMENTS_DIR / "results")
FIGURES_DIR = _from_env("JADE_FIGURES_DIR", EXPERIMENTS_DIR / "figures")
MCMC_REF_DIR = _from_env("JADE_MCMC_REF_DIR", EXPERIMENTS_DIR / "mcmc_log_normal")

# The model behind arXiv:2606.31988: configs/finetune.yaml finetuned from the
# configs/hybrid.yaml run e06v6sdj.
PAPER_RUN = "run-20260507_170014-7hnur00g"


def checkpoint_dir(run=PAPER_RUN):
    return WANDB_DIR / run / "files" / "checkpoints"
