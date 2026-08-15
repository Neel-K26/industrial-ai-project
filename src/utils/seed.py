"""Global random seed management for AMMPM experiments.

Reproducibility is a hard requirement for trustworthy industrial predictive
maintenance research: every reported RUL (Remaining Useful Life) metric,
uncertainty interval, and SHAP explanation must be traceable to a fixed seed
so results can be audited and reproduced by plant engineers and reviewers
alike. This module centralizes seeding of every source of randomness used
across the pipeline (numpy, torch CPU/CUDA, and python's stdlib `random`)
and should be called at the very top of every experiment script/notebook,
before any data loading, model construction, or training.
"""

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Seed all random number generators used across the AMMPM pipeline.

    Fixes the seed for python's `random`, `numpy`, and `torch` (CPU and, if
    available, all CUDA devices), and configures torch/cuDNN for deterministic
    execution. This should be the first call in every experiment entry point
    (scripts under experiments/, notebooks under notebooks/) so that data
    splits, model initialization, dropout, and any stochastic augmentation
    are reproducible across machines and re-runs.

    Args:
        seed: Integer seed value to apply globally. Defaults to 42, the
            project-wide standard seed defined in
            configs/experiment_config.yaml.

    Returns:
        None. Has the side effect of mutating global RNG state.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Favor deterministic kernels over raw throughput: for predictive
        # maintenance research, bit-reproducible RUL predictions matter more
        # than shaving milliseconds off training on a single rig.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # torch may be absent in lightweight data-only environments (e.g. a
        # CI job that only exercises src/data). Seeding numpy/random still
        # covers those cases.
        pass


# Backwards/forwards-compatible alias so both call styles read naturally in
# experiment scripts: `set_global_seed(42)` or `set_seed(42)`.
set_seed = set_global_seed
