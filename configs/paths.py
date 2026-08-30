"""Centralized filesystem paths for the AMMPM project.

Every module that reads raw sensor logs, writes processed feature tables,
saves model checkpoints, or dumps evaluation results should import its
location from here rather than hard-coding relative paths. This keeps the
pipeline portable across machines (a researcher's laptop, a lab workstation,
a shared HPC node) and avoids the classic "works only from my cwd" failure
mode when running experiment scripts, notebooks, or agents from different
working directories.

All paths are resolved relative to PROJECT_ROOT, which is derived from this
file's own location, not from `os.getcwd()`.
"""

from pathlib import Path

# Root of the repository: configs/paths.py -> project_root/configs/paths.py
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# --- Configuration ---
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
EXPERIMENT_CONFIG_PATH: Path = CONFIGS_DIR / "experiment_config.yaml"

# --- Data ---
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

# Dataset-specific raw data locations (industrial asset degradation sources).
CMAPSS_RAW_DIR: Path = RAW_DATA_DIR / "cmapss"
XJTU_SY_RAW_DIR: Path = RAW_DATA_DIR / "xjtu_sy"
AI4I_2020_RAW_DIR: Path = RAW_DATA_DIR / "ai4i_2020"

# --- Source ---
SRC_DIR: Path = PROJECT_ROOT / "src"

# --- Experiments & notebooks ---
EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# --- Results & reporting ---
RESULTS_DIR: Path = PROJECT_ROOT / "results"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

# --- Model checkpoints ---
# Not present in the initial scaffold; created on demand by training scripts
# via ensure_dirs() below. Kept under results/ so a single artifact bundle
# (checkpoints + metrics + plots) can be archived per experiment run.
CHECKPOINTS_DIR: Path = RESULTS_DIR / "checkpoints"

# E2 Transformer (state_dict) and E7 Quantile Regression (state_dict) checkpoints,
# trained once by notebooks/E8_langgraph_agentic_system.ipynb's bootstrap step and
# loaded by src/agents/prediction_agent.py and src/agents/uncertainty_agent.py for
# fast per-engine inference (no retraining inside the agent pipeline).
E2_TRANSFORMER_CHECKPOINT: Path = CHECKPOINTS_DIR / "e2_transformer.pt"
E7_QUANTILE_CHECKPOINT: Path = CHECKPOINTS_DIR / "e7_quantile_regressor.pt"

# Reference artifacts for the agentic pipeline (src/agents/): a sample of training
# windows for domain-shift (MMD) comparison, and a cached SHAP background sample.
AGENTS_REFERENCE_DIR: Path = RESULTS_DIR / "agents_reference"
TRAIN_DISTRIBUTION_REFERENCE_PATH: Path = AGENTS_REFERENCE_DIR / "train_distribution_reference.npz"
SHAP_BACKGROUND_PATH: Path = AGENTS_REFERENCE_DIR / "shap_background.npz"
CHANNEL_REFERENCE_STATS_PATH: Path = AGENTS_REFERENCE_DIR / "channel_reference_stats.npz"


def ensure_dirs() -> None:
    """Create all project directories that are expected to exist at runtime.

    Safe to call at the start of any experiment/training script: creates
    missing directories (e.g. dataset-specific raw folders, the checkpoints
    directory) without raising if they already exist, so first-time setup on
    a fresh machine "just works" before data download or model training
    begins.

    Returns:
        None. Has the side effect of creating directories on disk.
    """
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        CMAPSS_RAW_DIR,
        XJTU_SY_RAW_DIR,
        AI4I_2020_RAW_DIR,
        RESULTS_DIR,
        REPORTS_DIR,
        CHECKPOINTS_DIR,
        AGENTS_REFERENCE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
