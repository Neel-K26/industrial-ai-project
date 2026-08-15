"""NASA C-MAPSS turbofan engine degradation dataset loader.

C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) is one of the
canonical benchmarks for data-driven Remaining Useful Life (RUL) prediction
in industrial predictive maintenance: run-to-failure sensor trajectories from
simulated turbofan engines, split into four operating/fault-mode subsets
(FD001-FD004) of increasing difficulty (more operating conditions, more
fault modes). This module handles acquisition (download or manual fallback),
loading, standard preprocessing (min-max normalization + RUL capping), and
train/test array construction for downstream sequence models (LSTM/GRU/TCN/
Transformer/PatchTST/Chronos) and SHAP-based explainability analysis.

Reference:
    A. Saxena, K. Goebel (2008). "Turbofan Engine Degradation Simulation
    Data Set", NASA Ames Prognostics Data Repository.
"""

import io
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs.paths import CMAPSS_RAW_DIR  # noqa: E402

# Official NASA PHM/PCoE data repository mirror (Amazon S3) for the
# "Turbofan Engine Degradation Simulation Data Set" (C-MAPSS). NASA
# occasionally reorganizes the Prognostics Center of Excellence (PCoE) data
# repository, so this link can go stale — if download_cmapss() fails, follow
# the manual instructions it prints (or pass a fresher `url=`).
#
# Note: this archive is nested — it wraps "CMAPSSData.zip" (which contains
# the actual train/test/RUL .txt files) inside an outer zip alongside a PDF
# and readme. See _extract_cmapss_archive().
CMAPSS_DOWNLOAD_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)

# Official landing page, kept for the manual-download fallback message.
CMAPSS_SOURCE_PAGE = (
    "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/"
    "pcoe/pcoe-data-set-repository/"
)

# All four C-MAPSS operating/fault-mode subsets.
FD_SUBSETS: Tuple[int, ...] = (1, 2, 3, 4)

# C-MAPSS raw columns: unit id, cycle index, 3 operational settings,
# 21 sensor measurements (26 columns total, whitespace-delimited, no header).
INDEX_COLUMNS: List[str] = ["unit_number", "time_in_cycles"]
OP_SETTING_COLUMNS: List[str] = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS: List[str] = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLUMNS: List[str] = INDEX_COLUMNS + OP_SETTING_COLUMNS + SENSOR_COLUMNS
FEATURE_COLUMNS: List[str] = OP_SETTING_COLUMNS + SENSOR_COLUMNS


def _manual_instructions(raw_dir: Path) -> str:
    """Build a human-readable manual-download fallback message.

    Used whenever automatic acquisition of C-MAPSS fails (network
    restriction, expired mirror URL, no internet access on an air-gapped
    plant workstation, etc.) so the researcher can retrieve the dataset by
    hand without digging through source code.

    Args:
        raw_dir: Directory where the extracted train/test/RUL .txt files are
            expected to live.

    Returns:
        A formatted multi-line instruction string.
    """
    return (
        "\nCould not automatically download NASA C-MAPSS.\n"
        "Manual download instructions:\n"
        f"  1. Visit the NASA PCoE Data Set Repository:\n     {CMAPSS_SOURCE_PAGE}\n"
        "  2. Locate '6. Turbofan Engine Degradation Simulation Data Set' and "
        "download 'CMAPSSData.zip'.\n"
        f"  3. Extract its contents into: {raw_dir}\n"
        "  4. Confirm the following files are present (per subset FD001-FD004):\n"
        "       train_FD00X.txt, test_FD00X.txt, RUL_FD00X.txt\n"
        "  5. Re-run this loader — it will detect the local files and skip "
        "download.\n"
    )


def _is_cmapss_data_file(member_name: str) -> bool:
    """Whether a zip member is one of the train/test/RUL .txt data files."""
    name = Path(member_name).name.lower()
    return name.endswith(".txt") and (
        name.startswith("train_") or name.startswith("test_") or name.startswith("rul_")
    )


def _extract_cmapss_archive(zip_path: Path, raw_dir: Path) -> None:
    """Flatten the train/test/RUL .txt files out of a CMAPSS zip into raw_dir.

    The official NASA PHM mirror wraps the real `CMAPSSData.zip` inside an
    outer archive alongside a PDF and readme, one directory level down (and
    the manually-downloaded NASA PCoE archive nests the same way). This
    walks the top-level archive, descending into any nested .zip member it
    finds, and copies every matching .txt file straight into `raw_dir`
    (discarding directory structure) so callers don't need to know how many
    levels of nesting a given mirror uses.

    Args:
        zip_path: Path to the downloaded (outer) zip archive.
        raw_dir: Destination directory for the flattened .txt files.
    """
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.namelist():
            if _is_cmapss_data_file(member):
                with archive.open(member) as src, open(
                    raw_dir / Path(member).name, "wb"
                ) as dst:
                    shutil.copyfileobj(src, dst)
            elif member.lower().endswith(".zip"):
                with archive.open(member) as inner_file:
                    inner_bytes = inner_file.read()
                with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_archive:
                    for inner_member in inner_archive.namelist():
                        if _is_cmapss_data_file(inner_member):
                            with inner_archive.open(inner_member) as src, open(
                                raw_dir / Path(inner_member).name, "wb"
                            ) as dst:
                                shutil.copyfileobj(src, dst)


def download_cmapss(
    dest_dir: Optional[Path] = None, url: str = CMAPSS_DOWNLOAD_URL
) -> bool:
    """Download and extract the NASA C-MAPSS dataset archive.

    Fetches the archive from a public URL and flattens its train/test/RUL
    .txt files into `dest_dir` (see _extract_cmapss_archive() for why this
    isn't a plain extractall — the official mirror nests CMAPSSData.zip
    inside an outer zip). If the download fails for any reason (stale
    mirror, offline environment, permissions), prints clear manual-download
    instructions instead of raising, so pipeline setup can fail gracefully
    on a locked-down industrial workstation.

    Args:
        dest_dir: Destination directory for the extracted .txt files.
            Defaults to configs/paths.py::CMAPSS_RAW_DIR.
        url: Direct URL to the CMAPSS archive. Overridable in case the
            default NASA PHM/PCoE mirror has moved.

    Returns:
        True if the archive was downloaded and extracted successfully,
        False otherwise (manual instructions are printed in that case).
    """
    raw_dir = Path(dest_dir) if dest_dir is not None else CMAPSS_RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "_cmapss_download.zip"
    try:
        print(f"[cmapss_loader] Downloading C-MAPSS archive from {url} ...")
        with urllib.request.urlopen(url, timeout=60) as response, open(
            zip_path, "wb"
        ) as out_file:
            shutil.copyfileobj(response, out_file)

        _extract_cmapss_archive(zip_path, raw_dir)
        print(f"[cmapss_loader] Extracted C-MAPSS data files into {raw_dir}")
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            zipfile.BadZipFile) as exc:
        print(f"[cmapss_loader] Automatic download failed: {exc}")
        print(_manual_instructions(raw_dir))
        return False
    finally:
        zip_path.unlink(missing_ok=True)


def _require_local_files(raw_dir: Path, fd_num: int) -> None:
    """Verify that the raw train/test/RUL files for a subset exist locally.

    Args:
        raw_dir: Directory expected to contain the C-MAPSS .txt files.
        fd_num: Subset number (1-4).

    Raises:
        FileNotFoundError: If any of the three required files is missing,
            with the manual-download instructions included in the message.
    """
    required = [
        raw_dir / f"train_FD{fd_num:03d}.txt",
        raw_dir / f"test_FD{fd_num:03d}.txt",
        raw_dir / f"RUL_FD{fd_num:03d}.txt",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing C-MAPSS FD{fd_num:03d} files: {missing}\n"
            + _manual_instructions(raw_dir)
        )


def _read_subset_files(
    raw_dir: Path, fd_num: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the raw whitespace-delimited train/test/RUL files for one subset.

    Args:
        raw_dir: Directory containing the extracted C-MAPSS .txt files.
        fd_num: Subset number (1-4), selecting FD001/FD002/FD003/FD004.

    Returns:
        A tuple (train_df, test_df, rul_df):
            train_df / test_df: raw run-to-failure sensor trajectories with
                columns ALL_COLUMNS.
            rul_df: single-column DataFrame of ground-truth RUL at the final
                recorded cycle of each test engine (one row per unit).
    """
    _require_local_files(raw_dir, fd_num)

    train_df = pd.read_csv(
        raw_dir / f"train_FD{fd_num:03d}.txt",
        sep=r"\s+",
        header=None,
        names=ALL_COLUMNS,
    )
    test_df = pd.read_csv(
        raw_dir / f"test_FD{fd_num:03d}.txt",
        sep=r"\s+",
        header=None,
        names=ALL_COLUMNS,
    )
    rul_df = pd.read_csv(
        raw_dir / f"RUL_FD{fd_num:03d}.txt", sep=r"\s+", header=None, names=["RUL"]
    )
    return train_df, test_df, rul_df


def _add_train_rul(df: pd.DataFrame, max_rul: int) -> pd.DataFrame:
    """Compute and cap the Remaining Useful Life target for training data.

    Each training engine runs to failure, so RUL at cycle t is simply
    (final observed cycle - t). RUL is then capped at `max_rul`: near the
    start of an engine's life, degradation has not yet begun, so treating
    RUL as unbounded encourages the model to chase an unlearnable early
    trend. Capping (standard practice for C-MAPSS, per Heimes 2008 / Saxena
    2008 and most subsequent literature) reflects that "healthy" RUL is
    effectively saturated from an operator's point of view.

    Args:
        df: Raw training DataFrame with columns ALL_COLUMNS.
        max_rul: Ceiling applied to the computed RUL (piecewise-linear
            degradation assumption).

    Returns:
        A copy of `df` with an added integer "RUL" column.
    """
    df = df.copy()
    max_cycle = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df["RUL"] = (max_cycle - df["time_in_cycles"]).clip(upper=max_rul)
    return df


def _add_test_rul(
    df: pd.DataFrame, rul_df: pd.DataFrame, max_rul: int
) -> pd.DataFrame:
    """Compute and cap the RUL target for test data at every observed cycle.

    Test trajectories are truncated before failure; `rul_df` gives the
    ground-truth RUL at the *last* recorded cycle of each unit. RUL at any
    earlier cycle t is therefore
        (last_recorded_cycle - t) + RUL_at_last_recorded_cycle,
    again capped at `max_rul` for consistency with the training target.

    Args:
        df: Raw test DataFrame with columns ALL_COLUMNS.
        rul_df: Ground-truth final-cycle RUL, one row per unit, in unit order.
        max_rul: Ceiling applied to the computed RUL.

    Returns:
        A copy of `df` with an added integer "RUL" column.
    """
    df = df.copy()
    final_rul = rul_df["RUL"].to_numpy()
    unit_ids = sorted(df["unit_number"].unique())
    final_rul_by_unit = dict(zip(unit_ids, final_rul))

    max_cycle = df.groupby("unit_number")["time_in_cycles"].transform("max")
    tail_rul = df["unit_number"].map(final_rul_by_unit)
    df["RUL"] = ((max_cycle - df["time_in_cycles"]) + tail_rul).clip(upper=max_rul)
    return df


def _min_max_normalize(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply min-max normalization to sensor/operational-setting features.

    Scalers are fit exclusively on the training split and applied to both
    splits, avoiding test-set leakage into the normalization statistics —
    standard practice for any RUL benchmark comparison. Constant-valued
    columns (zero training range, common for a handful of C-MAPSS sensors
    in the single-operating-condition subsets FD001/FD003) are mapped to a
    constant 0.0 rather than producing NaNs from division by zero.

    Args:
        train_df: Training DataFrame containing `feature_cols`.
        test_df: Test DataFrame containing `feature_cols`.
        feature_cols: Names of the operational-setting/sensor columns to
            normalize in place.

    Returns:
        A tuple (train_df, test_df) with `feature_cols` rescaled to [0, 1]
        based on training-set min/max.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    col_min = train_df[feature_cols].min()
    col_max = train_df[feature_cols].max()
    col_range = (col_max - col_min).replace(0, 1)  # guard constant sensors

    train_df[feature_cols] = (train_df[feature_cols] - col_min) / col_range
    test_df[feature_cols] = (test_df[feature_cols] - col_min) / col_range

    # Constant training-set columns: pin the normalized value to 0.0 instead
    # of an arbitrary offset, since they carry no discriminative signal.
    constant_cols = col_max[col_max == col_min].index.tolist()
    if constant_cols:
        train_df[constant_cols] = 0.0
        test_df[constant_cols] = 0.0

    return train_df, test_df


def get_cmapss(
    fd_num: int = 1,
    max_rul: int = 125,
    raw_dir: Optional[Path] = None,
    normalize: bool = True,
    auto_download: bool = True,
) -> Dict[str, object]:
    """Load, preprocess, and split one C-MAPSS subset for RUL prediction.

    Applies the standard preprocessing pipeline used across the AMMPM
    benchmark suite: min-max normalization of operational settings/sensor
    channels (fit on train only) and RUL capping at `max_rul` cycles, then
    returns both DataFrame and numpy-array views so downstream code can pick
    whichever is convenient (DataFrames for EDA/agents, arrays for tensors).

    Args:
        fd_num: Which C-MAPSS subset to load: 1, 2, 3, or 4 (FD001-FD004).
        max_rul: RUL capping threshold in cycles. Defaults to 125, the
            de facto standard in the C-MAPSS RUL prediction literature.
        raw_dir: Directory containing (or to receive) the raw .txt files.
            Defaults to configs/paths.py::CMAPSS_RAW_DIR.
        normalize: If True, min-max normalize operational-setting/sensor
            columns using training-set statistics.
        auto_download: If True and the local files are missing, attempt to
            download the dataset automatically before falling back to
            manual instructions.

    Returns:
        A dictionary with keys:
            "train_df": preprocessed training DataFrame (with "RUL" column).
            "test_df": preprocessed test DataFrame (with "RUL" column).
            "X_train": numpy array (n_train_rows, n_features) of feature
                columns (FEATURE_COLUMNS order).
            "y_train": numpy array (n_train_rows,) of capped RUL targets.
            "X_test": numpy array (n_test_rows, n_features) of feature
                columns.
            "y_test": numpy array (n_test_rows,) of capped RUL targets.
            "feature_columns": list of feature column names, in array order.
            "fd_num": the requested subset number, for bookkeeping.

    Raises:
        FileNotFoundError: If the raw files are unavailable locally and
            either `auto_download` is False or the download attempt fails.
    """
    if fd_num not in FD_SUBSETS:
        raise ValueError(f"fd_num must be one of {FD_SUBSETS}, got {fd_num}")

    raw_dir = Path(raw_dir) if raw_dir is not None else CMAPSS_RAW_DIR

    required = [
        raw_dir / f"train_FD{fd_num:03d}.txt",
        raw_dir / f"test_FD{fd_num:03d}.txt",
        raw_dir / f"RUL_FD{fd_num:03d}.txt",
    ]
    if auto_download and not all(p.exists() for p in required):
        download_cmapss(dest_dir=raw_dir)

    train_raw, test_raw, rul_raw = _read_subset_files(raw_dir, fd_num)

    train_df = _add_train_rul(train_raw, max_rul=max_rul)
    test_df = _add_test_rul(test_raw, rul_raw, max_rul=max_rul)

    if normalize:
        train_df, test_df = _min_max_normalize(train_df, test_df, FEATURE_COLUMNS)

    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = train_df["RUL"].to_numpy(dtype=np.float32)
    X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_test = test_df["RUL"].to_numpy(dtype=np.float32)

    return {
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_columns": FEATURE_COLUMNS,
        "fd_num": fd_num,
    }


def get_all_cmapss_subsets(
    max_rul: int = 125,
    raw_dir: Optional[Path] = None,
    normalize: bool = True,
    auto_download: bool = True,
) -> Dict[int, Dict[str, object]]:
    """Load all four C-MAPSS subsets (FD001-FD004) for multi-machine study.

    Convenience wrapper around `get_cmapss` for the AMMPM setting, where
    generalization *across* engine fleets/operating regimes (not just within
    a single subset) is the object of study.

    Args:
        max_rul: RUL capping threshold in cycles, applied uniformly to all
            four subsets.
        raw_dir: Directory containing (or to receive) the raw .txt files.
            Defaults to configs/paths.py::CMAPSS_RAW_DIR.
        normalize: If True, min-max normalize features within each subset
            independently (each FD subset is treated as its own machine
            population with its own operating-condition statistics).
        auto_download: If True, attempt automatic download for any missing
            subset before falling back to manual instructions.

    Returns:
        A dictionary mapping fd_num (1-4) to the corresponding `get_cmapss`
        result dictionary.
    """
    return {
        fd_num: get_cmapss(
            fd_num=fd_num,
            max_rul=max_rul,
            raw_dir=raw_dir,
            normalize=normalize,
            auto_download=auto_download,
        )
        for fd_num in FD_SUBSETS
    }


def _local_subset_status(raw_dir: Path) -> Dict[int, bool]:
    """Check which FD001-FD004 subsets have all three files present locally.

    Args:
        raw_dir: Directory expected to contain the C-MAPSS .txt files.

    Returns:
        A dict mapping fd_num (1-4) to True if train/test/RUL are all
        present for that subset, False otherwise.
    """
    status = {}
    for fd_num in FD_SUBSETS:
        required = [
            raw_dir / f"train_FD{fd_num:03d}.txt",
            raw_dir / f"test_FD{fd_num:03d}.txt",
            raw_dir / f"RUL_FD{fd_num:03d}.txt",
        ]
        status[fd_num] = all(p.exists() for p in required)
    return status


if __name__ == "__main__":
    # Standalone acquisition entry point: `python src/data/cmapss_loader.py`
    # downloads (if needed) and confirms all four C-MAPSS subsets, without
    # running the normalization/RUL-labeling pipeline in get_cmapss().
    raw_dir = CMAPSS_RAW_DIR
    status = _local_subset_status(raw_dir)

    if not all(status.values()):
        download_cmapss(dest_dir=raw_dir)
        status = _local_subset_status(raw_dir)

    print(f"\nC-MAPSS raw data directory: {raw_dir}")
    for fd_num, ok in status.items():
        print(f"  FD{fd_num:03d}: {'OK' if ok else 'MISSING'}")

    if all(status.values()):
        print("\nAll FD001-FD004 files present.")
    else:
        print("\nSome C-MAPSS files are still missing.")
        sys.exit(1)
