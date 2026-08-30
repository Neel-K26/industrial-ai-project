"""Sensor Validation Agent: the first line of defense in the AMMPM pipeline.

Every downstream agent (domain-shift, prediction, SHAP, uncertainty) assumes
its input window is at least *plausible* sensor data. This agent checks that
assumption before anything expensive runs: missing values, signal drift
relative to the training distribution, flat/dead channels, and out-of-range
readings. A window that fails critically (missing data, or most channels
failing) is routed by the orchestrator straight to the decision agent with a
forced-low trust score, rather than fed to a prediction model whose output
would be meaningless on invalid input.

Reference statistics (per-channel training mean/std/min/max) are precomputed
once and cached at `configs.paths.CHANNEL_REFERENCE_STATS_PATH` — see
`notebooks/E8_langgraph_agentic_system.ipynb`'s bootstrap section for how
that cache is built.
"""

from typing import Any, Dict

import numpy as np

from configs.paths import CHANNEL_REFERENCE_STATS_PATH
from src.agents.state import AgentState

# A window failing on more than this fraction of channels, or containing any
# missing value, is treated as a critical failure (see `validate_sensor_window`).
CRITICAL_FAILURE_FRACTION = 0.5
DRIFT_ZSCORE_THRESHOLD = 3.0
FLAT_SIGNAL_STD_THRESHOLD = 1e-3


def _load_reference_stats():
    if not CHANNEL_REFERENCE_STATS_PATH.exists():
        raise FileNotFoundError(
            f"{CHANNEL_REFERENCE_STATS_PATH} not found — run the bootstrap section of "
            "notebooks/E8_langgraph_agentic_system.ipynb first to build the reference cache."
        )
    data = np.load(CHANNEL_REFERENCE_STATS_PATH, allow_pickle=True)
    return data["mean"], data["std"], data["min"], data["max"], list(data["feature_columns"])


def validate_sensor_window(sensor_data: np.ndarray) -> Dict[str, Any]:
    """Run all four checks per channel on one (window_length, n_features) sensor window.

    Returns a dict: {"passed": bool, "critical_failure": bool, "n_channels_failed": int,
    "failed_channels": [...], "channels": {channel_name: {check: bool, ..., "pass": bool}}}.
    """
    train_mean, train_std, train_min, train_max, feature_columns = _load_reference_stats()

    channel_results = {}
    failed_channels = []

    for i, name in enumerate(feature_columns):
        column = sensor_data[:, i]

        missing = bool(np.isnan(column).any())
        window_std = float(np.nanstd(column))
        flat = window_std < FLAT_SIGNAL_STD_THRESHOLD

        # z-score drift: how far this window's mean reading sits from the training
        # distribution, in training standard deviations (guarded against a zero
        # reference std, which would otherwise make a truly constant channel's
        # z-score undefined rather than "no drift by construction").
        ref_std = train_std[i] if train_std[i] > 1e-9 else 1.0
        window_mean = float(np.nanmean(column)) if not missing else float("nan")
        zscore = abs(window_mean - train_mean[i]) / ref_std if not missing else float("inf")
        drift = zscore > DRIFT_ZSCORE_THRESHOLD

        range_violation = bool(
            not missing and ((column < train_min[i] - ref_std).any() or (column > train_max[i] + ref_std).any())
        )

        channel_pass = not (missing or drift or range_violation)  # a flat channel alone is not disqualifying
        # (many FD001 channels are legitimately constant, per E5/E6 — see CONSTANT_SENSORS_FD001
        # elsewhere in this study; only missing/drift/range indicate an actual data problem)

        channel_results[name] = {
            "missing": missing,
            "drift": drift,
            "flat": flat,
            "range_violation": range_violation,
            "zscore": None if missing else round(zscore, 3),
            "pass": channel_pass,
        }
        if not channel_pass:
            failed_channels.append(name)

    n_failed = len(failed_channels)
    any_missing = any(r["missing"] for r in channel_results.values())
    critical_failure = any_missing or (n_failed / len(feature_columns)) > CRITICAL_FAILURE_FRACTION

    return {
        "passed": n_failed == 0,
        "critical_failure": critical_failure,
        "n_channels_failed": n_failed,
        "failed_channels": failed_channels,
        "channels": channel_results,
    }


def run(state: AgentState) -> AgentState:
    """LangGraph node entry point: validates state["sensor_data"], returns the update."""
    errors = list(state.get("errors", []))
    try:
        validation_result = validate_sensor_window(state["sensor_data"])
    except Exception as exc:  # noqa: BLE001 - a validation crash is itself a critical finding, not a bug to hide
        errors.append(f"sensor_validation_agent: {exc}")
        validation_result = {
            "passed": False,
            "critical_failure": True,
            "n_channels_failed": None,
            "failed_channels": [],
            "channels": {},
        }
    return {"validation_result": validation_result, "errors": errors}
