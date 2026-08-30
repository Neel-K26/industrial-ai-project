"""Domain Shift Agent: is this engine's sensor behavior still "FD001-like"?

A model trained on FD001 (single operating condition, one fault mode) has no
guarantee its predictions stay meaningful if fed data from a differently
-behaving population — a different machine, a different operating regime, a
sensor recalibration. This agent quantifies that with Maximum Mean
Discrepancy (MMD): a kernel two-sample statistic comparing the current
window's per-channel profile against a cached sample of FD001 training
windows (`configs.paths.TRAIN_DISTRIBUTION_REFERENCE_PATH`).

Score construction (documented here because it's a real design choice, not
an obvious formula): each window is collapsed to a single point (its
per-channel mean, 24-dim). MMD-squared between "this one point" and the
reference sample is a well-defined two-sample statistic, but its raw
magnitude has no intrinsic scale. So the reference set's own leave-one-out
MMD-squared distribution (each reference point vs. the rest of the reference
set) is precomputed as a baseline of "how atypical does a genuinely
in-distribution point look" — and `domain_shift_score` is the current
window's percentile rank against that baseline: 0 means perfectly typical,
close to 1 means more extreme than nearly every reference point. This is
naturally bounded in [0, 1] and needs no arbitrary scaling constant.
"""

from typing import Tuple

import numpy as np

from configs.paths import TRAIN_DISTRIBUTION_REFERENCE_PATH
from src.agents.state import AgentState

SHIFT_THRESHOLD = 0.95  # domain_shift_score above this -> shift_detected


def _load_reference() -> Tuple[np.ndarray, np.ndarray, float]:
    if not TRAIN_DISTRIBUTION_REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"{TRAIN_DISTRIBUTION_REFERENCE_PATH} not found — run the bootstrap section of "
            "notebooks/E8_langgraph_agentic_system.ipynb first to build the reference cache."
        )
    data = np.load(TRAIN_DISTRIBUTION_REFERENCE_PATH)
    return data["reference_points"], data["baseline_mmd_sq"], float(data["bandwidth"])


def _rbf_kernel(a: np.ndarray, b: np.ndarray, bandwidth: float) -> np.ndarray:
    """Gaussian/RBF kernel matrix between rows of a (n, d) and b (m, d)."""
    sq_dists = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-sq_dists / (2 * bandwidth**2))


def _mmd_squared_point_vs_sample(point: np.ndarray, reference: np.ndarray, bandwidth: float) -> float:
    """MMD^2 between a single point and a reference sample (biased estimator).

    With one sample degenerate to a single point, the "within-X" term is k(x,x)=1
    and the standard biased MMD^2 formula reduces to:
        1 + mean(k(ref, ref)) - 2 * mean(k(point, ref))
    """
    point = point.reshape(1, -1)
    k_point_ref = _rbf_kernel(point, reference, bandwidth).mean()
    k_ref_ref = _rbf_kernel(reference, reference, bandwidth).mean()
    return float(1.0 + k_ref_ref - 2 * k_point_ref)


def compute_domain_shift(sensor_data: np.ndarray) -> Tuple[float, bool]:
    """Returns (domain_shift_score in [0,1], shift_detected)."""
    reference_points, baseline_mmd_sq, bandwidth = _load_reference()

    window_mean_vector = sensor_data.mean(axis=0)  # (n_features,) -- collapse the window to one point
    mmd_sq = _mmd_squared_point_vs_sample(window_mean_vector, reference_points, bandwidth)

    domain_shift_score = float((baseline_mmd_sq < mmd_sq).mean())  # percentile rank vs. the in-distribution baseline
    shift_detected = domain_shift_score > SHIFT_THRESHOLD
    return domain_shift_score, shift_detected


def run(state: AgentState) -> AgentState:
    """LangGraph node entry point: computes domain shift for state["sensor_data"]."""
    errors = list(state.get("errors", []))
    try:
        score, detected = compute_domain_shift(state["sensor_data"])
    except Exception as exc:  # noqa: BLE001
        errors.append(f"domain_shift_agent: {exc}")
        score, detected = 1.0, True  # fail safe: treat an agent error itself as a shift signal
    return {"domain_shift_score": score, "shift_detected": detected, "errors": errors}
