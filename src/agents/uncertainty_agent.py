"""Uncertainty Agent: 80% prediction interval, using E7's winning method (Quantile Regression).

E7 compared three UQ methods head-to-head; Quantile Regression won on
calibration error (0.08, versus MC Dropout's 0.58 and the Deep Ensemble's
0.57 — both severely overconfident, see `results/E7_uq_fd001.json`). This
agent loads the checkpoint trained once by the E8 bootstrap section
(`configs.paths.E7_QUANTILE_CHECKPOINT`) and reads off [q10, q90] directly
from the model's three-quantile head — no post-hoc interval construction
needed, unlike the other two methods E7 evaluated.

`coverage_flag` is only meaningful when ground truth is available (i.e. in a
backtest/evaluation context, such as E8's own demonstration run over labeled
test engines) — it is `None` in a genuine live-deployment call where the true
RUL is, by definition, not yet known.
"""

from typing import List, Optional, Tuple

import torch

from configs.paths import E7_QUANTILE_CHECKPOINT
from src.agents.state import AgentState
from src.models.transformer import QuantileTransformerRULRegressor

QUANTILES = [0.1, 0.5, 0.9]
_MODEL_CACHE = {}


def load_quantile_model(input_size: int) -> QuantileTransformerRULRegressor:
    if input_size in _MODEL_CACHE:
        return _MODEL_CACHE[input_size]

    if not E7_QUANTILE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"{E7_QUANTILE_CHECKPOINT} not found — run the bootstrap section of "
            "notebooks/E8_langgraph_agentic_system.ipynb first to train and save it."
        )

    model = QuantileTransformerRULRegressor(input_size=input_size, n_quantiles=len(QUANTILES))
    model.load_state_dict(torch.load(E7_QUANTILE_CHECKPOINT, map_location="cpu"))
    model.eval()
    _MODEL_CACHE[input_size] = model
    return model


def predict_interval(sensor_data) -> Tuple[List[float], float]:
    """sensor_data: (window_length, n_features) -> ([lower, upper], interval_width)."""
    model = load_quantile_model(input_size=sensor_data.shape[-1])
    with torch.no_grad():
        x = torch.from_numpy(sensor_data).unsqueeze(0).to(torch.float32)
        q10, _q50, q90 = model(x).squeeze(0).tolist()
    lo, hi = min(q10, q90), max(q10, q90)  # guard the known quantile-crossing failure mode (see E7)
    return [lo, hi], hi - lo


def run(state: AgentState) -> AgentState:
    """LangGraph node entry point: computes the 80% prediction interval for state["sensor_data"]."""
    errors = list(state.get("errors", []))
    try:
        interval, width = predict_interval(state["sensor_data"])
    except Exception as exc:  # noqa: BLE001
        errors.append(f"uncertainty_agent: {exc}")
        interval, width = [None, None], None

    true_rul = state.get("true_rul")
    coverage_flag: Optional[bool] = None
    if true_rul is not None and interval[0] is not None:
        coverage_flag = interval[0] <= true_rul <= interval[1]

    return {
        "prediction_interval": interval,
        "interval_width": width,
        "coverage_flag": coverage_flag,
        "errors": errors,
    }
