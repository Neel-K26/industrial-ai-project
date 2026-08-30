"""SHAP Agent: per-engine feature attribution, using E6's methodology.

Reuses E6's finding directly: `shap.GradientExplainer` (not `DeepExplainer`,
which fails on this architecture's `LayerNorm` layers — see E6's notebook for
the empirical justification) against the same trained E2 Transformer, using a
cached 50-window background sample (`configs.paths.SHAP_BACKGROUND_PATH`,
built once by the E8 bootstrap section, sampled from FD001 training windows
exactly as E6 did).

The explainer object itself is reconstructed on each call rather than
pickled/cached across invocations — GradientExplainer is a thin wrapper
around the (already-cached) model and a small background tensor, so
reconstruction is cheap, and it sidesteps the fragility of trying to
serialize a live explainer bound to a PyTorch model.
"""

from typing import Dict, List, Tuple

import numpy as np
import shap
import torch

from configs.paths import E2_TRANSFORMER_CHECKPOINT, SHAP_BACKGROUND_PATH
from src.agents.prediction_agent import load_model
from src.agents.state import AgentState

NSAMPLES = 100  # per-call SHAP sample count; smaller than E6's 200 to stay comfortably under the 30s/engine budget


def _load_background() -> torch.Tensor:
    if not SHAP_BACKGROUND_PATH.exists():
        raise FileNotFoundError(
            f"{SHAP_BACKGROUND_PATH} not found — run the bootstrap section of "
            "notebooks/E8_langgraph_agentic_system.ipynb first to build the cache."
        )
    data = np.load(SHAP_BACKGROUND_PATH)
    return torch.from_numpy(data["background"])


def explain(sensor_data: np.ndarray, feature_columns: List[str]) -> Tuple[np.ndarray, Dict[str, float], List[str]]:
    """Returns (shap_values (window, n_features), sensor_rankings dict, top3_sensors)."""
    model = load_model(input_size=sensor_data.shape[-1])
    background = _load_background()

    explainer = shap.GradientExplainer(model, background)
    window_tensor = torch.from_numpy(sensor_data).unsqueeze(0).to(torch.float32)  # (1, window, n_features)
    shap_values = explainer.shap_values(window_tensor, nsamples=NSAMPLES)[0]  # (window, n_features)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)  # (n_features,)
    sensor_rankings = dict(
        sorted(zip(feature_columns, mean_abs_shap.tolist()), key=lambda kv: kv[1], reverse=True)
    )
    top3_sensors = list(sensor_rankings.keys())[:3]
    return shap_values, sensor_rankings, top3_sensors


def run(state: AgentState) -> AgentState:
    """LangGraph node entry point: computes per-engine SHAP attribution."""
    errors = list(state.get("errors", []))
    try:
        from src.data.cmapss_loader import FEATURE_COLUMNS  # local import: avoids a module-load-time data dependency

        shap_values, sensor_rankings, top3_sensors = explain(state["sensor_data"], FEATURE_COLUMNS)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"shap_agent: {exc}")
        shap_values, sensor_rankings, top3_sensors = None, {}, []
    return {
        "shap_values": shap_values,
        "sensor_rankings": sensor_rankings,
        "top3_sensors": top3_sensors,
        "errors": errors,
    }
