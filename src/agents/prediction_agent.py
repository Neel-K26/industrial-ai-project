"""Prediction Agent: RUL point estimate from the best-performing model (E2 Transformer).

Loads the checkpoint trained once by `notebooks/E8_langgraph_agentic_system.ipynb`'s
bootstrap section (`configs.paths.E2_TRANSFORMER_CHECKPOINT`) and runs a single
forward pass on the validated sensor window — no training happens here, which
is what keeps this agent well under the pipeline's 30-second-per-engine budget.
"""

import torch

from configs.paths import E2_TRANSFORMER_CHECKPOINT
from src.agents.state import AgentState
from src.models.transformer import TransformerRULRegressor

_MODEL_CACHE = {}


def load_model(input_size: int) -> TransformerRULRegressor:
    """Load (and cache in-process) the trained E2 Transformer checkpoint."""
    if input_size in _MODEL_CACHE:
        return _MODEL_CACHE[input_size]

    if not E2_TRANSFORMER_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"{E2_TRANSFORMER_CHECKPOINT} not found — run the bootstrap section of "
            "notebooks/E8_langgraph_agentic_system.ipynb first to train and save it."
        )

    model = TransformerRULRegressor(input_size=input_size)
    model.load_state_dict(torch.load(E2_TRANSFORMER_CHECKPOINT, map_location="cpu"))
    model.eval()
    _MODEL_CACHE[input_size] = model
    return model


def predict_rul(sensor_data) -> float:
    """sensor_data: (window_length, n_features) -> scalar RUL point estimate."""
    model = load_model(input_size=sensor_data.shape[-1])
    with torch.no_grad():
        x = torch.from_numpy(sensor_data).unsqueeze(0).to(torch.float32)  # (1, window, n_features)
        pred = model(x).item()
    return float(pred)


def run(state: AgentState) -> AgentState:
    """LangGraph node entry point: predicts RUL for state["sensor_data"]."""
    errors = list(state.get("errors", []))
    try:
        rul_prediction = predict_rul(state["sensor_data"])
    except Exception as exc:  # noqa: BLE001
        errors.append(f"prediction_agent: {exc}")
        rul_prediction = None
    return {"rul_prediction": rul_prediction, "errors": errors}
