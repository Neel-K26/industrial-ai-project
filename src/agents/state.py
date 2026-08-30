"""Shared state schema for the AMMPM agentic RUL-prediction pipeline (E8).

A single `AgentState` TypedDict is threaded through every node of the
LangGraph `StateGraph` built in `orchestrator.py`: each of the seven agents
reads whatever fields it needs and returns a partial update, which LangGraph
merges into the running state. Centralizing the schema here (rather than
letting each agent module define its own ad hoc dict shape) is what keeps the
graph's data contract auditable in one place — a hard requirement for a
system whose whole point is producing trustworthy, traceable maintenance
recommendations.

Note on `total=False`: fields are populated incrementally as the state moves
through the graph (e.g. `rul_prediction` doesn't exist yet when
`sensor_validation_agent` runs), so no field is required to be present from
the start.
"""

from typing import Any, Dict, List, Optional, TypedDict

import numpy as np


class AgentState(TypedDict, total=False):
    # --- Input (set before the graph is invoked) ---
    sensor_data: np.ndarray  # (window_length, n_features) raw sensor window for one engine
    engine_id: int
    machine_type: str  # e.g. "turbofan_FD001"; distinguishes machine population for domain-shift comparison
    true_rul: Optional[float]  # ground truth, present only in backtest/evaluation contexts (not a live deployment)

    # --- src/agents/sensor_validation_agent.py ---
    validation_result: Dict[str, Any]  # per-channel pass/fail + critical_failure flag (see that module's docstring)

    # --- src/agents/domain_shift_agent.py ---
    domain_shift_score: float  # in [0, 1]: percentile rank of this window's MMD against the training reference
    shift_detected: bool

    # --- src/agents/prediction_agent.py ---
    rul_prediction: float

    # --- src/agents/shap_agent.py ---
    shap_values: Any  # np.ndarray, (window_length, n_features) raw per-timestep SHAP attribution
    sensor_rankings: Dict[str, float]  # feature_name -> mean |SHAP|, descending
    top3_sensors: List[str]

    # --- src/agents/uncertainty_agent.py ---
    prediction_interval: List[float]  # [lower, upper], the 80% interval from the E7 quantile model
    interval_width: float
    coverage_flag: Optional[bool]  # true_rul within the interval; None if true_rul is unavailable

    # --- src/agents/decision_agent.py ---
    trust_score: float  # 0-100
    maintenance_recommendation: str

    # --- Cross-cutting ---
    errors: List[str]  # appended to (not overwritten) by any agent that hits a recoverable problem
