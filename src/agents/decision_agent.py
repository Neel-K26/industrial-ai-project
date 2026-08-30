"""Decision Agent: the pipeline's final step — a Trust Score and a maintenance recommendation.

Trust Score (0-100), starting at 100:
  -20  if domain shift was detected
  -10  per failed sensor-channel validation
  -15  if the 80% prediction interval is wider than 30 cycles (high uncertainty)
  -10  if the top-ranked SHAP sensor for this engine differs from the
       historically-established top sensor (E6's finding: Ps30/sensor_11,
       loaded from `results/E6_shap_fd001.json`)

Maintenance recommendation (checked in this order — the first matching rule wins):
  trust >= 80 and rul <= 30   -> "IMMEDIATE ACTION"
  trust >= 80 and rul <= 60   -> "SCHEDULE WITHIN WEEK"
  trust >= 60                 -> "MONITOR CLOSELY"
  otherwise                   -> "REQUIRES HUMAN REVIEW"

If sensor validation failed critically upstream, this agent is the only one
that runs (the orchestrator routes straight here) — trust is forced to a low
fixed value and the recommendation is "REQUIRES HUMAN REVIEW" outright, since
no prediction/interval/attribution exists to reason about.
"""

import json
from typing import Optional

from configs.paths import RESULTS_DIR
from src.agents.state import AgentState

CRITICAL_FAILURE_TRUST_SCORE = 10.0
INTERVAL_WIDTH_THRESHOLD = 30.0


def _load_historical_top_sensor() -> Optional[str]:
    e6_path = RESULTS_DIR / "E6_shap_fd001.json"
    if not e6_path.exists():
        return None
    with open(e6_path) as f:
        e6_results = json.load(f)
    top3 = e6_results.get("top3_sensors", [])
    return top3[0] if top3 else None


def compute_trust_score(
    shift_detected: bool,
    n_channels_failed: int,
    interval_width: Optional[float],
    top_sensor: Optional[str],
    historical_top_sensor: Optional[str],
) -> float:
    trust = 100.0
    if shift_detected:
        trust -= 20
    trust -= 10 * n_channels_failed
    if interval_width is not None and interval_width > INTERVAL_WIDTH_THRESHOLD:
        trust -= 15
    if historical_top_sensor is not None and top_sensor is not None and top_sensor != historical_top_sensor:
        trust -= 10
    return max(0.0, min(100.0, trust))


def recommend(trust_score: float, rul_prediction: Optional[float]) -> str:
    if rul_prediction is not None and trust_score >= 80 and rul_prediction <= 30:
        return "IMMEDIATE ACTION"
    if rul_prediction is not None and trust_score >= 80 and rul_prediction <= 60:
        return "SCHEDULE WITHIN WEEK"
    if trust_score >= 60:
        return "MONITOR CLOSELY"
    return "REQUIRES HUMAN REVIEW"


def run(state: AgentState) -> AgentState:
    """LangGraph node entry point: the pipeline's final trust score + recommendation."""
    errors = list(state.get("errors", []))
    validation_result = state.get("validation_result", {})

    if validation_result.get("critical_failure"):
        errors.append("decision_agent: routed here directly due to critical sensor validation failure")
        return {
            "trust_score": CRITICAL_FAILURE_TRUST_SCORE,
            "maintenance_recommendation": "REQUIRES HUMAN REVIEW",
            "errors": errors,
        }

    historical_top_sensor = _load_historical_top_sensor()
    top3_sensors = state.get("top3_sensors", [])
    current_top_sensor = top3_sensors[0] if top3_sensors else None

    trust_score = compute_trust_score(
        shift_detected=state.get("shift_detected", False),
        n_channels_failed=validation_result.get("n_channels_failed") or 0,
        interval_width=state.get("interval_width"),
        top_sensor=current_top_sensor,
        historical_top_sensor=historical_top_sensor,
    )
    maintenance_recommendation = recommend(trust_score, state.get("rul_prediction"))

    return {
        "trust_score": trust_score,
        "maintenance_recommendation": maintenance_recommendation,
        "errors": errors,
    }
