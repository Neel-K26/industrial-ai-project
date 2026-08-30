"""Orchestrator: wires the seven AMMPM agents into a LangGraph StateGraph.

Flow: sensor_validation -> domain_shift -> prediction -> shap -> uncertainty -> decision -> END

Conditional edge: if sensor_validation reports a critical failure, the graph
routes directly from sensor_validation to decision, skipping domain_shift,
prediction, shap, and uncertainty entirely — there is no point running a
prediction model (or explaining or bounding a prediction that was never
made) on sensor data that failed validation outright.

CRITICAL RULE (per the AMMPM project spec): every node below is a plain
Python function operating on numpy/PyTorch objects. LangGraph here manages
state transitions between deterministic components only — nothing in this
graph makes an LLM call or does natural-language reasoning; "orchestration"
means control flow, not delegation to a language model.
"""

from langgraph.graph import END, StateGraph

from src.agents import decision_agent, domain_shift_agent, prediction_agent, sensor_validation_agent, shap_agent, uncertainty_agent
from src.agents.state import AgentState


def _route_after_validation(state: AgentState) -> str:
    if state.get("validation_result", {}).get("critical_failure"):
        return "critical_failure"
    return "continue"


def build_graph() -> StateGraph:
    """Construct (but do not compile) the seven-agent AgentState graph."""
    graph = StateGraph(AgentState)

    graph.add_node("sensor_validation", sensor_validation_agent.run)
    graph.add_node("domain_shift", domain_shift_agent.run)
    graph.add_node("prediction", prediction_agent.run)
    graph.add_node("shap", shap_agent.run)
    graph.add_node("uncertainty", uncertainty_agent.run)
    graph.add_node("decision", decision_agent.run)

    graph.set_entry_point("sensor_validation")
    graph.add_conditional_edges(
        "sensor_validation",
        _route_after_validation,
        {"critical_failure": "decision", "continue": "domain_shift"},
    )
    graph.add_edge("domain_shift", "prediction")
    graph.add_edge("prediction", "shap")
    graph.add_edge("shap", "uncertainty")
    graph.add_edge("uncertainty", "decision")
    graph.add_edge("decision", END)

    return graph


def compile_agentic_system():
    """Build and compile the graph — the object notebooks/scripts should actually call `.invoke()` on."""
    return build_graph().compile()
