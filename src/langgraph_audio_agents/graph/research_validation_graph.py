"""LangGraph for research and validation workflow."""

from typing import Any

from langgraph.graph import END, StateGraph

from langgraph_audio_agents.agents.researcher import ResearcherAgent
from langgraph_audio_agents.agents.validator import ValidatorAgent
from langgraph_audio_agents.domain.entities.conversation_state import ConversationState
from langgraph_audio_agents.graph.nodes import researcher_node, validator_node


def create_research_validation_graph(
    researcher_agent: ResearcherAgent,
    validator_agent: ValidatorAgent,
) -> StateGraph[ConversationState]:
    """Create a graph with researcher and validator agents in sequence.

    Args:
        researcher_agent: Initialized researcher agent
        validator_agent: Initialized validator agent

    Returns:
        StateGraph workflow (call .compile() separately)
    """
    workflow = StateGraph(ConversationState)

    async def _researcher_node(state: ConversationState) -> dict[str, Any]:
        return await researcher_node(state, researcher_agent)

    async def _validator_node(state: ConversationState) -> dict[str, Any]:
        return await validator_node(state, validator_agent)

    workflow.add_node("researcher", _researcher_node)
    workflow.add_node("validator", _validator_node)

    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "validator")
    workflow.add_edge("validator", END)

    return workflow
