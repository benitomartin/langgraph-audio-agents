"""Basic LangGraph for researcher agent testing."""

from typing import Any

from langgraph.graph import END, StateGraph

from langgraph_audio_agents.agents.researcher import ResearcherAgent
from langgraph_audio_agents.domain.entities.conversation_state import ConversationState
from langgraph_audio_agents.graph.nodes import researcher_node


def create_researcher_graph(researcher_agent: ResearcherAgent) -> StateGraph[ConversationState]:
    """Create a simple graph for testing the researcher agent.

    Args:
        researcher_agent: Initialized researcher agent

    Returns:
        StateGraph workflow (call .compile() separately)
    """
    workflow = StateGraph(ConversationState)

    async def _researcher_node(state: ConversationState) -> dict[str, Any]:
        return await researcher_node(state, researcher_agent)

    workflow.add_node("researcher", _researcher_node)

    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", END)

    return workflow
