"""LangGraph nodes for agent workflows."""

from typing import Any

from langgraph_audio_agents.agents.researcher import ResearcherAgent
from langgraph_audio_agents.agents.validator import ValidatorAgent
from langgraph_audio_agents.domain.entities.conversation_state import ConversationState
from langgraph_audio_agents.domain.value_objects.message import Message
from langgraph_audio_agents.utils.context_manager import manage_conversation_context


async def researcher_node(
    state: ConversationState, researcher_agent: ResearcherAgent
) -> dict[str, Any]:
    """Node that runs the researcher agent.

    Args:
        state: Current conversation state
        researcher_agent: Initialized researcher agent

    Returns:
        Updated state with research results
    """
    response = await researcher_agent.process(state.messages)

    return {
        "research_result": response.content,
        "audio_data": response.audio_data,
        "messages": state.messages + [Message(role="agent", content=response.audio_summary)],
        "metadata": {**state.metadata, **response.metadata},
    }


async def validator_node(
    state: ConversationState, validator_agent: ValidatorAgent
) -> dict[str, Any]:
    """Node that runs the validator agent.

    Args:
        state: Current conversation state
        validator_agent: Initialized validator agent

    Returns:
        Updated state with validation results
    """
    # Extract previous validation results from metadata
    # Build validation history from state metadata
    validation_history = []
    if state.metadata:
        # Look for validation metadata in state
        if "validation_history" in state.metadata:
            validation_history = state.metadata.get("validation_history", [])
        else:
            # Extract from current metadata if it has validation info
            if "confidence_score" in state.metadata:
                validation_history = [
                    {
                        "confidence_score": state.metadata.get("confidence_score"),
                        "assessment": state.metadata.get("assessment", ""),
                        "is_validated": state.metadata.get("is_validated", False),
                    }
                ]

    # Debug logging - show what's in state.metadata
    print(
        f"\n[DEBUG] State metadata keys: {list(state.metadata.keys()) if state.metadata else 'None'}"
    )
    if validation_history:
        print(f"[DEBUG] Validation history found: {len(validation_history)} previous validation(s)")
        for i, val in enumerate(validation_history, 1):
            print(f"  Validation {i}: Score {val.get('confidence_score', 'N/A')}%")
            print(f"    Assessment preview: {val.get('assessment', '')[:100]}...")
    else:
        print("[DEBUG] No previous validation history found")
        if state.metadata:
            print(f"[DEBUG] Available metadata keys: {list(state.metadata.keys())}")
            if "confidence_score" in state.metadata:
                print(
                    f"[DEBUG] Found confidence_score in metadata: {state.metadata.get('confidence_score')}"
                )
                print(
                    f"[DEBUG] Found assessment in metadata: {state.metadata.get('assessment', '')[:100]}..."
                )

    response = await validator_agent.process(
        state.messages, previous_validations=validation_history if validation_history else None
    )

    # Update validation history in metadata
    updated_metadata = {**state.metadata, **response.metadata}
    validation_history.append(
        {
            "confidence_score": response.metadata.get("confidence_score"),
            "assessment": response.metadata.get("assessment", ""),
            "is_validated": response.metadata.get("is_validated", False),
        }
    )
    # Keep only last 2 validations
    updated_metadata["validation_history"] = validation_history[-2:]

    # Manage conversation context (summarize if needed)
    updated_messages = state.messages + [Message(role="agent", content=response.audio_summary)]

    # Summarize conversation if needed (requires LLM client from validator)
    if validator_agent.llm_client:
        updated_messages = await manage_conversation_context(
            updated_messages,
            validator_agent.llm_client,
            max_exchanges=5,
            max_tokens=10000,
        )

    return {
        "validation_result": response.content,
        "is_validated": response.metadata.get("is_validated", False),
        "audio_data": response.audio_data,
        "messages": updated_messages,
        "metadata": updated_metadata,
    }
