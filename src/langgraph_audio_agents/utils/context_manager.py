"""Context manager for conversation summarization and context building."""

from langgraph_audio_agents.application.services.conversation_summarizer import (
    summarize_conversation,
)
from langgraph_audio_agents.domain.value_objects.message import Message
from langgraph_audio_agents.infrastructure.llm.openai_client import OpenAIClient
from langgraph_audio_agents.utils.conversation_manager import (
    get_messages_to_summarize,
    get_recent_exchanges,
    should_summarize,
)


async def manage_conversation_context(
    messages: list[Message],
    llm_client: OpenAIClient,
    max_exchanges: int = 5,
    max_tokens: int = 10000,
) -> list[Message]:
    """Manage conversation context by summarizing when needed.

    Args:
        messages: Current conversation messages
        llm_client: LLM client for summarization
        max_exchanges: Maximum exchanges before summarizing (default: 5)
        max_tokens: Maximum tokens before summarizing (default: 10000)

    Returns:
        Updated messages list with summary if needed
    """
    # Check if we need to summarize
    if not should_summarize(messages, max_exchanges, max_tokens):
        return messages

    # Get messages to summarize and recent messages to keep
    messages_to_summarize = get_messages_to_summarize(messages, max_exchanges)
    recent_messages = get_recent_exchanges(messages, max_exchanges)

    if not messages_to_summarize:
        return messages

    # Create summary
    summary_text = await summarize_conversation(messages_to_summarize, llm_client)
    summary_message = Message(
        role="system", content=f"Previous conversation summary: {summary_text}"
    )

    # Combine: summary + recent messages
    return [summary_message] + recent_messages
