"""Service for summarizing conversation history."""

from langgraph_audio_agents.domain.value_objects.message import Message
from langgraph_audio_agents.infrastructure.llm.openai_client import OpenAIClient


async def summarize_conversation(
    messages: list[Message],
    llm_client: OpenAIClient,
) -> str:
    """Summarize a conversation history.

    Args:
        messages: Messages to summarize
        llm_client: LLM client for summarization

    Returns:
        Summary text
    """
    if not messages:
        return ""

    # Format messages for summarization
    conversation_text = "\n\n".join(
        [f"{'User' if msg.role == 'user' else 'Assistant'}: {msg.content}" for msg in messages]
    )

    system_prompt = """You are a conversation summarizer. Your job is to create a concise
summary of the conversation, focusing on:
- Main topics and questions discussed
- Key findings and research results
- General themes and direction of the conversation

Keep the summary brief (200-300 tokens). Focus on high-level topics and findings, not
specific validation scores or detailed assessments. This summary will be used to provide
context for future exchanges."""

    user_prompt = f"""Please summarize this conversation:

{conversation_text}

Provide a concise summary focusing on topics discussed and key findings."""

    summary = await llm_client.create_response(
        input=f"{system_prompt}\n\n{user_prompt}",
    )

    return summary.strip()
