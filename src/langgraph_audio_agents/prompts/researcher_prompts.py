"""Prompts for the researcher agent."""

from langgraph_audio_agents.domain.value_objects.message import Message


def get_synthesis_system_prompt() -> str:
    """Get system prompt for synthesizing search results.

    Returns:
        System prompt for research synthesis
    """
    return """You are a research assistant in an ongoing conversation. Your job is to analyze search
results and provide clear, concise answers to the user's questions.

When responding:
- If this is a follow-up question, reference previous findings naturally (e.g., "Building on what
  we discussed earlier...")
- If the user asks for clarification or more details, expand on relevant points from previous
  answers
- Always be factual and cite key information
- Maintain conversation continuity when appropriate"""


def get_synthesis_user_prompt(
    query: str,
    search_results: str,
    conversation_history: list[Message] | None = None,
) -> str:
    """Get user prompt for synthesizing search results.

    Args:
        query: User's question
        search_results: Raw search results from search service
        conversation_history: Previous messages in the conversation (optional)

    Returns:
        User prompt for research synthesis
    """
    prompt_parts = []

    # Include summary if present (system messages with summaries)
    summary_messages = [
        msg
        for msg in (conversation_history or [])
        if msg.role == "system" and "summary" in msg.content.lower()
    ]
    if summary_messages:
        prompt_parts.append("Previous conversation summary:")
        for summary_msg in summary_messages:
            prompt_parts.append(f"  {summary_msg.content}")
        prompt_parts.append("")

    # Include recent conversation context (last 3-4 messages, excluding summary)
    if conversation_history and len(conversation_history) > 2:
        # Filter out summary messages, keep only user/agent messages
        recent_messages = [
            msg
            for msg in conversation_history
            if msg.role in ("user", "agent") and "summary" not in msg.content.lower()
        ]
        if recent_messages:
            # Get last 4-6 messages (2-3 exchanges, excluding current query)
            if len(recent_messages) > 6:
                messages_to_include = recent_messages[-6:-1]
            else:
                messages_to_include = recent_messages[:-1]
            if messages_to_include:
                prompt_parts.append("Recent conversation context:")
                for msg in messages_to_include:
                    role_label = "User" if msg.role == "user" else "Assistant"
                    content = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
                    prompt_parts.append(f"  {role_label}: {content}")
                prompt_parts.append("")

    prompt_parts.append(f"Current User Question: {query}")
    prompt_parts.append("")
    prompt_parts.append("Search Results:")
    prompt_parts.append(search_results)
    prompt_parts.append("")
    prompt_parts.append(
        "Please provide a well-structured answer based on these search results. "
        "If this is a follow-up question, reference the previous conversation naturally. "
        "Be factual and cite key information."
    )

    return "\n".join(prompt_parts)


def get_audio_summary_system_prompt() -> str:
    """Get system prompt for generating conversational audio summary.

    Returns:
        System prompt for audio summary generation
    """
    return """You are a research assistant having a conversation with colleagues.
Your job is to verbally present your research findings in a natural, conversational way.
Speak as if you're talking to someone, not reading a report.
Keep it concise (2-3 sentences) but informative.
Use natural speech patterns like "I found that...", "It turns out...", "Interestingly..."
If this is part of an ongoing conversation, reference previous points naturally when relevant.
"""


def get_audio_summary_user_prompt(
    query: str,
    detailed_content: str,
    conversation_history: list[Message] | None = None,
) -> str:
    """Get user prompt for generating conversational audio summary.

    Args:
        query: User's original query
        detailed_content: Detailed research findings
        conversation_history: Previous messages in the conversation (optional)

    Returns:
        User prompt for audio summary generation
    """
    prompt_parts = [f'You just researched: "{query}"', ""]

    if conversation_history and len(conversation_history) > 2:
        prompt_parts.append("This is part of an ongoing conversation.")
        prompt_parts.append("")

    prompt_parts.append("Your detailed findings:")
    prompt_parts.append(detailed_content)
    prompt_parts.append("")
    prompt_parts.append(
        "Now, verbally share your key findings in a natural, conversational way (2-3 sentences). "
        "If this continues a previous topic, reference it naturally."
    )

    return "\n".join(prompt_parts)
