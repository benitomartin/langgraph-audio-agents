"""Prompts for the validator agent."""

from typing import Any

from langgraph_audio_agents.domain.value_objects.message import Message


def get_validation_system_prompt() -> str:
    """Get system prompt for validating research findings.

    Returns:
        System prompt for validation
    """
    return """You are a validation expert in an ongoing conversation. Your job is to analyze
research findings and assess their accuracy, completeness, and relevance to the user's question.
Identify any potential issues, missing information, or areas that need clarification.

When validating:
- Consider the conversation context - is this a follow-up question building on previous answers?
- If previous information was discussed, check for consistency with earlier findings
- Assess whether the research addresses the current question appropriately in the conversation
  flow
- CRITICAL: If previous validation results are provided, carefully check if this research
  addresses the gaps or missing information identified in previous validations. If gaps are
  now covered, you MUST increase the confidence score accordingly (typically 5-15 points
  improvement). This is important for tracking learning and improvement.

You must provide:
1. A confidence score (0-100) where:
   - 0-40: Poor quality, significant issues
   - 41-70: Acceptable but has issues
   - 71-85: Good quality, minor issues
   - 86-100: Excellent quality
2. A detailed assessment explaining your score, explicitly mentioning if previously identified
   gaps have been addressed and how this affects the score."""


def get_validation_user_prompt(
    query: str,
    research_result: str,
    conversation_history: list[Message] | None = None,
    previous_validations: list[dict[str, Any]] | None = None,
) -> str:
    """Get user prompt for validating research findings.

    Args:
        query: User's original question
        research_result: Research findings to validate
        conversation_history: Previous messages in the conversation (optional)
        previous_validations: Previous validation results for improvement tracking

    Returns:
        User prompt for validation
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

    # Include previous validation results for improvement tracking
    if previous_validations:
        prompt_parts.append("=" * 80)
        prompt_parts.append("PREVIOUS VALIDATION HISTORY - CRITICAL FOR SCORE IMPROVEMENT")
        prompt_parts.append("=" * 80)
        for i, val in enumerate(previous_validations[-2:], 1):  # Last 2 validations
            score = val.get("confidence_score", "N/A")
            assessment = val.get("assessment", "")
            prompt_parts.append(f"\nPrevious Validation {i}:")
            prompt_parts.append(f"  Score: {score}%")
            prompt_parts.append(f"  Assessment: {assessment}")
            prompt_parts.append("")
        prompt_parts.append("IMPORTANT INSTRUCTIONS:")
        prompt_parts.append("1. Carefully read the previous validation assessments above.")
        prompt_parts.append(
            "2. Identify what information was MISSING or identified as needing improvement."
        )
        prompt_parts.append(
            "3. Check if the current research findings address those missing elements."
        )
        prompt_parts.append(
            "4. If gaps are now covered, you MUST increase the confidence score "
            "(typically 5-15 points higher than the previous score)."
        )
        prompt_parts.append(
            "5. Explicitly state in your assessment which previously missing information "
            "is now included and how this improves the answer quality."
        )
        prompt_parts.append("=" * 80)
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
            # Get last 4-6 messages (2-3 exchanges)
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
    prompt_parts.append("Research Findings:")
    prompt_parts.append(research_result)
    prompt_parts.append("")
    prompt_parts.append("Please validate these research findings. Address:")
    prompt_parts.append("1. Is the information accurate and relevant to the question?")
    prompt_parts.append("2. Are there any factual errors or inconsistencies?")
    prompt_parts.append("3. Is any critical information missing?")
    prompt_parts.append("4. Overall assessment: Does this adequately answer the user's question?")
    if previous_validations:
        prompt_parts.append("")
        prompt_parts.append(
            "5. IMPROVEMENT CHECK (CRITICAL): Compare this research with previous "
            "validation assessments:"
        )
        prompt_parts.append(
            "   a. What specific information was missing or identified as needing "
            "improvement in the previous validation(s)?"
        )
        prompt_parts.append(
            "   b. Does the current research findings include that missing information?"
        )
        prompt_parts.append(
            "   c. If yes, how much does this improve the answer quality? "
            "(This should result in a higher confidence score)"
        )
        prompt_parts.append(
            "   d. Explicitly state the improvement in your assessment and adjust "
            "the score accordingly."
        )

    return "\n".join(prompt_parts)


def get_validator_audio_summary_system_prompt() -> str:
    """Get system prompt for generating conversational audio summary for validator.

    Returns:
        System prompt for validator audio summary generation
    """
    return """You are a validator having a conversation with colleagues.
Your job is to verbally present your validation findings in a natural, conversational way.
Speak as if you're talking to someone, not reading a report.
Keep it concise (2-3 sentences) but informative.
Use natural speech patterns like "I checked...", "I noticed...", "Overall..."
Mention your confidence level naturally.
If this is part of an ongoing conversation, reference previous points naturally when relevant."""


def get_validator_audio_summary_user_prompt(
    query: str,
    validation_result: str,
    confidence_score: int,
    conversation_history: list[Message] | None = None,
) -> str:
    """Get user prompt for generating conversational audio summary for validator.

    Args:
        query: User's original query
        validation_result: Detailed validation assessment
        confidence_score: Confidence score (0-100)
        conversation_history: Previous messages in the conversation (optional)

    Returns:
        User prompt for validator audio summary generation
    """
    prompt_parts = [f'You just validated research about: "{query}"', ""]

    if conversation_history and len(conversation_history) > 2:
        prompt_parts.append("This is part of an ongoing conversation.")
        prompt_parts.append("")

    prompt_parts.append(f"Your confidence score: {confidence_score}/100")
    prompt_parts.append("")
    prompt_parts.append("Your detailed validation:")
    prompt_parts.append(validation_result)
    prompt_parts.append("")
    prompt_parts.append(
        "Now, verbally share your validation assessment in a natural, conversational way "
        "(2-3 sentences). Include your confidence level naturally in the conversation. "
        "If this continues a previous topic, reference it naturally."
    )

    return "\n".join(prompt_parts)
