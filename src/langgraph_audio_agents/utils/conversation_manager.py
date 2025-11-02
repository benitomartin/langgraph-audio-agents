"""Conversation context management and summarization utilities."""

from typing import Any

import tiktoken

from langgraph_audio_agents.domain.value_objects.message import Message


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in text using tiktoken.

    Args:
        text: Text to count tokens for
        model: Model name for tokenizer (default: gpt-4o)

    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        # Fallback to cl100k_base encoding if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


def estimate_message_tokens(messages: list[Message]) -> int:
    """Estimate total tokens in a list of messages.

    Args:
        messages: List of messages

    Returns:
        Estimated token count
    """
    total = 0
    for msg in messages:
        # Approximate: role + content + formatting overhead
        total += count_tokens(f"{msg.role}: {msg.content}")
    return total


def count_exchanges(messages: list[Message]) -> int:
    """Count complete exchanges in conversation.

    One exchange = user question + researcher response + validator response (3 messages).

    Args:
        messages: List of messages

    Returns:
        Number of complete exchanges
    """
    user_count = sum(1 for msg in messages if msg.role == "user")
    return user_count


def extract_validation_results_from_metadata(
    metadata_history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract validation results from metadata history.

    Args:
        metadata_history: List of metadata dicts from conversation state

    Returns:
        List of validation result dicts with keys: confidence_score, assessment, is_validated
    """
    validations = []
    for meta in metadata_history:
        if "confidence_score" in meta and "agent" in meta and meta.get("agent") == "validator":
            validations.append(
                {
                    "confidence_score": meta.get("confidence_score"),
                    "assessment": meta.get("assessment", ""),
                    "is_validated": meta.get("is_validated", False),
                }
            )

    # Return last 2 validation results (most recent)
    return validations[-2:] if len(validations) >= 2 else validations


def should_summarize(
    messages: list[Message],
    max_exchanges: int = 5,
    max_tokens: int = 10000,
    model: str = "gpt-4o",
) -> bool:
    """Determine if conversation should be summarized.

    Args:
        messages: List of messages
        max_exchanges: Maximum number of exchanges before summarizing (default: 5)
        max_tokens: Maximum tokens before summarizing (default: 10000)
        model: Model name for token counting

    Returns:
        True if summarization is needed
    """
    # Check exchange count
    exchanges = count_exchanges(messages)
    if exchanges > max_exchanges:
        return True

    # Check token count
    token_count = estimate_message_tokens(messages)
    return token_count > max_tokens


def get_recent_exchanges(messages: list[Message], num_exchanges: int = 5) -> list[Message]:
    """Get the most recent N exchanges from messages.

    One exchange = user + researcher + validator (typically 3 messages).

    Args:
        messages: List of all messages
        num_exchanges: Number of exchanges to keep (default: 5)

    Returns:
        List of messages from recent exchanges
    """
    # Count user messages to find exchanges
    user_indices = [i for i, msg in enumerate(messages) if msg.role == "user"]

    if len(user_indices) <= num_exchanges:
        # Not enough exchanges, return all
        return messages

    # Get the starting index for the last N exchanges
    start_index = user_indices[-num_exchanges]
    return messages[start_index:]


def get_messages_to_summarize(messages: list[Message], num_exchanges: int = 5) -> list[Message]:
    """Get messages that should be summarized (everything except recent exchanges).

    Args:
        messages: List of all messages
        num_exchanges: Number of recent exchanges to keep (default: 5)

    Returns:
        List of messages to summarize
    """
    # Count user messages to find exchanges
    user_indices = [i for i, msg in enumerate(messages) if msg.role == "user"]

    if len(user_indices) <= num_exchanges:
        # Not enough exchanges, nothing to summarize
        return []

    # Get everything before the last N exchanges
    start_index = user_indices[-num_exchanges]
    return messages[:start_index]
