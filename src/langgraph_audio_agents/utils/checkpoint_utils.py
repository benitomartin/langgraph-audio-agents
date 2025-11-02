"""Utility functions for checkpoint management and thread ID handling."""

import re
import sqlite3
from pathlib import Path


def normalize_thread_id(user: str, topic: str) -> str:
    """Generate normalized thread_id from user and topic.

    Args:
        user: User name
        topic: Topic name

    Returns:
        Normalized thread_id in format: {user}:{topic}
    """
    # Normalize user: lowercase, trim, replace spaces with hyphens
    normalized_user = re.sub(r"[^a-z0-9_-]", "", user.lower().strip().replace(" ", "-"))
    if not normalized_user:
        normalized_user = "default-user"

    # Normalize topic: lowercase, trim, replace spaces with hyphens, limit length
    normalized_topic = re.sub(r"[^a-z0-9_-]", "", topic.lower().strip().replace(" ", "-"))[:50]
    if not normalized_topic:
        normalized_topic = "general"

    return f"{normalized_user}:{normalized_topic}"


def parse_thread_id(thread_id: str) -> tuple[str, str] | None:
    """Parse thread_id into user and topic components.

    Args:
        thread_id: Thread ID in format user:topic

    Returns:
        Tuple of (user, topic) or None if format is invalid
    """
    if ":" not in thread_id:
        return None

    parts = thread_id.split(":", 1)
    if len(parts) != 2:
        return None

    user, topic = parts
    return (user, topic)


async def list_all_thread_ids(db_path: str | Path) -> list[str]:
    """List all thread_ids from the checkpoint database.

    Args:
        db_path: Path to SQLite checkpoint database

    Returns:
        List of thread_ids
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return []

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # LangGraph stores checkpoints in 'checkpoints' table with 'thread_id' column
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id")
        thread_ids = [row[0] for row in cursor.fetchall()]

        conn.close()
        return thread_ids
    except sqlite3.Error:
        return []


def list_users(thread_ids: list[str]) -> list[str]:
    """Extract unique user names from thread_ids.

    Args:
        thread_ids: List of thread_ids in format user:topic

    Returns:
        Sorted list of unique user names
    """
    users = set()
    for thread_id in thread_ids:
        parsed = parse_thread_id(thread_id)
        if parsed:
            users.add(parsed[0])
    return sorted(list(users))


def list_topics_for_user(thread_ids: list[str], user: str) -> list[str]:
    """Extract topic names for a specific user.

    Args:
        thread_ids: List of thread_ids in format user:topic
        user: User name to filter by (case-insensitive)

    Returns:
        Sorted list of unique topic names for the user
    """
    user_lower = user.lower()
    topics = set()

    for thread_id in thread_ids:
        parsed = parse_thread_id(thread_id)
        if parsed:
            thread_user, topic = parsed
            if thread_user.lower() == user_lower:
                topics.add(topic)

    return sorted(list(topics))


def find_thread_id_for_user_topic(thread_ids: list[str], user: str, topic: str) -> str | None:
    """Find matching thread_id for user and topic (case-insensitive).

    Args:
        thread_ids: List of thread_ids in format user:topic
        user: User name
        topic: Topic name

    Returns:
        Matching thread_id if found, None otherwise
    """
    user_lower = user.lower()
    topic_lower = topic.lower()

    for thread_id in thread_ids:
        parsed = parse_thread_id(thread_id)
        if parsed:
            thread_user, thread_topic = parsed
            if thread_user.lower() == user_lower and thread_topic.lower() == topic_lower:
                return thread_id

    return None
