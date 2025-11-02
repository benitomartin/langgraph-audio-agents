"""Conversation summary model for long-term memory."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ConversationSummary(BaseModel):
    """Summary of a conversation for long-term memory."""

    conversation_id: str = Field(..., description="Unique identifier for the conversation")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When conversation occurred"
    )
    user_query: str = Field(..., description="Original user question")
    summary: str = Field(..., description="Condensed summary of the conversation")
    key_findings: list[str] = Field(default_factory=list, description="Important facts or findings")
    topics: list[str] = Field(default_factory=list, description="Main topics discussed")
    outcome: str = Field(default="", description="Final result/answer provided")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
