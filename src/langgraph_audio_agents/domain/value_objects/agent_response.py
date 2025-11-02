"""Agent response model."""

from typing import Any

from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """Response from an agent."""

    content: str = Field(default="", description="Agent's detailed response content (text)")
    audio_summary: str = Field(
        default="", description="Conversational summary for audio generation"
    )
    audio_data: bytes | None = Field(default=None, description="Generated audio bytes")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
