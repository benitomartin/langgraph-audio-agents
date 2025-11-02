"""Conversation state entity."""

from typing import Any

from pydantic import BaseModel, Field

from langgraph_audio_agents.domain.value_objects.message import Message


class ConversationState(BaseModel):
    """State of the multi-agent conversation."""

    messages: list[Message] = Field(default_factory=list, description="Conversation history")
    user_query: str = Field(default="", description="Original user question")
    research_result: str = Field(default="", description="Research agent findings")
    validation_result: str = Field(default="", description="Validator agent assessment")
    is_validated: bool = Field(default=False, description="Whether research is validated")
    final_answer: str = Field(default="", description="Final answer to user")
    audio_data: bytes | None = Field(default=None, description="Generated audio data")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional state metadata")
