"""Message model for conversation."""

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Represents a message in the conversation."""

    role: str = Field(
        default="user", description="Role of the message sender (user, agent, system)"
    )
    content: str = Field(default="", description="Content of the message")
