"""Memory configuration model."""

from pydantic import BaseModel, Field


class Memory(BaseModel):
    """LangGraph memory and checkpointer configuration."""

    checkpointer_type: str = Field(
        default="sqlite",
        description="Checkpointer type: memory, sqlite, or postgres",
    )
    sqlite_path: str = Field(
        default="data/checkpoints.db",
        description="Path to SQLite database for checkpoints",
    )
    max_message_history: int = Field(
        default=50,
        description="Maximum messages to keep in short-term memory",
        gt=0,
    )
    summarization_threshold: int = Field(
        default=10,
        description="Number of messages before triggering summarization",
        gt=0,
    )
