"""Research synthesis value object."""

from pydantic import BaseModel, Field


class ResearchSynthesis(BaseModel):
    """Synthesized research findings from search results."""

    answer: str = Field(
        ...,
        description="Clear, concise answer to the user's question based on search results",
    )
    key_facts: list[str] = Field(
        default_factory=list, description="Key facts extracted from search results"
    )
    sources: list[str] = Field(
        default_factory=list, description="Sources or citations from the search results"
    )
