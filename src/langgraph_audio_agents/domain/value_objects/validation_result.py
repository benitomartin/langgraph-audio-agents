"""Validation result value object."""

from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """Validation result representing the assessment of research findings."""

    confidence_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Confidence score from 0-100 where 0-40 is poor quality, 41-70 is acceptable, 71-85 is good, and 86-100 is excellent",
    )
    assessment: str = Field(..., description="Detailed assessment explaining the confidence score")
