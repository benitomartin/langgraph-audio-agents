"""Text-to-Speech response model."""

from pydantic import BaseModel, Field


class TTSResponse(BaseModel):
    """Response model from text-to-speech conversion."""

    audio_data: bytes = Field(default=b"", description="Generated audio data")
    duration: float | None = Field(
        default=None,
        description="Audio duration in seconds (if available)",
    )
    format: str = Field(
        default="mp3",
        description="Audio format",
    )
    voice_id: str = Field(default="", description="Voice ID used for synthesis")
