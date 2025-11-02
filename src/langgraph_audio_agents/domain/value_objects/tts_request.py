"""Text-to-Speech request model."""

from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    """Request model for text-to-speech conversion."""

    text: str = Field(default="", description="Text to convert to speech", min_length=1)
    voice_id: str = Field(default="", description="Voice ID to use for synthesis")
    model_id: str = Field(
        default="eleven_multilingual_v2",
        description="Model ID for TTS",
    )
    output_format: str = Field(
        default="mp3_44100_128",
        description="Audio output format",
    )
