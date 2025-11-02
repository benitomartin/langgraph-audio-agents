"""Groq Text-to-Speech implementation."""

from typing import Literal, cast

from groq import Groq

from langgraph_audio_agents.config import GroqSettings
from langgraph_audio_agents.domain.interfaces.audio_service import AudioService
from langgraph_audio_agents.domain.value_objects.tts_request import TTSRequest


class GroqTTS(AudioService):
    """Groq text-to-speech service implementation."""

    def __init__(self, settings: GroqSettings, voice_id: str | None = None):
        """Initialize Groq TTS service.

        Args:
            settings: Groq settings from config (includes API key)
            voice_id: Voice ID to use (defaults to researcher_voice_id from settings)
        """
        self.settings = settings
        self.voice_id = voice_id or settings.researcher_voice_id
        self.client = Groq(api_key=settings.api_key.get_secret_value())

    async def synthesize(self, text: str) -> bytes:
        """Convert text to audio using Groq.

        Args:
            text: Text to convert to speech

        Returns:
            Audio data as bytes
        """
        request = TTSRequest(
            text=text,
            voice_id=self.voice_id,
            model_id=self.settings.model_id,
            output_format=self.settings.output_format,
        )

        response_format_literal: Literal["mp3", "wav"] = cast(
            Literal["mp3", "wav"], request.output_format
        )

        response = self.client.audio.speech.create(
            model=request.model_id,
            voice=request.voice_id,
            input=request.text,
            response_format=response_format_literal,
        )

        return response.read()

    def use_validator_voice(self) -> None:
        """Switch to validator voice from settings."""
        self.voice_id = self.settings.validator_voice_id
