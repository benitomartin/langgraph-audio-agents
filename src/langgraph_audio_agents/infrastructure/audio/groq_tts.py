"""Groq Text-to-Speech implementation."""

from enum import Enum
from typing import Literal, cast

from groq import Groq

from langgraph_audio_agents.config import GroqSettings
from langgraph_audio_agents.domain.interfaces.audio_service import AudioService
from langgraph_audio_agents.domain.value_objects.tts_request import TTSRequest


class GroqVoiceType(str, Enum):
    """Available voice types for Groq TTS agents."""

    ARISTA_FEMALE = "Arista-PlayAI"
    FRITZ_MALE = "Fritz-PlayAI"


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

    # async def synthesize_with_response(self, text: str) -> TTSResponse:
    #     """Convert text to audio and return full response model.

    #     Args:
    #         text: Text to convert to speech

    #     Returns:
    #         TTSResponse with audio data and metadata
    #     """
    #     audio_data = await self.synthesize(text)

    #     return TTSResponse(
    #         audio_data=audio_data,
    #         format=self.settings.output_format,
    #         voice_id=self.voice_id,
    #         duration=None,
    #     )

    # def set_voice(self, voice_id: str | GroqVoiceType) -> None:
    #     """Change the voice for this TTS instance.

    #     Args:
    #         voice_id: Voice ID to use (can be GroqVoiceType enum or string)
    #     """
    #     self.voice_id = voice_id if isinstance(voice_id, str) else voice_id.value

    # def use_researcher_voice(self) -> None:
    #     """Switch to researcher voice from settings."""
    #     self.voice_id = self.settings.researcher_voice_id

    def use_validator_voice(self) -> None:
        """Switch to validator voice from settings."""
        self.voice_id = self.settings.validator_voice_id
