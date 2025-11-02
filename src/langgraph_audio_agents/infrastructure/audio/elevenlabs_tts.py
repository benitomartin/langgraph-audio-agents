"""ElevenLabs Text-to-Speech implementation."""

import io

from elevenlabs.client import ElevenLabs

from langgraph_audio_agents.config import ElevenLabsSettings
from langgraph_audio_agents.domain.interfaces.audio_service import AudioService
from langgraph_audio_agents.domain.value_objects.tts_request import TTSRequest


class ElevenLabsTTS(AudioService):
    """ElevenLabs text-to-speech service implementation."""

    def __init__(self, settings: ElevenLabsSettings, voice_id: str | None = None):
        """Initialize ElevenLabs TTS service.

        Args:
            settings: ElevenLabs settings from config (includes API key)
            voice_id: Voice ID to use (defaults to researcher_voice_id from settings)
        """
        self.settings = settings
        self.voice_id = voice_id or settings.researcher_voice_id
        self.client = ElevenLabs(api_key=settings.api_key.get_secret_value())

    async def synthesize(self, text: str) -> bytes:
        """Convert text to audio using ElevenLabs.

        Args:
            text: Text to convert to speech

        Returns:
            Audio data as bytes
        """
        # Create request model
        request = TTSRequest(
            text=text,
            voice_id=self.voice_id,
            model_id=self.settings.model_id,
            output_format=self.settings.output_format,
        )

        # ElevenLabs returns an iterator of audio chunks
        audio_generator = self.client.text_to_speech.convert(
            text=request.text,
            voice_id=request.voice_id,
            model_id=request.model_id,
            output_format=request.output_format,
        )

        # Collect all audio chunks into bytes
        audio_bytes = io.BytesIO()
        for chunk in audio_generator:
            audio_bytes.write(chunk)

        return audio_bytes.getvalue()

    def use_validator_voice(self) -> None:
        """Switch to validator voice from settings."""
        self.voice_id = self.settings.validator_voice_id
