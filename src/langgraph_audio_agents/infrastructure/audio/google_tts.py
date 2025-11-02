"""Google Cloud Text-to-Speech implementation."""

from enum import Enum

from google.cloud import texttospeech as tts

from langgraph_audio_agents.config import GoogleTTSSettings
from langgraph_audio_agents.domain.interfaces.audio_service import AudioService
from langgraph_audio_agents.domain.value_objects.tts_request import TTSRequest
from langgraph_audio_agents.domain.value_objects.tts_response import TTSResponse


class GoogleVoiceType(str, Enum):
    """Available voice types for Google TTS agents."""

    AOEDE_FEMALE = "en-US-Chirp3-HD-Aoede"
    PERSEUS_MALE = "en-US-Chirp3-HD-Perseus"


class GoogleTTS(AudioService):
    """Google Cloud text-to-speech service implementation."""

    def __init__(self, settings: GoogleTTSSettings, voice_id: str | None = None):
        """Initialize Google Cloud TTS service.

        Args:
            settings: Google TTS settings from config (includes credentials path)
            voice_id: Voice ID to use (defaults to researcher_voice_id from settings)
        """
        self.settings = settings
        self.voice_id = voice_id or settings.researcher_voice_id
        self.client = tts.TextToSpeechClient()

    async def synthesize(self, text: str) -> bytes:
        """Convert text to audio using Google Cloud TTS.

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

        # Voice configuration
        voice_params = tts.VoiceSelectionParams(
            language_code=self.settings.language_code,
            name=request.voice_id,
        )

        # Audio configuration - use MP3 for better compression and compatibility
        audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.MP3,
            sample_rate_hertz=24000,
        )

        # Synthesis input
        synthesis_input = tts.SynthesisInput(text=request.text)

        # Perform synthesis
        response = self.client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config,
        )

        return response.audio_content

    async def synthesize_with_response(self, text: str) -> TTSResponse:
        """Convert text to audio and return full response model.

        Args:
            text: Text to convert to speech

        Returns:
            TTSResponse with audio data and metadata
        """
        audio_data = await self.synthesize(text)

        return TTSResponse(
            audio_data=audio_data,
            format=self.settings.output_format,
            voice_id=self.voice_id,
            duration=None,
        )

    def set_voice(self, voice_id: str | GoogleVoiceType) -> None:
        """Change the voice for this TTS instance.

        Args:
            voice_id: Voice ID to use (can be GoogleVoiceType enum or string)
        """
        self.voice_id = voice_id if isinstance(voice_id, str) else voice_id.value

    def use_researcher_voice(self) -> None:
        """Switch to researcher voice from settings."""
        self.voice_id = self.settings.researcher_voice_id

    def use_validator_voice(self) -> None:
        """Switch to validator voice from settings."""
        self.voice_id = self.settings.validator_voice_id
