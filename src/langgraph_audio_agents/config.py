"""Application configuration using pydantic-settings."""

from typing import ClassVar, Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# -----------------------------
# ElevenLabs TTS settings
# -----------------------------
class ElevenLabsSettings(BaseModel):
    """ElevenLabs text-to-speech configuration."""

    api_key: SecretStr = Field(default=SecretStr(""), description="ElevenLabs API key")
    researcher_voice_id: str = Field(
        default="uYXf8XasLslADfZ2MB4u",
        description="Voice ID for researcher agent (Hope - female)",
    )
    validator_voice_id: str = Field(
        default="jRAAK67SEFE9m7ci5DhD",
        description="Voice ID for validator agent (Ollie - male)",
    )
    model_id: str = Field(
        default="eleven_flash_v2_5",  # "eleven_multilingual_v2",
        description="""ElevenLabs model ID: eleven_flash_v2_5 (faster and cheaper), 
                        eleven_multilingual_v2 (mode extensive, slower)""",
    )
    output_format: str = Field(
        default="mp3_44100_128",
        description="Audio output format",
    )


# -----------------------------
# Groq TTS settings
# -----------------------------
class GroqSettings(BaseModel):
    """Groq text-to-speech configuration."""

    api_key: SecretStr = Field(default=SecretStr(""), description="Groq API key")
    researcher_voice_id: str = Field(
        default="Arista-PlayAI",
        description="Voice ID for researcher agent (Arista - female)",
    )
    validator_voice_id: str = Field(
        default="Fritz-PlayAI",
        description="Voice ID for validator agent (Fritz - male)",
    )
    model_id: str = Field(
        default="playai-tts",
        description="Groq TTS model ID",
    )
    output_format: Literal["mp3", "wav"] = Field(
        default="mp3",
        description="Audio output format",
    )


# -----------------------------
# Google Cloud TTS settings
# -----------------------------
class GoogleTTSSettings(BaseModel):
    """Google Cloud text-to-speech configuration."""

    credentials_path: str = Field(
        default="",
        description="Path to Google Cloud credentials JSON file",
    )
    researcher_voice_id: str = Field(
        default="en-US-Chirp3-HD-Aoede",
        description="Voice ID for researcher agent (Aoede - female)",
    )
    validator_voice_id: str = Field(
        default="en-US-Chirp3-HD-Alnilam",
        description="Voice ID for validator agent (Alnilam - male)",
    )
    model_id: str = Field(
        default="Chirp3-HD",
        description="Google TTS model ID",
    )
    language_code: str = Field(
        default="en-US",
        description="Language code for TTS",
    )
    output_format: str = Field(
        default="wav",
        description="Audio output format",
    )


# -----------------------------
# OpenAI settings
# -----------------------------
class OpenAISettings(BaseModel):
    """OpenAI LLM configuration."""

    api_key: SecretStr = Field(default=SecretStr(""), description="OpenAI API key")
    model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model to use",
    )
    temperature: float = Field(
        default=0.7,
        description="Temperature for text generation",
        ge=0.0,
        le=2.0,
    )
    max_output_tokens: int = Field(
        default=100,
        description="Maximum tokens for completion",
        gt=0,
    )


# -----------------------------
# Tavily search settings
# -----------------------------
class TavilySettings(BaseModel):
    """Tavily web search configuration."""

    api_key: SecretStr = Field(default=SecretStr(""), description="Tavily API key")
    max_results: int = Field(default=5, description="Maximum number of search results", ge=1, le=20)
    search_depth: str = Field(
        default="advanced",
        description="Search depth: basic (1 API call) or advanced (2 API calls)",
    )


# -----------------------------
# Main Settings
# -----------------------------
class Settings(BaseSettings):
    """Main application settings."""

    tts_provider: str = Field(
        default="google",
        description="TTS provider to use: 'elevenlabs', 'groq', or 'google'",
    )
    elevenlabs: ElevenLabsSettings = Field(
        default_factory=ElevenLabsSettings,
        description="ElevenLabs TTS configuration",
    )
    groq: GroqSettings = Field(
        default_factory=GroqSettings,
        description="Groq TTS configuration",
    )
    google_tts: GoogleTTSSettings = Field(
        default_factory=GoogleTTSSettings,
        description="Google Cloud TTS configuration",
    )
    openai: OpenAISettings = Field(
        default_factory=OpenAISettings,
        description="OpenAI LLM configuration",
    )
    tavily: TavilySettings = Field(
        default_factory=TavilySettings,
        description="Tavily search configuration",
    )

    # Pydantic v2 model config
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=[".env"],
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
        case_sensitive=False,
        frozen=True,
    )


# -----------------------------
# Instantiate settings
# -----------------------------
settings = Settings()
