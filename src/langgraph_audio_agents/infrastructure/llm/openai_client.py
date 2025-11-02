"""OpenAI LLM client implementation."""

from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from langgraph_audio_agents.config import OpenAISettings

T = TypeVar("T", bound=BaseModel)


class OpenAIClient:
    """OpenAI Responses API client implementation."""

    def __init__(self, settings: OpenAISettings):
        """Initialize OpenAI client.

        Args:
            settings: OpenAI settings from config (includes API key, model, etc.)
        """
        self.settings = settings
        self.client = OpenAI(api_key=settings.api_key.get_secret_value())

    async def create_response(self, input: str, model: str | None = None) -> str:
        """Create a model response using OpenAI Responses API.

        Args:
            input: Text input for the model
            model: Model to use (defaults to settings.model)

        Returns:
            Text response from the model
        """
        response = self.client.responses.create(
            model=model or self.settings.model,
            input=input,
            temperature=self.settings.temperature,
            max_output_tokens=self.settings.max_output_tokens,
        )

        return response.output_text

    async def parse_response(
        self,
        system_prompt: str,
        user_prompt: str,
        text_format: type[T],
        model: str | None = None,
        max_output_tokens: int | None = None,
    ) -> T:
        """Parse a structured response using OpenAI Responses API.

        Args:
            system_prompt: System instructions
            user_prompt: User query/context
            text_format: Pydantic model class for structured output
            model: Model to use (defaults to settings.model)
            max_output_tokens: Maximum tokens for completion
                (defaults to 2000 for structured outputs)

        Returns:
            Parsed structured output as Pydantic model instance
        """
        # Use higher default for structured outputs to prevent JSON truncation
        output_tokens = max_output_tokens if max_output_tokens is not None else 2000

        response = self.client.responses.parse(
            model=model or self.settings.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=text_format,
            temperature=self.settings.temperature,
            max_output_tokens=output_tokens,
        )

        if response.output_parsed is None:
            raise ValueError("Failed to parse structured response from OpenAI")
        return response.output_parsed
