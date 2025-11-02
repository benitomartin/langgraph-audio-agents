"""Audio service interface."""

from abc import ABC, abstractmethod


class AudioService(ABC):
    """Abstract base class for text-to-speech services."""

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Convert text to audio.

        Args:
            text: Text to convert to speech

        Returns:
            Audio data as bytes
        """
        pass
