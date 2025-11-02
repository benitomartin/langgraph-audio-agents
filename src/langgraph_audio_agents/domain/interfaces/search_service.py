"""Search service interface."""

from abc import ABC, abstractmethod


class SearchService(ABC):
    """Abstract base class for search/retrieval services."""

    @abstractmethod
    async def search(self, query: str) -> str:
        """Search for information.

        Args:
            query: Search query

        Returns:
            Search results as text
        """
        pass
