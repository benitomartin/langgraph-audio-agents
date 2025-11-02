"""Tavily search service implementation."""

from tavily import TavilyClient

from langgraph_audio_agents.config import TavilySettings
from langgraph_audio_agents.domain.interfaces.search_service import SearchService


class TavilySearch(SearchService):
    """Tavily web search service implementation."""

    def __init__(self, settings: TavilySettings):
        """Initialize Tavily search service.

        Args:
            settings: Tavily settings from config (includes API key)
        """
        self.settings = settings
        self.client = TavilyClient(api_key=settings.api_key.get_secret_value())

    async def search(self, query: str) -> str:
        """Search for information using Tavily.

        Args:
            query: Search query

        Returns:
            Search results as formatted text
        """
        response = self.client.search(
            query=query,
            max_results=self.settings.max_results,
            search_depth=self.settings.search_depth,
        )

        return self._format_results(response)

    def _format_results(self, response: dict) -> str:
        """Format Tavily search results into readable text.

        Args:
            response: Raw response from Tavily API

        Returns:
            Formatted search results as text
        """
        results = response.get("results", [])

        if not results:
            return "No results found."

        formatted = []
        for idx, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "No content available")

            formatted.append(f"{idx}. {title}\n   URL: {url}\n   {content}\n")

        return "\n".join(formatted)
