"""Memory service interface."""

from abc import ABC, abstractmethod

from langgraph_audio_agents.domain.entities.conversation_summary import ConversationSummary
from langgraph_audio_agents.domain.value_objects.message import Message


class MemoryService(ABC):
    """Abstract base class for memory/summarization services."""

    @abstractmethod
    async def save_summary(self, summary: ConversationSummary) -> None:
        """Save conversation summary to long-term memory.

        Args:
            summary: Conversation summary to save
        """
        pass

    @abstractmethod
    async def retrieve_relevant_summaries(
        self, query: str, limit: int = 5
    ) -> list[ConversationSummary]:
        """Retrieve relevant past conversation summaries.

        Args:
            query: Current query to find relevant past conversations
            limit: Maximum number of summaries to return

        Returns:
            List of relevant conversation summaries
        """
        pass

    @abstractmethod
    async def generate_summary(
        self, messages: list[Message], user_query: str
    ) -> ConversationSummary:
        """Generate summary from conversation messages.

        Args:
            messages: Full conversation history
            user_query: Original user question

        Returns:
            Generated conversation summary
        """
        pass
