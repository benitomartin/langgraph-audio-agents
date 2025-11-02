"""Agent interface."""

from abc import ABC, abstractmethod

from langgraph_audio_agents.domain.value_objects.agent_response import AgentResponse
from langgraph_audio_agents.domain.value_objects.message import Message


class Agent(ABC):
    """Abstract base class for agent implementations."""

    @abstractmethod
    async def process(self, messages: list[Message]) -> AgentResponse:
        """Process messages and return a response.

        Args:
            messages: Conversation history

        Returns:
            Agent's response with content and metadata
        """
        pass
