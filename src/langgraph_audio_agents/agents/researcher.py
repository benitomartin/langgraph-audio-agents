"""Researcher agent implementation."""

from langgraph_audio_agents.domain.interfaces.agent import Agent
from langgraph_audio_agents.domain.interfaces.audio_service import AudioService
from langgraph_audio_agents.domain.interfaces.search_service import SearchService
from langgraph_audio_agents.domain.value_objects.agent_response import AgentResponse
from langgraph_audio_agents.domain.value_objects.message import Message
from langgraph_audio_agents.domain.value_objects.research_synthesis import (
    ResearchSynthesis,
)
from langgraph_audio_agents.infrastructure.llm.openai_client import OpenAIClient
from langgraph_audio_agents.prompts.researcher_prompts import (
    get_audio_summary_system_prompt,
    get_audio_summary_user_prompt,
    get_synthesis_system_prompt,
    get_synthesis_user_prompt,
)


class ResearcherAgent(Agent):
    """Agent that searches for information and produces conversational audio."""

    def __init__(
        self,
        search_service: SearchService,
        audio_service: AudioService,
        llm_client: OpenAIClient,
    ):
        """Initialize the researcher agent.

        Args:
            search_service: Service to perform searches (e.g., Tavily)
            audio_service: Service to generate audio from text (e.g., TTS)
            llm_client: LLM client for processing search results
        """
        self.search_service = search_service
        self.audio_service = audio_service
        self.llm_client = llm_client

    async def process(self, messages: list[Message]) -> AgentResponse:
        """Search for information and generate conversational audio response.

        Args:
            messages: Conversation history

        Returns:
            Research findings with text, audio summary, and generated audio
        """
        # Extract the user query from messages
        user_query = self._extract_user_query(messages)

        # Perform search using Tavily
        search_results = await self.search_service.search(user_query)

        # Use LLM to synthesize search results into detailed text
        detailed_content = await self._synthesize_results(user_query, search_results, messages)

        # Generate conversational audio summary (human-like, not reading full text)
        audio_summary_text = await self._generate_audio_summary(
            user_query, detailed_content, messages
        )

        # Convert audio summary to speech
        audio_data = await self.audio_service.synthesize(audio_summary_text)

        return AgentResponse(
            content=detailed_content,
            audio_summary=audio_summary_text,
            audio_data=audio_data,
            metadata={
                "agent": "researcher",
                "query": user_query,
                "raw_results": search_results,
            },
        )

    def _extract_user_query(self, messages: list[Message]) -> str:
        """Extract the user's question from message history.

        Args:
            messages: Conversation history

        Returns:
            User's query string
        """
        # Get the last user message as the query
        for message in reversed(messages):
            if message.role == "user":
                return message.content
        return ""

    async def _synthesize_results(
        self, query: str, search_results: str, messages: list[Message]
    ) -> str:
        """Use LLM to synthesize search results into a coherent answer.

        Args:
            query: User's original query
            search_results: Raw search results from Tavily
            messages: Conversation history for context

        Returns:
            Synthesized research findings
        """
        if not self.llm_client:
            # If no LLM client, return raw results
            return f"Research findings for '{query}':\n\n{search_results}"

        system_prompt = get_synthesis_system_prompt()
        user_prompt = get_synthesis_user_prompt(
            query, search_results, conversation_history=messages
        )

        # Use structured output for better parsing
        synthesis_result = await self.llm_client.parse_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            text_format=ResearchSynthesis,
        )

        # Format structured output into readable text
        content_parts = [synthesis_result.answer]
        if synthesis_result.key_facts:
            content_parts.append("\n\nKey Facts:")
            for fact in synthesis_result.key_facts:
                content_parts.append(f"  • {fact}")
        if synthesis_result.sources:
            content_parts.append("\n\nSources:")
            for source in synthesis_result.sources:
                content_parts.append(f"  • {source}")

        return "\n".join(content_parts)

    async def _generate_audio_summary(
        self, query: str, detailed_content: str, messages: list[Message]
    ) -> str:
        """Generate a conversational audio summary from detailed research.

        Args:
            query: User's original query
            detailed_content: Detailed research findings
            messages: Conversation history for context

        Returns:
            Conversational summary suitable for audio (human-like speech)
        """
        if not self.llm_client:
            # Fallback: simple conversational summary
            return f"""Based on my research about {query}, 
            here's what I found: {detailed_content[:200]}..."""

        system_prompt = get_audio_summary_system_prompt()
        user_prompt = get_audio_summary_user_prompt(
            query, detailed_content, conversation_history=messages
        )

        audio_summary = await self._call_llm(system_prompt, user_prompt)
        return audio_summary

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM to generate response.

        Args:
            system_prompt: System instructions
            user_prompt: User query/context

        Returns:
            LLM generated response
        """
        if self.llm_client:
            combined_input = f"{system_prompt}\n\n{user_prompt}"
            return await self.llm_client.create_response(input=combined_input)

        return f"Research results (unprocessed): {user_prompt}"
