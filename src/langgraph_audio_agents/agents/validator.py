"""Validator agent implementation."""

from typing import Any

from langgraph_audio_agents.prompts.validator_prompts import (
    get_validation_system_prompt,
    get_validation_user_prompt,
    get_validator_audio_summary_system_prompt,
    get_validator_audio_summary_user_prompt,
)
from langgraph_audio_agents.domain.interfaces.agent import Agent
from langgraph_audio_agents.domain.interfaces.audio_service import AudioService
from langgraph_audio_agents.domain.value_objects.agent_response import AgentResponse
from langgraph_audio_agents.domain.value_objects.message import Message
from langgraph_audio_agents.domain.value_objects.validation_result import ValidationResult
from langgraph_audio_agents.infrastructure.llm.openai_client import OpenAIClient


class ValidatorAgent(Agent):
    """Agent that validates research findings and produces conversational audio."""

    def __init__(
        self,
        audio_service: AudioService,
        llm_client: OpenAIClient | None = None,
        confidence_threshold: int = 70,
    ):
        """Initialize the validator agent.

        Args:
            audio_service: Service to generate audio from text (e.g., TTS)
            llm_client: LLM client for analyzing research results
            confidence_threshold: Minimum confidence score (0-100) to consider validated
        """
        self.audio_service = audio_service
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold

    async def process(
        self,
        messages: list[Message],
        previous_validations: list[dict[str, Any]] | None = None,
    ) -> AgentResponse:
        """Validate research findings and generate conversational audio response.

        Args:
            messages: Conversation history including user query and research results
            previous_validations: Previous validation results for improvement tracking

        Returns:
            Validation assessment with text, audio summary, and generated audio
        """
        user_query = self._extract_user_query(messages)
        research_result = self._extract_research_result(messages)

        validation_result = await self._validate_research(
            user_query, research_result, messages, previous_validations=previous_validations
        )

        confidence_score = validation_result.confidence_score
        detailed_validation = validation_result.assessment

        audio_summary_text = await self._generate_audio_summary(
            user_query, detailed_validation, confidence_score, messages
        )

        audio_data = await self.audio_service.synthesize(audio_summary_text)

        return AgentResponse(
            content=detailed_validation,
            audio_summary=audio_summary_text,
            audio_data=audio_data,
            metadata={
                "agent": "validator",
                "query": user_query,
                "confidence_score": confidence_score,
                "assessment": detailed_validation,
                "is_validated": confidence_score >= self.confidence_threshold,
            },
        )

    def _extract_user_query(self, messages: list[Message]) -> str:
        """Extract the user's question from message history.

        Args:
            messages: Conversation history

        Returns:
            User's query string
        """
        for message in reversed(messages):
            if message.role == "user":
                return message.content
        return ""

    def _extract_research_result(self, messages: list[Message]) -> str:
        """Extract the researcher's findings from message history.

        Args:
            messages: Conversation history

        Returns:
            Research result string
        """
        for message in reversed(messages):
            if message.role == "agent":
                return message.content
        return ""

    async def _validate_research(
        self,
        query: str,
        research_result: str,
        messages: list[Message],
        previous_validations: list[dict[str, Any]] | None = None,
    ) -> ValidationResult:
        """Use LLM to validate research findings.

        Args:
            query: User's original query
            research_result: Research findings to validate
            messages: Conversation history for context
            previous_validations: Previous validation results for improvement tracking

        Returns:
            ValidationResult with confidence_score and assessment
        """
        if not self.llm_client:
            return ValidationResult(
                confidence_score=75,
                assessment=f"Validation for '{query}': The research appears comprehensive.",
            )

        # Debug logging
        if previous_validations:
            print(
                f"\n[DEBUG] Validator received {len(previous_validations)} previous validation(s)"
            )
            for i, val in enumerate(previous_validations, 1):
                print(f"  Previous {i}: Score {val.get('confidence_score', 'N/A')}%")
        else:
            print("\n[DEBUG] Validator received no previous validations")

        system_prompt = get_validation_system_prompt()
        user_prompt = get_validation_user_prompt(
            query,
            research_result,
            conversation_history=messages,
            previous_validations=previous_validations,
        )

        # Use structured output for guaranteed parsing
        validation_result = await self.llm_client.parse_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            text_format=ValidationResult,
        )

        # Debug logging after validation
        if previous_validations:
            prev_score = previous_validations[-1].get("confidence_score", 0)
            new_score = validation_result.confidence_score
            if new_score > prev_score:
                print(
                    f"\n[DEBUG] Score improved: {prev_score}% → {new_score}% "
                    f"(+{new_score - prev_score} points)"
                )
            elif new_score == prev_score:
                print(
                    f"\n[DEBUG] Score unchanged: {new_score}% (check if gaps were actually covered)"
                )

        return validation_result

    async def _generate_audio_summary(
        self,
        query: str,
        validation_result: str,
        confidence_score: int,
        messages: list[Message],
    ) -> str:
        """Generate a conversational audio summary from validation.

        Args:
            query: User's original query
            validation_result: Detailed validation assessment
            confidence_score: Confidence score (0-100)
            messages: Conversation history for context

        Returns:
            Conversational summary suitable for audio (human-like speech)
        """
        if not self.llm_client:
            return f"I've reviewed the research about {query}. Confidence: {confidence_score}%"

        system_prompt = get_validator_audio_summary_system_prompt()
        user_prompt = get_validator_audio_summary_user_prompt(
            query, validation_result, confidence_score, conversation_history=messages
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

        return f"Validation results (unprocessed): {user_prompt}"
