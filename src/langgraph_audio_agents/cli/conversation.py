"""Interactive CLI for conversational research and validation with checkpoint persistence."""

import asyncio
import contextlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from langgraph_audio_agents.agents.researcher import ResearcherAgent
from langgraph_audio_agents.agents.validator import ValidatorAgent
from langgraph_audio_agents.config import settings
from langgraph_audio_agents.domain.entities.conversation_state import ConversationState
from langgraph_audio_agents.domain.value_objects.message import Message
from langgraph_audio_agents.graph.research_validation_graph import (
    create_research_validation_graph,
)
from langgraph_audio_agents.infrastructure.audio.elevenlabs_tts import ElevenLabsTTS
from langgraph_audio_agents.infrastructure.audio.google_tts import GoogleTTS
from langgraph_audio_agents.infrastructure.audio.groq_tts import GroqTTS
from langgraph_audio_agents.infrastructure.llm.openai_client import OpenAIClient
from langgraph_audio_agents.infrastructure.search.tavily_search import TavilySearch
from langgraph_audio_agents.utils.checkpoint_utils import (
    list_all_thread_ids,
    list_topics_for_user,
    list_users,
    normalize_thread_id,
)


def play_audio_sync(audio_data: bytes, format: str = "wav") -> None:
    """Play audio synchronously using mpv or ffplay.

    Args:
        audio_data: Audio bytes to play
        format: Audio format (wav or mp3)
    """
    players = [
        (["mpv", "--no-video", "--really-quiet"], "mpv"),
        (["ffplay", "-nodisp", "-autoexit"], "ffplay"),
    ]

    for player_cmd, _player_name in players:
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name

            subprocess.run(
                player_cmd + [temp_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            os.unlink(temp_path)
            return

        except FileNotFoundError:
            continue
        except Exception:
            with contextlib.suppress(Exception):
                os.unlink(temp_path)
            continue

    print("⚠️  No audio player found. Install mpv or ffmpeg")


async def main() -> None:
    """Interactive CLI for research and validation with checkpoint persistence and audio."""
    print("🎤 Interactive Research & Validation Conversation")
    print("=" * 80)
    print("\nInitializing services...")

    tavily = TavilySearch(settings.tavily)

    from langgraph_audio_agents.domain.interfaces.audio_service import (
        AudioService,
    )

    if settings.tts_provider == "groq":
        researcher_tts: AudioService = GroqTTS(settings.groq)  # type: ignore[assignment]
        validator_tts: AudioService = GroqTTS(settings.groq)  # type: ignore[assignment]
        validator_tts.use_validator_voice()  # type: ignore[attr-defined]
        audio_format = "wav"
        print("✓ Using Groq TTS")
    elif settings.tts_provider == "google":
        researcher_tts = GoogleTTS(settings.google_tts)
        validator_tts = GoogleTTS(settings.google_tts)
        validator_tts.use_validator_voice()  # type: ignore[attr-defined]
        audio_format = "mp3"
        print("✓ Using Google Cloud TTS")
    else:
        researcher_tts = ElevenLabsTTS(settings.elevenlabs)
        validator_tts = ElevenLabsTTS(settings.elevenlabs)
        validator_tts.use_validator_voice()  # type: ignore[attr-defined]
        audio_format = "mp3"
        print("✓ Using ElevenLabs TTS")

    llm = OpenAIClient(settings.openai)

    researcher = ResearcherAgent(
        search_service=tavily,
        audio_service=researcher_tts,
        llm_client=llm,
    )

    validator = ValidatorAgent(
        audio_service=validator_tts,
        llm_client=llm,
        confidence_threshold=70,
    )

    print("Creating graph with async SQLite checkpointer...")
    workflow = create_research_validation_graph(researcher, validator)
    db_path = Path("checkpoints.db")

    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)

        # Get all existing thread_ids from database
        thread_ids = await list_all_thread_ids(db_path)

        # Step 1: Select or create user
        print("\n" + "=" * 80)
        print("STEP 1: Select User")
        print("=" * 80)

        users = list_users(thread_ids)
        if users:
            print("\nAvailable users:")
            for i, user in enumerate(users, 1):
                print(f"  {i}. {user}")
            print(f"  {len(users) + 1}. Create new user")

            while True:
                try:
                    choice = input(f"\nSelect user (1-{len(users) + 1}): ").strip()
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(users):
                        selected_user = users[choice_num - 1]
                        break
                    elif choice_num == len(users) + 1:
                        selected_user = input("Enter new user name: ").strip()
                        if not selected_user:
                            print("User name cannot be empty. Please try again.")
                            continue
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(users) + 1}")
                except ValueError:
                    print("Please enter a valid number")
        else:
            print("\nNo existing users found.")
            selected_user = input("Enter new user name: ").strip()
            if not selected_user:
                selected_user = "default-user"
                print(f"Using default user: {selected_user}")

        print(f"\nSelected user: {selected_user}")

        # Step 2: Select or create topic for the user
        print("\n" + "=" * 80)
        print("STEP 2: Select Topic")
        print("=" * 80)

        topics = list_topics_for_user(thread_ids, selected_user)
        if topics:
            print(f"\nAvailable topics for '{selected_user}':")
            for i, topic in enumerate(topics, 1):
                print(f"  {i}. {topic}")
            print(f"  {len(topics) + 1}. Create new topic")

            while True:
                try:
                    choice = input(f"\nSelect topic (1-{len(topics) + 1}): ").strip()
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(topics):
                        selected_topic = topics[choice_num - 1]
                        break
                    elif choice_num == len(topics) + 1:
                        selected_topic = input("Enter new topic name: ").strip()
                        if not selected_topic:
                            print("Topic name cannot be empty. Please try again.")
                            continue
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(topics) + 1}")
                except ValueError:
                    print("Please enter a valid number")
        else:
            print(f"\nNo existing topics for '{selected_user}'.")
            selected_topic = input("Enter new topic name: ").strip()
            if not selected_topic:
                selected_topic = "general"
                print(f"Using default topic: {selected_topic}")

        print(f"\nSelected topic: {selected_topic}")

        # Generate thread_id
        thread_id = normalize_thread_id(selected_user, selected_topic)
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        print(f"\nUsing thread_id: {thread_id}")
        print("This allows you to continue the conversation across multiple runs!\n")

        # Check if we have previous state
        try:
            previous_state = await graph.aget_state(config)
            if previous_state.values and previous_state.values.get("messages"):
                print("=" * 80)
                print("PREVIOUS CONVERSATION FOUND!")
                print("=" * 80)
                for i, msg in enumerate(previous_state.values["messages"], 1):
                    print(f"{i}. [{msg.role}]: {msg.content[:100]}...")
                print("\nContinuing conversation...\n")
        except Exception:
            print("No previous conversation found. Starting fresh!\n")

        # Get user input
        user_query = input("Enter your question (or press Enter for default): ").strip()
        if not user_query:
            user_query = "What is LangGraph and how does it work?"

        print(f"\nUser query: {user_query}\n")

        # Get existing state and append new message
        try:
            previous_state = await graph.aget_state(config)
            if previous_state.values:
                existing_messages = previous_state.values.get("messages", [])
                # Preserve metadata from previous state (includes validation_history)
                existing_metadata = previous_state.values.get("metadata", {})
            else:
                existing_messages = []
                existing_metadata = {}
        except Exception:
            existing_messages = []
            existing_metadata = {}

        initial_state = ConversationState(
            messages=existing_messages + [Message(role="user", content=user_query)],
            user_query=user_query,
            metadata=existing_metadata,  # Preserve metadata including validation_history
        )

        print("Running research + validation workflow...")
        print("=" * 80)

        # Stream events to capture both agent responses and play audio sequentially
        researcher_data = None
        validator_data = None

        async for event in graph.astream(initial_state, config=config):  # type: ignore[arg-type]
            if "researcher" in event:
                researcher_data = event["researcher"]
                print("\n🔬 Researcher says:")
                print(f"   {researcher_data['messages'][-1].content[:200]}...")

                # Play researcher audio
                if researcher_data.get("audio_data"):
                    print("   🔊 Playing researcher audio...")
                    play_audio_sync(researcher_data["audio_data"], audio_format)
                    print("   ✓ Audio finished\n")

            if "validator" in event:
                validator_data = event["validator"]
                confidence = validator_data["metadata"].get("confidence_score", "N/A")
                is_validated = validator_data.get("is_validated", False)
                status = "✓ VALIDATED" if is_validated else "✗ NOT VALIDATED"

                print("\n✅ Validator says:")
                print(f"   {validator_data['messages'][-1].content[:200]}...")
                print(f"   📊 Confidence: {confidence}% - {status}")

                # Play validator audio
                if validator_data.get("audio_data"):
                    print("   🔊 Playing validator audio...")
                    play_audio_sync(validator_data["audio_data"], audio_format)
                    print("   ✓ Audio finished\n")

        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)

        if researcher_data and validator_data:
            print(f"\nTotal messages in thread: {len(validator_data.get('messages', []))}")
            print(f"\nResearch Result:\n{researcher_data.get('research_result', 'N/A')}\n")
            print(f"Validation Result:\n{validator_data.get('validation_result', 'N/A')}\n")
            status = "✓ VALIDATED" if validator_data.get("is_validated") else "✗ NOT VALIDATED"
            print(f"Validation Status: {status}")
            print(
                f"Confidence Score: {validator_data['metadata'].get('confidence_score', 'N/A')}%\n"
            )

        print("=" * 80)
        print(f"Conversation state saved to checkpoints.db with thread_id: {thread_id}")
        print("Run this script again to continue the conversation!")
        print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
