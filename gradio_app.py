"""Gradio web interface for conversational research and validation with checkpoint persistence."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gradio as gr
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.langgraph_audio_agents.agents.researcher import ResearcherAgent
from src.langgraph_audio_agents.agents.validator import ValidatorAgent
from src.langgraph_audio_agents.config import settings
from src.langgraph_audio_agents.domain.entities.conversation_state import ConversationState
from src.langgraph_audio_agents.domain.interfaces.audio_service import AudioService
from src.langgraph_audio_agents.graph.research_validation_graph import (
    create_research_validation_graph,
)
from src.langgraph_audio_agents.infrastructure.audio.elevenlabs_tts import ElevenLabsTTS
from src.langgraph_audio_agents.infrastructure.audio.google_tts import GoogleTTS
from src.langgraph_audio_agents.infrastructure.audio.groq_tts import GroqTTS
from src.langgraph_audio_agents.infrastructure.llm.openai_client import OpenAIClient
from src.langgraph_audio_agents.infrastructure.search.tavily_search import TavilySearch
from src.langgraph_audio_agents.utils.checkpoint_utils import (
    list_all_thread_ids,
    list_topics_for_user,
    list_users,
    normalize_thread_id,
)

executor = ThreadPoolExecutor(max_workers=1)
event_loop = None


def get_event_loop():  # type: ignore[no-untyped-def]
    """Get or create persistent event loop for async operations."""
    global event_loop
    if event_loop is None or event_loop.is_closed():
        event_loop = asyncio.new_event_loop()

        def run_loop():  # type: ignore[no-untyped-def]
            asyncio.set_event_loop(event_loop)
            event_loop.run_forever()

        executor.submit(run_loop)
    return event_loop


class ConversationApp:
    def __init__(self) -> None:
        self.graph = None
        self.checkpointer = None
        self.checkpointer_context = None
        self.db_path = Path(settings.checkpoint_db_path)
        self.audio_format = "wav"
        self.researcher = None
        self.validator = None
        self.initialized = False

    async def initialize_services(self) -> str:
        """Initialize all services including agents, TTS, and graph."""
        if self.initialized:
            return "Services already initialized"

        tavily = TavilySearch(settings.tavily)

        if settings.tts_provider == "groq":
            researcher_tts: AudioService = GroqTTS(settings.groq)  # type: ignore[assignment]
            validator_tts: AudioService = GroqTTS(settings.groq)  # type: ignore[assignment]
            validator_tts.use_validator_voice()  # type: ignore[attr-defined]
            self.audio_format = "wav"
            status = "Using Groq TTS"
        elif settings.tts_provider == "google":
            researcher_tts = GoogleTTS(settings.google_tts)
            validator_tts = GoogleTTS(settings.google_tts)
            validator_tts.use_validator_voice()  # type: ignore[attr-defined]
            self.audio_format = "mp3"
            status = "Using Google Cloud TTS"
        else:
            researcher_tts = ElevenLabsTTS(settings.elevenlabs)
            validator_tts = ElevenLabsTTS(settings.elevenlabs)
            validator_tts.use_validator_voice()  # type: ignore[attr-defined]
            self.audio_format = "mp3"
            status = "Using ElevenLabs TTS"

        llm = OpenAIClient(settings.openai)

        self.researcher = ResearcherAgent(
            search_service=tavily,
            audio_service=researcher_tts,
            llm_client=llm,
        )

        self.validator = ValidatorAgent(
            audio_service=validator_tts,
            llm_client=llm,
            confidence_threshold=70,
        )

        workflow = create_research_validation_graph(self.researcher, self.validator)

        self.checkpointer_context = AsyncSqliteSaver.from_conn_string(str(self.db_path))
        self.checkpointer = await self.checkpointer_context.__aenter__()
        self.graph = workflow.compile(checkpointer=self.checkpointer)
        self.initialized = True

        return f"Services initialized. {status}"

    async def get_users(self) -> list[str]:
        """Get list of all users from checkpoint database."""
        thread_ids = await list_all_thread_ids(self.db_path)
        users = list_users(thread_ids)
        return ["[Create New User]"] + users

    async def get_topics(self, user: str) -> list[str]:
        """Get list of topics for a specific user."""
        if user == "[Create New User]" or not user:
            return ["[Create New Topic]"]
        thread_ids = await list_all_thread_ids(self.db_path)
        topics = list_topics_for_user(thread_ids, user)
        return ["[Create New Topic]"] + topics

    async def load_conversation_history(self, user: str, topic: str) -> list[list[str]]:
        """Load conversation history for a user and topic."""
        if not self.graph:
            return []

        if user == "[Create New User]" or topic == "[Create New Topic]":
            return []

        thread_id = normalize_thread_id(user, topic)
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        try:
            previous_state = await self.graph.aget_state(config)
            if previous_state.values and previous_state.values.get("messages"):
                messages = previous_state.values["messages"]
                history = []
                current_user = None
                agent_responses = []

                for msg in messages:
                    if msg.role == "user":
                        if current_user is not None:
                            response = "\n\n".join(agent_responses) if agent_responses else None
                            history.append([current_user, response])
                            agent_responses = []
                        current_user = msg.content
                    elif msg.role == "agent" or msg.role == "assistant":
                        agent_responses.append(msg.content)

                if current_user is not None:
                    response = "\n\n".join(agent_responses) if agent_responses else None
                    history.append([current_user, response])

                return history
        except Exception as e:
            print(f"Error loading history: {e}")

        return []

    async def process_conversation_stream(
        self,
        user: str,
        topic: str,
        new_user: str,
        new_topic: str,
        user_query: str,
        history: list[list[str]],
    ):  # type: ignore[no-untyped-def]
        """Process conversation stream with researcher and validator agents."""
        if not self.graph:
            await self.initialize_services()

        final_user = new_user if user == "[Create New User]" else user
        final_topic = new_topic if topic == "[Create New Topic]" else topic

        if not final_user or not final_topic or not user_query:
            yield history, None, None, "", ""
            return

        thread_id = normalize_thread_id(final_user, final_topic)
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        try:
            previous_state = await self.graph.aget_state(config)
            if previous_state.values:
                existing_messages = previous_state.values.get("messages", [])
                existing_metadata = previous_state.values.get("metadata", {})
            else:
                existing_messages = []
                existing_metadata = {}
        except Exception:
            existing_messages = []
            existing_metadata = {}

        messages_list = []
        for msg in existing_messages:
            if isinstance(msg, dict):
                messages_list.append(msg)
            else:
                messages_list.append({"role": msg.role, "content": msg.content})

        messages_list.append({"role": "user", "content": user_query})

        initial_state = ConversationState(
            messages=messages_list,
            user_query=user_query,
            metadata=existing_metadata,
        )

        history.append([user_query, None])

        researcher_audio_path = None
        validator_audio_path = None
        researcher_text = ""
        validator_text = ""

        async for event in self.graph.astream(initial_state, config=config):  # type: ignore[arg-type]
            if "researcher" in event:
                researcher_data = event["researcher"]
                researcher_text = researcher_data["messages"][-1].content
                if researcher_data.get("audio_data"):
                    researcher_audio_path = self._save_temp_audio(
                        researcher_data["audio_data"], "researcher"
                    )
                    partial_response = (
                        f"**Researcher:**\n{researcher_text}\n\n**Validator:**\n_Processing..._"
                    )
                    history[-1][1] = partial_response
                    yield (
                        history,
                        researcher_audio_path,
                        None,
                        "",
                        "Researcher complete. Validating...",
                    )

            if "validator" in event:
                validator_data = event["validator"]
                validator_text = validator_data["messages"][-1].content
                confidence = validator_data["metadata"].get("confidence_score", "N/A")
                is_validated = validator_data.get("is_validated", False)
                status = "VALIDATED" if is_validated else "NOT VALIDATED"
                validator_text = f"{validator_text}\n\nConfidence: {confidence}% - {status}"

                if validator_data.get("audio_data"):
                    validator_audio_path = self._save_temp_audio(
                        validator_data["audio_data"], "validator"
                    )

        response = f"**Researcher:**\n{researcher_text}\n\n**Validator:**\n{validator_text}"
        history[-1][1] = response

        yield (
            history,
            researcher_audio_path,
            validator_audio_path,
            "",
            f"Conversation saved with thread_id: {thread_id}",
        )

    def _save_temp_audio(self, audio_data: bytes, prefix: str) -> str:
        """Save audio data to temporary file and return path."""
        import tempfile

        with tempfile.NamedTemporaryFile(
            suffix=f".{self.audio_format}", delete=False, prefix=prefix
        ) as temp_file:
            temp_file.write(audio_data)
            return temp_file.name

    async def cleanup(self) -> None:
        """Clean up async resources."""
        if self.checkpointer_context:
            await self.checkpointer_context.__aexit__(None, None, None)


app = ConversationApp()


def run_async(coro):  # type: ignore[no-untyped-def]
    """Run async coroutine in persistent event loop."""
    loop = get_event_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def sync_initialize() -> tuple[str, gr.Dropdown]:
    """Initialize services and return status with user dropdown."""
    status = run_async(app.initialize_services())
    users = run_async(app.get_users())
    return status, gr.Dropdown(choices=users, value=users[0] if users else None)


def sync_get_users() -> list[str]:
    """Get list of users synchronously."""
    return run_async(app.get_users())


def sync_get_topics(user: str) -> tuple[gr.Dropdown, list[list[str]]]:
    """Get topics for user and return dropdown with empty chat history."""
    topics = run_async(app.get_topics(user))
    return gr.Dropdown(choices=topics, value=topics[0] if topics else None), []


def sync_load_history(user: str, topic: str) -> list[list[str]]:
    """Load conversation history synchronously."""
    return run_async(app.load_conversation_history(user, topic))


def sync_process(
    user: str,
    topic: str,
    new_user: str,
    new_topic: str,
    query: str,
    history: list[list[str]],
):  # type: ignore[no-untyped-def]
    """Process conversation query and stream results."""
    loop = get_event_loop()
    async_gen = app.process_conversation_stream(user, topic, new_user, new_topic, query, history)

    while True:
        try:
            future = asyncio.run_coroutine_threadsafe(async_gen.__anext__(), loop)
            result = future.result()
            yield result
        except StopAsyncIteration:
            break


with gr.Blocks(title="Research & Validation Conversation") as demo:
    gr.Markdown("# Interactive Research & Validation Conversation")
    gr.Markdown("Ask questions and get validated research responses with audio playback")

    with gr.Row():
        init_btn = gr.Button("Initialize Services", variant="primary")
        status_text = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        with gr.Column(scale=1):
            user_dropdown = gr.Dropdown(
                label="Select User",
                choices=["[Create New User]"],
                value="[Create New User]",
                interactive=True,
            )
            new_user_input = gr.Textbox(
                label="New User Name (if creating new)", placeholder="Enter user name"
            )

        with gr.Column(scale=1):
            topic_dropdown = gr.Dropdown(
                label="Select Topic",
                choices=["[Create New Topic]"],
                value="[Create New Topic]",
                interactive=True,
            )
            new_topic_input = gr.Textbox(
                label="New Topic Name (if creating new)", placeholder="Enter topic name"
            )

    load_btn = gr.Button("Load Conversation History")

    chatbot = gr.Chatbot(label="Conversation", height=400, type="tuples")

    with gr.Row():
        query_input = gr.Textbox(
            label="Your Question",
            placeholder="What would you like to know?",
            scale=4,
        )
        submit_btn = gr.Button("Submit", variant="primary", scale=1)

    with gr.Row():
        researcher_audio = gr.Audio(label="Researcher Audio", autoplay=True)
        validator_audio = gr.Audio(label="Validator Audio", autoplay=False)

    thread_info = gr.Textbox(label="Thread Info", interactive=False)

    init_btn.click(
        fn=sync_initialize,
        outputs=[status_text, user_dropdown],
    )

    user_dropdown.change(
        fn=sync_get_topics,
        inputs=user_dropdown,
        outputs=[topic_dropdown, chatbot],
    )

    load_btn.click(
        fn=sync_load_history,
        inputs=[user_dropdown, topic_dropdown],
        outputs=chatbot,
    )

    submit_btn.click(
        fn=sync_process,
        inputs=[
            user_dropdown,
            topic_dropdown,
            new_user_input,
            new_topic_input,
            query_input,
            chatbot,
        ],
        outputs=[chatbot, researcher_audio, validator_audio, query_input, thread_info],
    )

    query_input.submit(
        fn=sync_process,
        inputs=[
            user_dropdown,
            topic_dropdown,
            new_user_input,
            new_topic_input,
            query_input,
            chatbot,
        ],
        outputs=[chatbot, researcher_audio, validator_audio, query_input, thread_info],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
