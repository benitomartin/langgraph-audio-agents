from langgraph_audio_agents import config


def test_config_importable() -> None:
    """Ensure config module can be imported."""
    if not hasattr(config, "__file__"):
        raise AssertionError("config module missing __file__ attribute")

