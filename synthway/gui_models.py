"""Model choices for the desktop interface, independent of API credentials."""

GUI_MODELS = {
    "Claude Opus 4.5": "anthropic/claude-opus-4.5",
    "Llama 4 Maverick": "meta-llama/llama-4-maverick",
    "Nemotron 3 Nano 30B A3B": "nvidia/nemotron-3-nano-30b-a3b",
}

DEFAULT_GENERATOR = "Claude Opus 4.5"
DEFAULT_JUDGE = "Llama 4 Maverick"


def resolve_model_pair(generator: str, judge: str) -> tuple[str, str]:
    """Translate the two displayed names to exact OpenRouter identifiers."""
    try:
        return GUI_MODELS[generator], GUI_MODELS[judge]
    except KeyError as exc:
        raise ValueError("Choose a listed model for both Generation and Judge.") from exc
