"""
Shared Settings
===============

Centralizes the model, database, and environment flags
so all agents share the same resources.
"""

from os import getenv
from typing import Any, Dict, Optional

from agno.models.deepseek import DeepSeek

from db import get_postgres_db
from db.property_db import get_default_model_config, get_model_config_by_model_id

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
agent_db = get_postgres_db()

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _deepseek_api_key() -> str:
    return getenv("DEEPSEEK_API_KEY", "")


def _deepseek_base_url() -> str:
    return getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")


def build_model(model_id: Optional[str] = None, **overrides) -> DeepSeek:
    """Build a DeepSeek model instance from DB config or fallback to env defaults.

    If model_id is provided, look it up in model_configs; otherwise use the
    default config. Any extra keyword arguments override the stored params.
    """
    config = None
    if model_id:
        config = get_model_config_by_model_id(model_id)
    if not config:
        config = get_default_model_config()

    if config:
        cfg_id = config.get("model_id") or "deepseek-v4-pro"
        cfg_params = config.get("model_params") or {}
        use_thinking = overrides.get("use_thinking", cfg_params.get("use_thinking", True))
        # Always prefer environment variables for secrets; DB value is a fallback.
        api_key = overrides.get("api_key") or _deepseek_api_key() or config.get("api_key")
        base_url = overrides.get("base_url", config.get("base_url") or _deepseek_base_url())
        return DeepSeek(
            id=cfg_id,
            api_key=api_key,
            base_url=base_url,
            use_thinking=use_thinking,
            timeout=120,
        )

    # Fallback to environment defaults when no DB config exists.
    return DeepSeek(
        id=model_id or getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-flash"),
        api_key=_deepseek_api_key(),
        base_url=_deepseek_base_url(),
        use_thinking=overrides.get("use_thinking", True),
    )


# Default production model: DeepSeek V4 Flash with reasoning enabled.
# V4 Pro is reserved for high-quality comparisons (e.g. Skill A/B tests).
MODEL = DeepSeek(
    id=getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-flash"),
    api_key=_deepseek_api_key(),
    base_url=_deepseek_base_url(),
    use_thinking=True,
    timeout=120,
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
RUNTIME_ENV = getenv("RUNTIME_ENV", "prd")
SCHEDULER_BASE_URL = getenv("AGENTOS_URL", "http://127.0.0.1:8000")
SLACK_TOKEN = getenv("SLACK_TOKEN", "")
SLACK_SIGNING_SECRET = getenv("SLACK_SIGNING_SECRET", "")

# ---------------------------------------------------------------------------
# Optional tools
# ---------------------------------------------------------------------------
PARALLEL_API_KEY = getenv("PARALLEL_API_KEY", "")


def get_parallel_tools(**kwargs) -> list:
    """Return ParallelTools if PARALLEL_API_KEY is set, else empty list."""
    if PARALLEL_API_KEY:
        from agno.tools.parallel import ParallelTools

        return [ParallelTools(**kwargs)]
    return []
