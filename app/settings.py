"""
Shared Settings
===============

Centralizes the model, database, and environment flags
so all agents share the same resources.
"""

import json
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

# Runtime default for all owner-facing chat paths.
MODEL_ID = "deepseek-v4-flash"
USE_THINKING = True


def _deepseek_api_key() -> str:
    return getenv("DEEPSEEK_API_KEY", "")


def _deepseek_base_url() -> str:
    return getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")


def _model_params_from_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = config.get("model_params") if config else None
    if not raw:
        return {}
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return dict(raw)


def build_model(model_id: Optional[str] = None, **overrides) -> DeepSeek:
    """Build a DeepSeek model instance.

    The single runtime default is ``MODEL_ID`` (deepseek-v4-flash). If a
    caller passes an explicit ``model_id`` (e.g. deepseek-v4-pro for Darwin),
    that value is used strictly. SQLite model_configs may supply a fallback
    api_key/base_url/params, but they can never override the resolved id.
    """
    resolved_id = model_id or MODEL_ID

    # Try to enrich from the matching DB config; otherwise fall back to the
    # default config only for non-secret metadata. Secrets always come from env.
    config = get_model_config_by_model_id(resolved_id)
    if not config:
        config = get_default_model_config()

    cfg_params = _model_params_from_config(config) if config else {}
    use_thinking = overrides.get(
        "use_thinking",
        cfg_params.get("use_thinking", USE_THINKING),
    )

    # Environment credentials take precedence over DB-stored secrets.
    api_key = overrides.get("api_key") or _deepseek_api_key() or (config.get("api_key") if config else None)
    base_url = overrides.get(
        "base_url",
        (config.get("base_url") if config else None) or _deepseek_base_url(),
    )

    return DeepSeek(
        id=resolved_id,
        api_key=api_key,
        base_url=base_url,
        use_thinking=use_thinking,
        timeout=120,
    )


# Default production model instance: DeepSeek V4 Flash with reasoning enabled.
# V4 Pro is reserved for explicit calls (A/B tests and Darwin deep-fix).
MODEL = build_model(MODEL_ID)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
RUNTIME_ENV = getenv("RUNTIME_ENV", "prd")
RUNTIME_ENGINE = getenv("RUNTIME_ENGINE", "v18").strip().lower()
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
