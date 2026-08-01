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
from app.runtime.provider_evidence import (
    capture_provider_response,
    provider_evidence_from_run,
)
from app.runtime.provider_accounting import (
    begin_provider_attempt,
    capture_active_provider_evidence,
    finalize_provider_attempt,
    mark_provider_attempt_dispatched,
    reset_active_provider_attempt,
)

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


class EvidenceDeepSeek(DeepSeek):
    """DeepSeek adapter with one durable row per outbound SDK invocation."""

    def _capture_request_evidence(self, parsed):
        capture_active_provider_evidence(provider_evidence_from_run(parsed))
        return parsed

    def _parse_provider_response(self, response, response_format=None):
        parsed = super()._parse_provider_response(response, response_format)
        parsed = capture_provider_response(parsed, response)
        return self._capture_request_evidence(parsed)

    def _parse_provider_response_delta(self, response_delta):
        parsed = super()._parse_provider_response_delta(response_delta)
        parsed = capture_provider_response(parsed, response_delta)
        return self._capture_request_evidence(parsed)

    @staticmethod
    def _run_response(args, kwargs):
        if kwargs.get("run_response") is not None:
            return kwargs.get("run_response")
        return args[5] if len(args) > 5 else None

    def invoke(self, *args, **kwargs):
        attempt, token = begin_provider_attempt(
            requested_model=self.id,
            thinking_enabled=getattr(self, "use_thinking", None),
            stream=False,
            run_response=self._run_response(args, kwargs),
        )
        try:
            mark_provider_attempt_dispatched(attempt)
            result = super().invoke(*args, **kwargs)
        except BaseException as exc:
            try:
                finalize_provider_attempt(
                    attempt,
                    normal_completion=False,
                    exception=exc,
                    phase="non_stream_provider_call",
                )
            finally:
                reset_active_provider_attempt(token)
            raise
        try:
            finalize_provider_attempt(attempt, normal_completion=True)
            return result
        finally:
            reset_active_provider_attempt(token)

    async def ainvoke(self, *args, **kwargs):
        attempt, token = begin_provider_attempt(
            requested_model=self.id,
            thinking_enabled=getattr(self, "use_thinking", None),
            stream=False,
            run_response=self._run_response(args, kwargs),
        )
        try:
            mark_provider_attempt_dispatched(attempt)
            result = await super().ainvoke(*args, **kwargs)
        except BaseException as exc:
            try:
                finalize_provider_attempt(
                    attempt,
                    normal_completion=False,
                    exception=exc,
                    phase="non_stream_provider_call",
                )
            finally:
                reset_active_provider_attempt(token)
            raise
        try:
            finalize_provider_attempt(attempt, normal_completion=True)
            return result
        finally:
            reset_active_provider_attempt(token)

    def invoke_stream(self, *args, **kwargs):
        attempt, token = begin_provider_attempt(
            requested_model=self.id,
            thinking_enabled=getattr(self, "use_thinking", None),
            stream=True,
            run_response=self._run_response(args, kwargs),
        )
        try:
            mark_provider_attempt_dispatched(attempt)
            for chunk in super().invoke_stream(*args, **kwargs):
                yield chunk
            attempt.sdk_stream_exhausted = True
        except BaseException as exc:
            finalize_provider_attempt(
                attempt,
                normal_completion=False,
                exception=exc,
                phase="stream_provider_call",
            )
            raise
        else:
            finalize_provider_attempt(attempt, normal_completion=True)
        finally:
            reset_active_provider_attempt(token)

    async def ainvoke_stream(self, *args, **kwargs):
        attempt, token = begin_provider_attempt(
            requested_model=self.id,
            thinking_enabled=getattr(self, "use_thinking", None),
            stream=True,
            run_response=self._run_response(args, kwargs),
        )
        try:
            mark_provider_attempt_dispatched(attempt)
            async for chunk in super().ainvoke_stream(*args, **kwargs):
                yield chunk
            attempt.sdk_stream_exhausted = True
        except BaseException as exc:
            finalize_provider_attempt(
                attempt,
                normal_completion=False,
                exception=exc,
                phase="stream_provider_call",
            )
            raise
        else:
            finalize_provider_attempt(attempt, normal_completion=True)
        finally:
            reset_active_provider_attempt(token)


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

    return EvidenceDeepSeek(
        id=resolved_id,
        api_key=api_key,
        base_url=base_url,
        use_thinking=use_thinking,
        timeout=120,
        retries=0,
        # The OpenAI-compatible SDK otherwise retries twice below our
        # accounting boundary. Explicit application retries re-enter this
        # gateway and therefore receive independent Provider attempt rows.
        max_retries=0,
    )


# Default production model instance: DeepSeek V4 Flash with reasoning enabled.
# V4 Pro is reserved for explicit calls (A/B tests and Darwin deep-fix).
MODEL = build_model(MODEL_ID)

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
