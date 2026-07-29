"""Capture provider-returned model identity and raw DeepSeek usage evidence.

Agno 2.6.21 keeps the provider request id, but its generic OpenAI-compatible
adapter does not retain ``response.model`` or DeepSeek's cache-hit/cache-miss
usage fields.  This module adds only those missing evidence fields; it never
infers one token class from another.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional


def _value(source: Any, name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        value = source.get(name)
        if value is not None:
            return value
        extra = source.get("model_extra")
    else:
        value = getattr(source, name, None)
        if value is not None:
            return value
        extra = getattr(source, "model_extra", None)
    return extra.get(name) if isinstance(extra, dict) else None


def _integer_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _first_integer(source: Any, *names: str) -> Optional[int]:
    for name in names:
        value = _integer_or_none(_value(source, name))
        if value is not None:
            return value
    return None


def raw_provider_usage(usage: Any) -> Dict[str, Optional[int]]:
    """Return only provider-originated values; never derive a missing split."""
    completion_details = _value(usage, "completion_tokens_details")
    return {
        "input_cache_hit_tokens": _first_integer(
            usage,
            "prompt_cache_hit_tokens",
            "input_cache_hit_tokens",
        ),
        "input_cache_miss_tokens": _first_integer(
            usage,
            "prompt_cache_miss_tokens",
            "input_cache_miss_tokens",
        ),
        "input_tokens": _first_integer(usage, "prompt_tokens", "input_tokens"),
        "output_tokens": _first_integer(
            usage,
            "completion_tokens",
            "output_tokens",
        ),
        "reasoning_tokens": (
            _first_integer(completion_details, "reasoning_tokens")
            if completion_details is not None
            else _first_integer(usage, "reasoning_tokens")
        ),
        "total_tokens": _first_integer(usage, "total_tokens"),
    }


def capture_provider_response(model_response: Any, provider_response: Any) -> Any:
    """Attach raw response identity and usage to Agno's provider_data."""
    provider_data = dict(getattr(model_response, "provider_data", None) or {})
    response_model = _value(provider_response, "model")
    if response_model:
        provider_data["response_model"] = str(response_model)

    usage_object = _value(provider_response, "usage")
    if usage_object is not None:
        usage = raw_provider_usage(usage_object)
        if any(value is not None for value in usage.values()):
            provider_data["usage"] = usage

    model_response.provider_data = provider_data
    return model_response


def provider_evidence_from_run(value: Any) -> Dict[str, Any]:
    """Read evidence propagated by Agno RunOutput/RunOutputEvent objects."""
    provider_data = getattr(value, "model_provider_data", None)
    if not isinstance(provider_data, dict):
        provider_data = getattr(value, "provider_data", None)
    if not isinstance(provider_data, dict):
        provider_data = {}

    usage = provider_data.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    request_id = provider_data.get("id")
    return {
        "provider_response_model": provider_data.get("response_model"),
        "provider_request_id": str(request_id) if request_id else None,
        "usage": dict(usage),
    }


def remember_provider_request(
    journal: List[Dict[str, Any]],
    evidence: Dict[str, Any],
) -> None:
    """Merge one parsed Provider response into a per-model request journal.

    Streaming adapters may expose the same Provider request over many deltas.
    A real request id is therefore the primary identity.  Providers without an
    id are appended only when Usage arrives and receive a stable run ordinal.
    """
    request_id = evidence.get("provider_request_id")
    response_model = evidence.get("provider_response_model")
    usage = dict(evidence.get("usage") or {})
    has_usage = any(value is not None for value in usage.values())
    if not request_id and not has_usage:
        return

    existing = None
    if request_id:
        existing = next(
            (
                item
                for item in journal
                if item.get("provider_request_id") == str(request_id)
            ),
            None,
        )
    if existing is None:
        sequence = len(journal) + 1
        existing = {
            "provider_request_id": str(request_id) if request_id else None,
            "provider_request_sequence": sequence,
            "provider_request_key": (
                f"request_id:{request_id}"
                if request_id
                else f"run_sequence:{sequence}"
            ),
            "provider_request_identity_source": (
                "provider_request_id" if request_id else "run_sequence"
            ),
            "provider_response_model": None,
            "usage": {},
            "status": "success",
        }
        journal.append(existing)
    if response_model:
        existing["provider_response_model"] = str(response_model)
    if usage:
        existing["usage"].update(usage)


def provider_requests_from_model(model: Any, *, consume: bool = False) -> List[Dict[str, Any]]:
    """Return every captured Provider request for one model instance."""
    journal = getattr(model, "_provider_request_journal", None)
    if not isinstance(journal, list):
        return []
    result = deepcopy(journal)
    if consume:
        journal.clear()
    return result
