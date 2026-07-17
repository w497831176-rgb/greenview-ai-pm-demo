"""
Model Config API
================

CRUD for model_configs table, default model selection, and A/B test endpoint
that calls flash and pro models concurrently.
"""

import asyncio
import json
import os
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.observability import _check_budget
from app.settings import build_model
from app.utils.cost_utils import build_price_snapshot, compute_cost_cny
from db.property_db import (
    create_model_config as db_create_model_config,
    delete_model_config as db_delete_model_config,
    get_enabled_price_for_model,
    get_model_config as db_get_model_config,
    list_model_configs as db_list_model_configs,
    record_model_call,
    set_default_model_config as db_set_default_model_config,
    update_model_config as db_update_model_config,
)

_BUDGET_BLOCKED_DETAIL = "预算已达上限，Darwin/AI 分类等 Pro/额外评估操作被阻止，请联系管理员调整预算或等待次日刷新"

router = APIRouter(prefix="/api/model-configs", tags=["model-configs"])


class ModelConfigCreate(BaseModel):
    model_id: str
    name: str
    provider: str = "deepseek"
    base_url: Optional[str] = None
    model_params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    is_default: bool = False
    enabled: bool = True
    description: str = ""


class ModelConfigUpdate(BaseModel):
    name: str
    provider: str = "deepseek"
    base_url: Optional[str] = None
    model_params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    is_default: bool = False
    enabled: bool = True
    description: str = ""


class AbTestRequest(BaseModel):
    prompt: str
    # model_a / model_b are intentionally ignored. A/B is fixed to Flash vs Pro
    # so that callers cannot use this endpoint to route Pro into owner chat.
    model_a: Optional[str] = None
    model_b: Optional[str] = None


# Sensitive fields that must never be returned by read endpoints.
_SENSITIVE_FIELDS = {"api_key"}


def _sanitize_config(config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return a frontend-safe view of the config.

    Never returns api_key. Adds a read-only credential_status so the UI can
    show whether the shared server-side credential is configured.
    """
    if not config:
        return None
    safe = dict(config)
    for field in _SENSITIVE_FIELDS:
        safe.pop(field, None)

    model_id = safe.get("model_id") or ""
    params = safe.get("model_params") or {}
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except Exception:
            params = {}

    safe["key"] = model_id
    safe["thinking_enabled"] = bool(params.get("use_thinking", False))
    safe["credential_status"] = "server_env" if os.getenv("DEEPSEEK_API_KEY") else "missing"
    return safe


@router.get("")
async def list_model_configs():
    """List all model configurations."""
    configs = db_list_model_configs()
    return {"model_configs": [_sanitize_config(c) for c in configs], "count": len(configs)}


@router.get("/{config_id}")
async def get_model_config(config_id: int):
    """Get a single model configuration."""
    config = db_get_model_config(config_id)
    if not config:
        raise HTTPException(status_code=404, detail="not found")
    return {"model_config": _sanitize_config(config)}


@router.post("")
async def create_model_config(request: ModelConfigCreate):
    """Create a new model configuration.

    Browser-submitted API keys are intentionally ignored; the runtime uses the
    shared server-side credential configured in the deployment environment.
    """
    config = db_create_model_config(
        model_id=request.model_id,
        name=request.name,
        provider=request.provider,
        api_key=None,
        base_url=request.base_url,
        model_params=request.model_params or {},
        is_default=request.is_default,
        enabled=request.enabled,
        description=request.description,
    )
    return {"model_config": _sanitize_config(config)}


@router.put("/{config_id}")
async def update_model_config(config_id: int, request: ModelConfigUpdate):
    """Update a model configuration.

    Browser-submitted API keys are intentionally ignored; the runtime uses the
    shared server-side credential configured in the deployment environment.
    """
    config = db_update_model_config(
        config_id=config_id,
        name=request.name,
        provider=request.provider,
        api_key=None,
        base_url=request.base_url,
        model_params=request.model_params or {},
        is_default=request.is_default,
        enabled=request.enabled,
        description=request.description,
    )
    if not config:
        raise HTTPException(status_code=404, detail="not found")
    return {"model_config": _sanitize_config(config)}


@router.delete("/{config_id}")
async def delete_model_config(config_id: int):
    """Delete a model configuration."""
    deleted = db_delete_model_config(config_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="not found")
    return {"ok": True, "deleted_id": config_id}


@router.post("/{config_id}/default")
async def set_default_model_config(config_id: int):
    """Set a model configuration as the default."""
    config = db_set_default_model_config(config_id)
    if not config:
        raise HTTPException(status_code=404, detail="not found")
    return {"model_config": _sanitize_config(config)}


@router.post("/ab-test")
async def ab_test_models(request: AbTestRequest):
    """Concurrently call two models (default flash and pro) and return both responses.

    Useful for evaluating which model works better for a specific prompt.
    """

    trace_id = uuid.uuid4().hex[:16]

    # A/B test uses Pro; enforce the daily budget before spending budget.
    budget = _check_budget("ab_test")
    if budget.get("alert_level") == "blocked":
        try:
            record_model_call(
                trace_id=trace_id,
                stage="ab_test",
                model_id="deepseek-v4-pro",
                status="blocked",
                latency_ms=0,
                usage_source="unavailable",
                model_selection_reason="A/B test blocked by daily budget",
                error_summary=budget.get("reason") or _BUDGET_BLOCKED_DETAIL,
                estimated_cost_cny=None,
                price_snapshot=None,
            )
        except Exception:
            pass
        raise HTTPException(status_code=403, detail=_BUDGET_BLOCKED_DETAIL)

    def _get_price_snapshot(model_id: str) -> Optional[Dict[str, Any]]:
        price = get_enabled_price_for_model(model_id)
        return build_price_snapshot(price)

    def _calculate_cost(model_id: str, usage: Dict[str, Optional[int]]) -> tuple:
        snapshot = _get_price_snapshot(model_id)
        cost, _status = compute_cost_cny(snapshot, usage)
        return cost, snapshot

    async def _collect_response(generator) -> tuple:
        response = ""
        usage = {}
        try:
            if isinstance(generator, str):
                return generator, usage
            if hasattr(generator, "__aiter__"):
                async for chunk in generator:
                    if hasattr(chunk, "content") and chunk.content:
                        response += str(chunk.content)
                    elif hasattr(chunk, "delta") and chunk.delta:
                        response += str(chunk.delta)
                    elif isinstance(chunk, str):
                        response += chunk
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage = _extract_usage(chunk.usage)
                return response.strip(), usage
            result = await generator
            if hasattr(result, "content"):
                if hasattr(result, "usage") and result.usage:
                    usage = _extract_usage(result.usage)
                return str(result.content).strip(), usage
            if isinstance(result, str):
                return result.strip(), usage
            return "", usage
        except Exception:
            import traceback
            traceback.print_exc()
            return "", usage

    def _extract_usage(usage_obj: Any) -> Dict[str, Optional[int]]:
        if isinstance(usage_obj, dict):
            return {
                "input_tokens": usage_obj.get("input_tokens") or usage_obj.get("prompt_tokens"),
                "output_tokens": usage_obj.get("output_tokens") or usage_obj.get("completion_tokens"),
                "reasoning_tokens": usage_obj.get("reasoning_tokens"),
                "cached_tokens": usage_obj.get("cached_tokens") or usage_obj.get("prompt_cache_hit_tokens"),
                "total_tokens": usage_obj.get("total_tokens"),
            }
        return {
            "input_tokens": getattr(usage_obj, "input_tokens", None) or getattr(usage_obj, "prompt_tokens", None),
            "output_tokens": getattr(usage_obj, "output_tokens", None) or getattr(usage_obj, "completion_tokens", None),
            "reasoning_tokens": getattr(usage_obj, "reasoning_tokens", None),
            "cached_tokens": getattr(usage_obj, "cached_tokens", None) or getattr(usage_obj, "prompt_cache_hit_tokens", None),
            "total_tokens": getattr(usage_obj, "total_tokens", None),
        }

    async def _run_model(stage: str, model_id: str, prompt: str) -> Dict[str, Any]:
        start = time.time()
        status = "success"
        error_summary = None
        response_text = ""
        usage = {}
        try:
            model = build_model(model_id)
            from agno.agent import Agent

            agent = Agent(model=model, markdown=False)
            response_text, usage = await _collect_response(agent.arun(prompt, stream=False))
        except Exception as e:
            import traceback
            traceback.print_exc()
            status = "failed"
            error_summary = str(e)[:300]
        latency_ms = int((time.time() - start) * 1000)
        total_tokens = usage.get("total_tokens") or (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
        usage_source = "provider_reported" if usage.get("total_tokens") else "estimated_tokenization" if total_tokens else "unavailable"
        cost_cny, snapshot = _calculate_cost(model_id, usage)
        try:
            record_model_call(
                trace_id=trace_id,
                stage=stage,
                model_id=model_id,
                status=status,
                latency_ms=latency_ms,
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                reasoning_tokens=usage.get("reasoning_tokens"),
                cached_tokens=usage.get("cached_tokens"),
                total_tokens=total_tokens,
                usage_source=usage_source,
                model_selection_reason="A/B test fixed model",
                error_summary=error_summary,
                price_snapshot=snapshot,
                estimated_cost_cny=cost_cny,
            )
        except Exception:
            pass
        return {
            "model_id": model_id,
            "response": response_text,
            "error": error_summary,
            "latency_ms": latency_ms,
            "total_tokens": total_tokens,
            "usage_source": usage_source,
            "estimated_cost_cny": cost_cny,
        }

    # A/B is fixed to Flash vs Pro regardless of what the client sends.
    FIXED_MODEL_A = "deepseek-v4-flash"
    FIXED_MODEL_B = "deepseek-v4-pro"
    a_task = asyncio.create_task(_run_model("ab_test_a", FIXED_MODEL_A, request.prompt))
    b_task = asyncio.create_task(_run_model("ab_test_b", FIXED_MODEL_B, request.prompt))
    a_result, b_result = await asyncio.gather(a_task, b_task)

    return {
        "trace_id": trace_id,
        "prompt": request.prompt,
        "model_a": FIXED_MODEL_A,
        "model_b": FIXED_MODEL_B,
        "model_a_result": a_result,
        "model_b_result": b_result,
    }
