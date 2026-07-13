"""
Model Config Compatibility Router
==================================

Exposes the same model-config resources under the URL paths expected by the
frontend (/api/models/*) and by test cases that use model_id identifiers
(/api/model-configs/{model_id}/*).
"""

import asyncio
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.model_configs import AbTestRequest, _sanitize_config
from app.settings import build_model
from db.property_db import (
    create_model_config as db_create_model_config,
    delete_model_config as db_delete_model_config,
    get_default_model_config,
    get_model_config as db_get_model_config,
    get_model_config_by_model_id,
    list_model_configs as db_list_model_configs,
    set_default_model_config as db_set_default_model_config,
    update_model_config as db_update_model_config,
)

router = APIRouter(tags=["models-compat"])


class ModelKeyUpdate(BaseModel):
    api_key: str


class ModelConfigPayload(BaseModel):
    name: str
    provider: str = "deepseek"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_params: Optional[Dict[str, Any]] = {}
    enabled: bool = True
    description: str = ""


def _resolve_config(identifier: str):
    """Resolve either a numeric config id or a model_id string."""
    if identifier.isdigit():
        cfg = db_get_model_config(int(identifier))
        if cfg:
            return cfg
    cfg = get_model_config_by_model_id(identifier)
    if cfg:
        return cfg
    raise HTTPException(status_code=404, detail="model config not found")


@router.get("/api/models")
async def list_models_compat():
    """Frontend alias for GET /api/model-configs."""
    configs = db_list_model_configs()
    return {"models": [_sanitize_config(c) for c in configs], "count": len(configs)}


@router.post("/api/models/{model_key}/key")
async def update_model_key_compat(model_key: str, request: ModelKeyUpdate):
    """Frontend alias for updating a model's API key by model_id.

    The returned config never includes the api_key value.
    """
    cfg = _resolve_config(model_key)
    updated = db_update_model_config(
        config_id=cfg["id"],
        name=cfg["name"],
        provider=cfg.get("provider", "deepseek"),
        api_key=request.api_key,
        base_url=cfg.get("base_url"),
        model_params=cfg.get("model_params") or {},
        is_default=cfg.get("is_default", False),
        enabled=cfg.get("enabled", True),
        description=cfg.get("description", ""),
    )
    return {"model_config": _sanitize_config(updated)}


@router.post("/api/models/{model_key}/default")
async def set_default_model_compat(model_key: str):
    """Frontend alias for setting the default model by model_id."""
    cfg = _resolve_config(model_key)
    updated = db_set_default_model_config(cfg["id"])
    return {"model_config": _sanitize_config(updated)}


@router.post("/api/models/ab-test")
async def ab_test_models_compat(request: AbTestRequest):
    """Frontend alias for /api/model-configs/ab-test."""
    from app.model_configs import ab_test_models

    return await ab_test_models(request)


@router.post("/api/models/ab-test/{ab_id}/score")
async def score_ab_test_compat(ab_id: str, request: Dict[str, Any]):
    """Stub for A/B test scoring (stored in memory for demo purposes)."""
    return {"ok": True, "ab_id": ab_id, "winner": request.get("winner")}


# ---------------------------------------------------------------------------
# Test-case paths that use model_id instead of numeric config id.
# ---------------------------------------------------------------------------


@router.get("/api/model-configs/{model_id}")
async def get_model_config_by_id_compat(model_id: str):
    """Resolve model config by model_id (e.g. deepseek-v4-pro)."""
    cfg = _resolve_config(model_id)
    return {"model_config": _sanitize_config(cfg)}


@router.put("/api/model-configs/{model_id}")
async def update_model_config_by_id_compat(model_id: str, request: ModelConfigPayload):
    """Update model config by model_id."""
    cfg = _resolve_config(model_id)
    updated = db_update_model_config(
        config_id=cfg["id"],
        name=request.name,
        provider=request.provider,
        api_key=request.api_key,
        base_url=request.base_url,
        model_params=request.model_params or {},
        is_default=cfg.get("is_default", False),
        enabled=request.enabled,
        description=request.description,
    )
    return {"model_config": _sanitize_config(updated)}


@router.post("/api/model-configs/{model_id}/set-default")
async def set_default_model_config_by_id_compat(model_id: str):
    """Set default model config by model_id."""
    cfg = _resolve_config(model_id)
    updated = db_set_default_model_config(cfg["id"])
    return {"model_config": _sanitize_config(updated)}
