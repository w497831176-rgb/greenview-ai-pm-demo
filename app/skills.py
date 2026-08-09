"""Skill management and governed runtime diagnostics.

The console treats a Skill as a versioned business capability, not a hidden
prompt fragment.  CRUD stays compatible with the V1.1 UI while the APIs add
deterministic trigger diagnostics, release snapshots and rollback.
"""

import io
import shutil
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.skill_runtime import canonical_metadata, next_patch_version, select_skills, skill_contract
from db.property_db import (
    create_skill as db_create_skill,
    create_skill_version,
    delete_skill as db_delete_skill,
    get_agent_by_agent_id,
    get_agent_skills,
    get_skill as db_get_skill,
    list_skill_versions,
    list_skills as db_list_skills,
    update_skill as db_update_skill,
)
import skill_storage


router = APIRouter(prefix="/api/skills", tags=["skills"])


class SkillPayload(BaseModel):
    name: str
    description: str = ""
    instructions: str = ""
    category: str = "未分类"
    enabled: bool = True
    trigger_condition: str = ""
    skill_metadata: Dict[str, Any] = Field(default_factory=dict)
    model_id: Optional[str] = None


class SkillCreate(SkillPayload):
    pass


class SkillUpdate(SkillPayload):
    pass


class SkillMdUpdate(BaseModel):
    metadata: Dict[str, Any] = Field(default_factory=dict)
    body: str = ""


class GitImportRequest(BaseModel):
    git_url: str
    trigger_condition: str = ""
    enabled: bool = True


class TestSkillRequest(BaseModel):
    message: str
    agent_id: Optional[str] = None


class ApplyDarwinRequest(BaseModel):
    suggested_prompt: str


def _serialize_skill(skill: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(skill)
    item["has_files"] = skill_storage._skill_dir(item["id"]).exists()
    item["runtime_contract"] = skill_contract(item)
    return item


def _get_or_404(skill_id: int) -> Dict[str, Any]:
    skill = db_get_skill(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail="skill not found")
    return skill


def _persist_current_skill_file(skill: Dict[str, Any]):
    metadata = skill.get("skill_metadata") or {}
    skill_storage.write_skill_md(skill["id"], metadata, skill.get("instructions") or "")
    version = str(metadata.get("version") or "legacy-1.0.0")
    skill_storage.write_skill_revision(skill["id"], version, metadata, skill.get("instructions") or "")


def _save_update(existing: Dict[str, Any], payload: SkillPayload, summary: str) -> Dict[str, Any]:
    """Save a governed Skill edit and record one new immutable version."""
    source = {
        "trigger_condition": payload.trigger_condition,
        "skill_metadata": existing.get("skill_metadata") or {},
    }
    metadata = canonical_metadata(source, payload.skill_metadata)
    # ``trigger_condition`` remains the backwards-compatible database field.
    # When a legacy caller changes it without supplying structured metadata,
    # it must still take effect at runtime.
    if payload.trigger_condition.strip() != (existing.get("trigger_condition") or "").strip():
        metadata["positive_triggers"] = [
            term.strip() for term in payload.trigger_condition.replace("，", ",").split(",") if term.strip()
        ]
    else:
        metadata["positive_triggers"] = metadata.get("positive_triggers") or [
            term.strip() for term in payload.trigger_condition.replace("，", ",").split(",") if term.strip()
        ]
    metadata["version"] = next_patch_version(str(metadata.get("version") or "legacy-1.0.0"))
    updated = db_update_skill(
        skill_id=existing["id"],
        name=payload.name.strip(),
        description=payload.description.strip(),
        instructions=payload.instructions.strip(),
        category=payload.category.strip() or "未分类",
        enabled=payload.enabled,
        trigger_condition=payload.trigger_condition.strip(),
        skill_metadata=metadata,
        storage_path=existing.get("storage_path") or "",
        model_id=existing.get("model_id"),
    )
    _persist_current_skill_file(updated)
    create_skill_version(updated["id"], metadata["version"], summary)
    return updated


def _evaluate_for_agent(skill: Dict[str, Any], message: str, agent_id: Optional[str]) -> Dict[str, Any]:
    """Use the same selector as chat, but without making a model call."""
    target_agent = None
    bound = None
    candidates = [skill]
    if agent_id:
        target_agent = get_agent_by_agent_id(agent_id)
        if not target_agent:
            raise HTTPException(status_code=404, detail="agent not found")
        if agent_id != "router":
            bound_ids = {int(item) for item in get_agent_skills(agent_id)}
            bound = skill["id"] in bound_ids
            candidates = [
                item for item in db_list_skills()
                if item.get("enabled") and int(item.get("id")) in bound_ids
            ]
        else:
            candidates = [item for item in db_list_skills() if item.get("enabled")]
            bound = True
    selected, decisions = select_skills(candidates, message)
    decision = next((item for item in decisions if item.get("skill_id") == skill["id"]), None)
    if decision is None:
        # A disabled Skill is not part of candidate selection, but must still
        # produce an honest diagnostic instead of a misleading empty preview.
        decision = {
            "skill_id": skill["id"], "name": skill.get("name", ""), "matched": False,
            "selected": False, "outcome": "disabled", "match_reason": "Skill 已禁用",
            "contract": skill_contract(skill), "matched_positive": [], "matched_negative": [],
        }
    selected_ids = [item.get("skill_id") for item in selected]
    eligible = bool(skill.get("enabled")) and (bound is not False) and decision.get("selected", False)
    return {
        "message": message,
        "agent": {"agent_id": agent_id, "name": (target_agent or {}).get("name")} if agent_id else None,
        "binding": "bound" if bound is True else ("not_bound" if bound is False else "not_checked"),
        "decision": decision,
        "selected_skill_ids": selected_ids,
        "will_inject": eligible and agent_id != "router",
        "authority_note": "Skill 命中只决定业务方法注入；真实 MCP 工具权限始终由 Agent 的绑定决定。",
    }


@router.get("/import-git")
async def import_git_help():
    """Avoid dynamic-route ambiguity; actual import is POST only."""
    return {"message": "请使用 POST /api/skills/import-git 导入 Skill"}


@router.post("/import-git")
async def import_skill_from_git(request: GitImportRequest):
    tmp = None
    try:
        imported = skill_storage.import_from_git(request.git_url)
        tmp = imported["path"]
        source_metadata = imported["parsed"]["metadata"]
        body = imported["parsed"]["body"]
        name = source_metadata.get("name") or Path(tmp).name
        trigger = request.trigger_condition or source_metadata.get("trigger_condition", "")
        metadata = canonical_metadata({"trigger_condition": trigger}, source_metadata)
        metadata["version"] = str(metadata.get("version") or "1.0.0")
        skill = db_create_skill(
            name=name, description=source_metadata.get("description", ""), instructions=body,
            category="导入", enabled=request.enabled, trigger_condition=trigger,
            skill_metadata=metadata, model_id=source_metadata.get("model_id"),
        )
        skill_storage.copy_skill_files(tmp, skill["id"])
        _persist_current_skill_file(skill)
        create_skill_version(skill["id"], metadata["version"], "从 Git 导入")
        return {"skill": _serialize_skill(db_get_skill(skill["id"]))}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"import failed: {exc}")
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


@router.post("/import-zip")
async def import_skill_from_zip(
    file: UploadFile = File(...),
    trigger_condition: str = Form(""),
    enabled: bool = Form(True),
):
    tmp = None
    try:
        imported = skill_storage.import_from_zip(await file.read())
        tmp = imported["path"]
        source_metadata = imported["parsed"]["metadata"]
        body = imported["parsed"]["body"]
        name = source_metadata.get("name") or (file.filename or "imported-skill").replace(".zip", "")
        trigger = trigger_condition or source_metadata.get("trigger_condition", "")
        metadata = canonical_metadata({"trigger_condition": trigger}, source_metadata)
        metadata["version"] = str(metadata.get("version") or "1.0.0")
        skill = db_create_skill(
            name=name, description=source_metadata.get("description", ""), instructions=body,
            category="导入", enabled=enabled, trigger_condition=trigger,
            skill_metadata=metadata, model_id=source_metadata.get("model_id"),
        )
        skill_storage.copy_skill_files(tmp, skill["id"])
        _persist_current_skill_file(skill)
        create_skill_version(skill["id"], metadata["version"], "从 Zip 导入")
        return {"skill": _serialize_skill(db_get_skill(skill["id"]))}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"import failed: {exc}")
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


@router.get("")
async def list_skills():
    skills = [_serialize_skill(skill) for skill in db_list_skills()]
    return {"skills": skills, "count": len(skills)}


@router.post("")
async def create_skill(request: SkillCreate):
    if not request.name.strip():
        raise HTTPException(status_code=400, detail="skill name required")
    metadata = canonical_metadata({"trigger_condition": request.trigger_condition}, request.skill_metadata)
    metadata["version"] = "1.0.0"
    skill = db_create_skill(
        name=request.name.strip(), description=request.description.strip(), instructions=request.instructions.strip(),
        category=request.category.strip() or "未分类", enabled=request.enabled,
        trigger_condition=request.trigger_condition.strip(), skill_metadata=metadata, model_id=request.model_id,
    )
    _persist_current_skill_file(skill)
    create_skill_version(skill["id"], metadata["version"], "创建 Skill")
    return {"skill": _serialize_skill(db_get_skill(skill["id"]))}


@router.get("/{skill_id}")
async def get_skill(skill_id: int):
    skill = _get_or_404(skill_id)
    item = _serialize_skill(skill)
    item["skill_md"] = skill_storage.read_skill_md(skill_id)
    item["files"] = skill_storage.list_skill_files(skill_id)
    item["versions"] = list_skill_versions(skill_id)
    return {"skill": item}


@router.put("/{skill_id}")
async def update_skill(skill_id: int, request: SkillUpdate):
    existing = _get_or_404(skill_id)
    updated = _save_update(existing, request, "平台管理页编辑")
    return {"skill": _serialize_skill(updated)}


@router.delete("/{skill_id}")
async def delete_skill(skill_id: int):
    _get_or_404(skill_id)
    deleted = db_delete_skill(skill_id)
    skill_storage.delete_skill_dir(skill_id)
    return {"ok": True, "deleted_id": skill_id, "deleted": deleted}


@router.post("/{skill_id}/skill-md")
async def update_skill_md(skill_id: int, request: SkillMdUpdate):
    existing = _get_or_404(skill_id)
    metadata = canonical_metadata(existing, request.metadata)
    metadata["version"] = next_patch_version(str(metadata.get("version") or "legacy-1.0.0"))
    trigger = ",".join(metadata.get("positive_triggers") or [])
    updated = db_update_skill(
        skill_id=skill_id,
        name=str(metadata.get("name") or existing["name"]),
        description=str(metadata.get("description") or existing.get("description") or ""),
        instructions=request.body,
        category=existing.get("category") or "未分类",
        enabled=bool(existing.get("enabled")),
        trigger_condition=trigger,
        skill_metadata=metadata,
        storage_path=existing.get("storage_path") or "",
        model_id=existing.get("model_id"),
    )
    _persist_current_skill_file(updated)
    create_skill_version(skill_id, metadata["version"], "编辑 SKILL.md")
    return {"skill": _serialize_skill(updated)}


@router.get("/{skill_id}/versions")
async def get_skill_versions(skill_id: int):
    _get_or_404(skill_id)
    return {"versions": list_skill_versions(skill_id)}


@router.post("/{skill_id}/versions/{version}/rollback")
async def rollback_skill_version(skill_id: int, version: str):
    existing = _get_or_404(skill_id)
    records = list_skill_versions(skill_id)
    record = next((item for item in records if item.get("version") == version), None)
    if not record:
        raise HTTPException(status_code=404, detail="skill version not found")
    # Fetch the full snapshot because the list endpoint deliberately contains
    # only a summary to keep management responses small.
    from db.property_db import get_skill_version
    detail = get_skill_version(skill_id, version) or {}
    snapshot = detail.get("snapshot") or {}
    if not snapshot:
        raise HTTPException(status_code=409, detail="skill version snapshot unavailable")
    metadata = canonical_metadata(snapshot, snapshot.get("skill_metadata"))
    metadata["version"] = next_patch_version(str((existing.get("skill_metadata") or {}).get("version") or "legacy-1.0.0"))
    updated = db_update_skill(
        skill_id=skill_id, name=snapshot.get("name") or existing["name"],
        description=snapshot.get("description") or "", instructions=snapshot.get("instructions") or "",
        category=snapshot.get("category") or "未分类", enabled=bool(snapshot.get("enabled")),
        trigger_condition=snapshot.get("trigger_condition") or "", skill_metadata=metadata,
        storage_path=snapshot.get("storage_path") or "", model_id=snapshot.get("model_id"),
    )
    _persist_current_skill_file(updated)
    create_skill_version(skill_id, metadata["version"], f"从 {version} 回滚")
    return {"skill": _serialize_skill(updated), "rolled_back_from": version}


@router.post("/{skill_id}/evaluate")
async def evaluate_skill(skill_id: int, request: TestSkillRequest):
    skill = _get_or_404(skill_id)
    return _evaluate_for_agent(skill, request.message, request.agent_id)


@router.post("/{skill_id}/test")
async def test_skill(skill_id: int, request: TestSkillRequest):
    skill = _get_or_404(skill_id)
    result = _evaluate_for_agent(skill, request.message, request.agent_id)
    if result["will_inject"]:
        result["instructions_preview"] = skill_storage.build_instructions(skill_id, skill)[:1200]
    else:
        result["instructions_preview"] = "未注入：请依据上方绑定、触发、负向条件或冲突规则修正配置。"
    result["note"] = "这是确定性的运行时诊断，不调用大模型，也不消耗 Token。"
    return result


@router.get("/{skill_id}/files")
async def list_skill_files(skill_id: int):
    _get_or_404(skill_id)
    return {"files": skill_storage.list_skill_files(skill_id)}


@router.post("/{skill_id}/files")
async def upload_skill_file(skill_id: int, file: UploadFile = File(...), path: str = Form("")):
    _get_or_404(skill_id)
    rel_path = path or file.filename
    if not rel_path:
        raise HTTPException(status_code=400, detail="path required")
    skill_storage.save_skill_file(skill_id, rel_path, await file.read())
    return {"ok": True, "path": rel_path}


@router.delete("/{skill_id}/files/{file_path:path}")
async def delete_skill_file(skill_id: int, file_path: str):
    _get_or_404(skill_id)
    skill_storage.delete_skill_file(skill_id, file_path)
    return {"ok": True}


@router.get("/{skill_id}/export")
async def export_skill(skill_id: int):
    skill = _get_or_404(skill_id)
    zip_bytes = skill_storage.export_to_zip(skill_id)
    filename = skill["name"] or f"skill-{skill_id}"
    safe_name = filename.encode("ascii", "ignore").decode() or f"skill-{skill_id}"
    encoded_name = urllib.parse.quote(filename, safe="")
    return StreamingResponse(
        io.BytesIO(zip_bytes), media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=\"{safe_name}.zip\"; filename*=UTF-8''{encoded_name}.zip"},
    )


@router.post("/{skill_id}/apply-darwin")
async def apply_darwin_optimization(skill_id: int, request: ApplyDarwinRequest):
    existing = _get_or_404(skill_id)
    payload = SkillUpdate(
        name=existing["name"], description=existing.get("description") or "",
        instructions=request.suggested_prompt.strip(), category=existing.get("category") or "未分类",
        enabled=bool(existing.get("enabled")), trigger_condition=existing.get("trigger_condition") or "",
        skill_metadata=existing.get("skill_metadata") or {}, model_id=existing.get("model_id"),
    )
    updated = _save_update(existing, payload, "应用 Darwin 建议（待人工回归确认）")
    return {"skill": _serialize_skill(updated), "applied": True}
