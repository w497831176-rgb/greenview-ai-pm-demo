"""
Skill Management API
====================

REST endpoints for platform skill metadata CRUD and Agent Skills import.
"""

import io
import shutil
import tempfile
import urllib.parse
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from db.property_db import (
    create_skill as db_create_skill,
    delete_skill as db_delete_skill,
    get_skill as db_get_skill,
    list_skills as db_list_skills,
    update_skill as db_update_skill,
)
import skill_storage

router = APIRouter(prefix="/api/skills", tags=["skills"])


class SkillCreate(BaseModel):
    name: str
    description: str = ""
    instructions: str = ""
    category: str = "未分类"
    enabled: bool = True
    trigger_condition: str = ""
    model_id: Optional[str] = None


class SkillUpdate(BaseModel):
    name: str
    description: str = ""
    instructions: str = ""
    category: str = "未分类"
    enabled: bool = True
    trigger_condition: str = ""
    model_id: Optional[str] = None


class SkillMdUpdate(BaseModel):
    metadata: dict
    body: str


class GitImportRequest(BaseModel):
    git_url: str
    trigger_condition: str = ""
    enabled: bool = True


class TestSkillRequest(BaseModel):
    message: str


class ApplyDarwinRequest(BaseModel):
    suggested_prompt: str


@router.get("")
async def list_skills():
    """List all skills."""
    skills = db_list_skills()
    for s in skills:
        s["has_files"] = skill_storage._skill_dir(s["id"]).exists()
    return {"skills": skills, "count": len(skills)}


@router.get("/{skill_id}")
async def get_skill(skill_id: int):
    """Get a single skill including SKILL.md content and file list."""
    skill = db_get_skill(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail="not found")
    skill["skill_md"] = skill_storage.read_skill_md(skill_id)
    skill["files"] = skill_storage.list_skill_files(skill_id)
    skill["has_files"] = skill_storage._skill_dir(skill_id).exists()
    return {"skill": skill}


@router.post("")
async def create_skill(request: SkillCreate):
    """Create a new skill."""
    skill = db_create_skill(
        name=request.name,
        description=request.description,
        instructions=request.instructions,
        category=request.category,
        enabled=request.enabled,
        trigger_condition=request.trigger_condition,
        model_id=request.model_id,
    )
    # Create empty skill directory with a default SKILL.md.
    skill_storage.ensure_skill_dir(skill["id"])
    skill_storage.write_skill_md(
        skill["id"],
        metadata={
            "name": request.name,
            "description": request.description,
            "version": "1.0.0",
        },
        body=request.instructions or "请在此处填写 Skill 指令。",
    )
    return {"skill": db_get_skill(skill["id"])}


@router.put("/{skill_id}")
async def update_skill(skill_id: int, request: SkillUpdate):
    """Update a skill."""
    skill = db_get_skill(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail="not found")
    skill = db_update_skill(
        skill_id=skill_id,
        name=request.name,
        description=request.description,
        instructions=request.instructions,
        category=request.category,
        enabled=request.enabled,
        trigger_condition=request.trigger_condition,
        model_id=request.model_id,
    )
    # Sync SKILL.md metadata.
    parsed = skill_storage.read_skill_md(skill_id) or {"metadata": {}, "body": ""}
    metadata = parsed.get("metadata", {})
    metadata["name"] = request.name
    metadata["description"] = request.description
    if request.model_id is not None:
        metadata["model_id"] = request.model_id
    skill_storage.write_skill_md(skill_id, metadata, parsed.get("body", request.instructions))
    return {"skill": skill}


@router.delete("/{skill_id}")
async def delete_skill(skill_id: int):
    """Delete a skill and its storage."""
    deleted = db_delete_skill(skill_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="not found")
    skill_storage.delete_skill_dir(skill_id)
    return {"ok": True, "deleted_id": skill_id}


@router.post("/import-git")
async def import_skill_from_git(request: GitImportRequest):
    """Import a skill from a Git URL."""
    tmp = None
    try:
        imported = skill_storage.import_from_git(request.git_url)
        tmp = imported["path"]
        metadata = imported["parsed"]["metadata"]
        body = imported["parsed"]["body"]
        name = metadata.get("name") or Path(tmp).name
        description = metadata.get("description", "")
        trigger = request.trigger_condition or metadata.get("trigger_condition", "")

        skill = db_create_skill(
            name=name,
            description=description,
            instructions=body,
            category="导入",
            enabled=request.enabled,
            trigger_condition=trigger,
            skill_metadata=metadata,
            model_id=metadata.get("model_id"),
        )
        skill_storage.copy_skill_files(tmp, skill["id"])
        return {"skill": db_get_skill(skill["id"])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"import failed: {e}")
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


@router.post("/import-zip")
async def import_skill_from_zip(
    file: UploadFile = File(...),
    trigger_condition: str = Form(""),
    enabled: bool = Form(True),
):
    """Import a skill from a zip upload."""
    tmp = None
    try:
        contents = await file.read()
        imported = skill_storage.import_from_zip(contents)
        tmp = imported["path"]
        metadata = imported["parsed"]["metadata"]
        body = imported["parsed"]["body"]
        name = metadata.get("name") or (file.filename or "imported-skill").replace(".zip", "")
        description = metadata.get("description", "")
        trigger = trigger_condition or metadata.get("trigger_condition", "")

        skill = db_create_skill(
            name=name,
            description=description,
            instructions=body,
            category="导入",
            enabled=enabled,
            trigger_condition=trigger,
            skill_metadata=metadata,
            model_id=metadata.get("model_id"),
        )
        skill_storage.copy_skill_files(tmp, skill["id"])
        return {"skill": db_get_skill(skill["id"])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"import failed: {e}")
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


@router.post("/{skill_id}/skill-md")
async def update_skill_md(skill_id: int, request: SkillMdUpdate):
    """Update SKILL.md metadata and body."""
    skill = db_get_skill(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail="not found")
    skill_storage.write_skill_md(skill_id, request.metadata, request.body)
    # Sync DB fields from metadata.
    name = request.metadata.get("name", skill["name"])
    description = request.metadata.get("description", skill.get("description", ""))
    db_update_skill(
        skill_id=skill_id,
        name=name,
        description=description,
        instructions=skill["instructions"],
        category=skill.get("category", "未分类"),
        enabled=skill.get("enabled", True),
        trigger_condition=skill.get("trigger_condition", ""),
        skill_metadata=request.metadata,
        model_id=request.metadata.get("model_id"),
    )
    return {"skill": db_get_skill(skill_id)}


@router.get("/{skill_id}/files")
async def list_skill_files(skill_id: int):
    """List files in a skill directory."""
    skill = db_get_skill(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail="not found")
    return {"files": skill_storage.list_skill_files(skill_id)}


@router.post("/{skill_id}/files")
async def upload_skill_file(skill_id: int, file: UploadFile = File(...), path: str = Form("")):
    """Upload a file into the skill directory."""
    skill = db_get_skill(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail="not found")
    rel_path = path or file.filename
    if not rel_path:
        raise HTTPException(status_code=400, detail="path required")
    contents = await file.read()
    skill_storage.save_skill_file(skill_id, rel_path, contents)
    return {"ok": True, "path": rel_path}


@router.delete("/{skill_id}/files/{file_path:path}")
async def delete_skill_file(skill_id: int, file_path: str):
    """Delete a file from the skill directory."""
    skill = db_get_skill(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail="not found")
    skill_storage.delete_skill_file(skill_id, file_path)
    return {"ok": True}


@router.post("/{skill_id}/test")
async def test_skill(skill_id: int, request: TestSkillRequest):
    """Test a skill with a single message."""
    skill = db_get_skill(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail="not found")
    if not skill.get("enabled"):
        raise HTTPException(status_code=400, detail="skill is disabled")

    instructions = skill_storage.build_instructions(skill_id, skill)
    # Simple direct LLM call would require model setup; for demo, return instructions preview.
    # In a real implementation, invoke the property agent with the skill instructions.
    return {
        "skill_id": skill_id,
        "message": request.message,
        "instructions_preview": instructions[:500],
        "note": "V1.1 测试接口返回注入的指令预览；完整对话测试建议通过业主 AI 助手进行。",
    }


@router.get("/{skill_id}/export")
async def export_skill(skill_id: int):
    """Export a skill as a zip file."""
    skill = db_get_skill(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail="not found")
    from fastapi.responses import StreamingResponse

    zip_bytes = skill_storage.export_to_zip(skill_id)
    # Encode filename per RFC 5987 to support non-ASCII skill names in headers.
    filename = skill['name'] or f"skill-{skill_id}"
    safe_name = filename.encode("ascii", "ignore").decode() or f"skill-{skill_id}"
    encoded_name = urllib.parse.quote(filename, safe="")
    content_disposition = f"attachment; filename=\"{safe_name}.zip\"; filename*=UTF-8''{encoded_name}.zip"
    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": content_disposition},
    )


@router.post("/{skill_id}/apply-darwin")
async def apply_darwin_optimization(skill_id: int, request: ApplyDarwinRequest):
    """Apply a Darwin-suggested prompt optimization to a skill."""
    skill = db_get_skill(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail="not found")

    new_instructions = request.suggested_prompt.strip()
    updated = db_update_skill(
        skill_id=skill_id,
        name=skill["name"],
        description=skill.get("description", ""),
        instructions=new_instructions,
        category=skill.get("category", "未分类"),
        enabled=skill.get("enabled", True),
        trigger_condition=skill.get("trigger_condition", ""),
        model_id=skill.get("model_id"),
    )
    # Sync SKILL.md body with the new instructions.
    parsed = skill_storage.read_skill_md(skill_id) or {"metadata": {}, "body": ""}
    skill_storage.write_skill_md(skill_id, parsed.get("metadata", {}), new_instructions)
    return {"skill": updated, "applied": True}
