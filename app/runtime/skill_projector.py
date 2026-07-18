"""Project published DB Skills into safe Agno Skill packages.

V1.8 exposes only SKILL.md and references.  Scripts are deliberately omitted,
so Agno progressive discovery remains available without granting code
execution to a dynamically configured Skill.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.runtime.contracts import SkillActivation, content_hash


DATA_DIR = Path(os.getenv("PROPERTY_DATA_DIR", "/app/data"))
MAX_SKILL_CHARS = 6000
MAX_TOTAL_SKILL_CHARS = 12000


def _package_name(skill_id: int) -> str:
    return f"skill-{skill_id}"


def _safe_description(value: str) -> str:
    return json.dumps((value or "YIAI runtime skill")[:1024], ensure_ascii=False)


def project_skills(
    release_id: str,
    skills: Iterable[Dict[str, Any]],
    match_reasons: Optional[Dict[int, str]] = None,
) -> Tuple[Optional[Path], List[SkillActivation]]:
    root = DATA_DIR / "runtime-releases" / release_id / "skills"
    root.mkdir(parents=True, exist_ok=True)
    total_chars = 0
    activations: List[SkillActivation] = []
    for skill in skills:
        if not skill.get("enabled"):
            continue
        skill_id = int(skill["skill_id"])
        instructions = str(skill.get("instructions_fallback") or "")
        remaining = max(0, MAX_TOTAL_SKILL_CHARS - total_chars)
        body = instructions[: min(MAX_SKILL_CHARS, remaining)]
        if not body:
            continue
        total_chars += len(body)
        package_name = _package_name(skill_id)
        package_dir = root / package_name
        package_dir.mkdir(parents=True, exist_ok=True)
        skill_md = (
            "---\n"
            f"name: {package_name}\n"
            f"description: {_safe_description(str(skill.get('description') or skill.get('name') or package_name))}\n"
            "---\n\n"
            f"# {skill.get('name') or package_name}\n\n"
            f"{body}\n"
        )
        target = package_dir / "SKILL.md"
        tmp = package_dir / ".SKILL.md.tmp"
        tmp.write_text(skill_md, encoding="utf-8")
        tmp.replace(target)

        # Materialize immutable reference snapshots from RuntimeRelease. Never
        # read mutable live Skill storage and never copy a scripts directory.
        copied_refs: List[str] = []
        snapshots = skill.get("reference_snapshots") or []
        if snapshots:
            destination = package_dir / "references"
            destination.mkdir(parents=True, exist_ok=True)
            for reference in snapshots:
                relative = Path(str(reference.get("path") or ""))
                if (
                    not relative.parts
                    or relative.is_absolute()
                    or ".." in relative.parts
                ):
                    continue
                out = destination / relative
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(
                    str(reference.get("content") or ""),
                    encoding="utf-8",
                )
                copied_refs.append(relative.as_posix())

        activations.append(
            SkillActivation(
                skill_id=skill_id,
                version=str(skill.get("version") or "legacy-1.0.0"),
                content_hash=str(skill.get("content_hash") or content_hash(body)),
                name=str(skill.get("name") or package_name),
                match_reason=(match_reasons or {}).get(skill_id, "published binding"),
                loaded_resources=["SKILL.md", *[f"references/{item}" for item in copied_refs]],
            )
        )
        if total_chars >= MAX_TOTAL_SKILL_CHARS:
            break
    return (root if activations else None), activations
