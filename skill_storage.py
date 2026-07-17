"""
Skill file storage following the Agent Skills standard.

Imported / created skills live under /app/data/skills/{skill_id}/.
Each skill directory contains:
    SKILL.md
    scripts/      (static context, not executed in V1.1)
    references/
    assets/
"""

import io
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

SKILLS_BASE_DIR = Path(os.getenv("PROPERTY_DATA_DIR", "/app/data")) / "skills"
SKILLS_BASE_DIR.mkdir(parents=True, exist_ok=True)


def _skill_dir(skill_id: int) -> Path:
    return SKILLS_BASE_DIR / str(skill_id)


def ensure_skill_dir(skill_id: int) -> Path:
    d = _skill_dir(skill_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "scripts").mkdir(exist_ok=True)
    (d / "references").mkdir(exist_ok=True)
    (d / "assets").mkdir(exist_ok=True)
    return d


def parse_skill_md(text: str) -> Dict[str, Any]:
    """Parse SKILL.md with YAML frontmatter + markdown body."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            frontmatter = yaml.safe_load(parts[1]) or {}
            body = parts[2].strip()
            return {"metadata": frontmatter, "body": body}
    # No frontmatter: treat first # title as name and rest as body.
    title = ""
    body = text
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if m:
        title = m.group(1).strip()
    return {"metadata": {"name": title}, "body": body}


def read_skill_md(skill_id: int) -> Optional[Dict[str, Any]]:
    skill_md = _skill_dir(skill_id) / "SKILL.md"
    if not skill_md.exists():
        return None
    return parse_skill_md(skill_md.read_text(encoding="utf-8"))


def write_skill_md(skill_id: int, metadata: Dict[str, Any], body: str):
    ensure_skill_dir(skill_id)
    skill_md = _skill_dir(skill_id) / "SKILL.md"
    yaml_text = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False)
    skill_md.write_text(f"---\n{yaml_text}---\n\n{body}\n", encoding="utf-8")


def write_skill_revision(skill_id: int, version: str, metadata: Dict[str, Any], body: str):
    """Keep a file-level snapshot alongside the database audit record.

    Runtime always reads the root ``SKILL.md``.  Revision files are immutable
    operator evidence and are never auto-injected into a chat context.
    """
    revision_dir = ensure_skill_dir(skill_id) / "revisions" / str(version)
    revision_dir.mkdir(parents=True, exist_ok=True)
    yaml_text = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False)
    (revision_dir / "SKILL.md").write_text(
        f"---\n{yaml_text}---\n\n{body}\n", encoding="utf-8"
    )


def list_skill_files(skill_id: int) -> List[Dict[str, str]]:
    d = _skill_dir(skill_id)
    if not d.exists():
        return []
    files = []
    for root, _dirs, fs in os.walk(d):
        for f in fs:
            full = Path(root) / f
            rel = full.relative_to(d).as_posix()
            files.append({"path": rel, "size": full.stat().st_size})
    return files


def save_skill_file(skill_id: int, rel_path: str, content: bytes):
    ensure_skill_dir(skill_id)
    target = _skill_dir(skill_id) / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def delete_skill_file(skill_id: int, rel_path: str):
    target = _skill_dir(skill_id) / rel_path
    if target.exists():
        target.unlink()


def delete_skill_dir(skill_id: int):
    d = _skill_dir(skill_id)
    if d.exists():
        shutil.rmtree(d)


def import_from_git(git_url: str, temp_clone_dir: Optional[str] = None) -> Dict[str, Any]:
    """Clone a Git repository and return parsed SKILL.md + directory path."""
    try:
        import git
    except Exception as exc:
        raise RuntimeError("Git 命令不可用，请在容器中安装 git 或改用 zip 导入") from exc
    tmp = tempfile.mkdtemp(dir=temp_clone_dir)
    git.Repo.clone_from(git_url, tmp, depth=1)
    skill_md_path = Path(tmp) / "SKILL.md"
    if not skill_md_path.exists():
        shutil.rmtree(tmp)
        raise ValueError("Repository missing SKILL.md")
    parsed = parse_skill_md(skill_md_path.read_text(encoding="utf-8"))
    return {"path": tmp, "parsed": parsed}


def import_from_zip(zip_bytes: bytes, temp_extract_dir: Optional[str] = None) -> Dict[str, Any]:
    """Extract a zip archive and return parsed SKILL.md + directory path."""
    tmp = tempfile.mkdtemp(dir=temp_extract_dir)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        z.extractall(tmp)
    # Handle nested root folder.
    entries = [e for e in Path(tmp).iterdir() if not e.name.startswith("__")]
    root = Path(tmp)
    if len(entries) == 1 and entries[0].is_dir() and (entries[0] / "SKILL.md").exists():
        root = entries[0]

    skill_md_path = root / "SKILL.md"
    if not skill_md_path.exists():
        shutil.rmtree(tmp)
        raise ValueError("Archive missing SKILL.md")
    parsed = parse_skill_md(skill_md_path.read_text(encoding="utf-8"))
    return {"path": str(root), "parsed": parsed}


def copy_skill_files(source_dir: str, skill_id: int):
    """Copy imported files into the skill storage directory."""
    src = Path(source_dir)
    dst = ensure_skill_dir(skill_id)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def export_to_zip(skill_id: int) -> bytes:
    """Export a skill directory as zip bytes."""
    d = _skill_dir(skill_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(d):
            for f in files:
                full = Path(root) / f
                rel = full.relative_to(d).as_posix()
                zf.write(full, rel)
    return buf.getvalue()


def build_instructions(skill_id: int, skill_record: Dict[str, Any]) -> str:
    """Build the final instructions injected into the Agent context."""
    parts = []
    # 1. SKILL.md body
    parsed = read_skill_md(skill_id)
    if parsed:
        parts.append(parsed["body"])
    # 2. References as additional context
    ref_dir = _skill_dir(skill_id) / "references"
    if ref_dir.exists():
        refs = []
        for f in sorted(ref_dir.rglob("*.md")):
            refs.append(f"## {f.relative_to(ref_dir)}\n\n{f.read_text(encoding='utf-8')}")
        if refs:
            parts.append("\n\n## 参考资料\n\n" + "\n\n".join(refs))
    # 3. Legacy instructions from DB (for backward compatibility)
    legacy = skill_record.get("instructions") or ""
    if legacy and legacy not in "\n\n".join(parts):
        parts.append(legacy)
    return "\n\n".join(parts)
