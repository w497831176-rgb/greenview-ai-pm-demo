"""Prepare public Git MCP repositories for the YIAI stdio runtime.

This module belongs to the control plane.  It does not publish a
RuntimeRelease and it never grants a discovered tool permission.  Its only
jobs are to clone a public repository into the persistent demo-data volume,
detect a Python or Node entrypoint, install the package locally and return a
stdio launch specification that the normal MCP discovery path can verify.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse


DEFAULT_PACKAGE_ROOT = "/app/data/mcp_packages"
IGNORED_DIRECTORIES = {
    ".git",
    ".yiai-venv",
    "__pycache__",
    "node_modules",
    "tests",
    "test",
    "examples",
    "example",
}


class McpImportError(RuntimeError):
    """A user-facing import failure with a stable machine code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        detail: str = "",
        steps: Optional[List[Dict[str, str]]] = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail
        self.steps = list(steps or [])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "detail": self.detail,
            "steps": self.steps,
        }


@dataclass
class PreparedMcpPackage:
    source_url: str
    source_type: str
    name: str
    runtime_type: str
    package_path: str
    command: str
    args: List[str]
    detected_entrypoint: str
    required_env_names: List[str] = field(default_factory=list)
    steps: List[Dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _step(steps: List[Dict[str, str]], name: str, status: str, detail: str) -> None:
    steps.append({"name": name, "status": status, "detail": detail})


def _tail(text: str, limit: int = 1200) -> str:
    normalized = (text or "").strip()
    return normalized[-limit:]


def _run(
    command: List[str],
    *,
    cwd: Optional[Path] = None,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=dict(os.environ),
        )
    except FileNotFoundError as exc:
        raise McpImportError(
            "runtime_command_missing",
            f"运行环境缺少命令：{command[0]}",
            detail=str(exc),
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise McpImportError(
            "package_prepare_timeout",
            "仓库准备超时，请换用依赖更轻的 MCP 仓库",
            detail=_tail((exc.stderr or "") if isinstance(exc.stderr, str) else ""),
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = _tail(exc.stderr or exc.stdout or "")
        raise McpImportError(
            "package_prepare_failed",
            "仓库依赖安装或构建失败",
            detail=detail or f"exit_code={exc.returncode}",
        ) from exc


def _clone_git_repository(source_url: str, destination: Path) -> None:
    """Prefer a shallow clone, with a full-clone fallback for dumb HTTP."""

    try:
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--single-branch",
                "--",
                source_url,
                str(destination),
            ],
            timeout=120,
        )
    except McpImportError as exc:
        if "dumb http transport does not support shallow" not in exc.detail.casefold():
            raise
        _safe_remove(destination, destination.parent)
        _run(
            ["git", "clone", "--", source_url, str(destination)],
            timeout=120,
        )


def _validate_git_url(git_url: str) -> str:
    value = (git_url or "").strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise McpImportError(
            "invalid_git_url",
            "请填写公开的 HTTP/HTTPS Git 仓库地址",
        )
    if parsed.username or parsed.password:
        raise McpImportError(
            "credential_in_git_url",
            "Git 地址中不能包含账号、Token 或密码",
        )
    return value


def _slug(value: str) -> str:
    candidate = re.sub(r"\.git$", "", value.rstrip("/").split("/")[-1])
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate).strip("-._")
    return (candidate or "mcp-package")[:60]


def _safe_remove(path: Path, root: Path) -> None:
    try:
        resolved_path = path.resolve()
        resolved_root = root.resolve()
        resolved_path.relative_to(resolved_root)
    except (OSError, ValueError):
        return
    if resolved_path != resolved_root and resolved_path.exists():
        shutil.rmtree(resolved_path)


def _walk_files(root: Path, names: Iterable[str]) -> List[Path]:
    accepted = set(names)
    matches: List[Path] = []
    for current, directories, files in os.walk(root):
        relative = Path(current).relative_to(root)
        if len(relative.parts) > 3:
            directories[:] = []
            continue
        directories[:] = [
            item for item in directories if item not in IGNORED_DIRECTORIES
        ]
        for filename in files:
            if filename in accepted:
                matches.append(Path(current) / filename)
    return matches


def _candidate_roots(repo_root: Path) -> List[Path]:
    marker_files = {"yiai-mcp.json", "pyproject.toml", "setup.py", "package.json"}
    candidates = [repo_root]
    for marker in _walk_files(repo_root, marker_files):
        parent = marker.parent
        if parent not in candidates:
            candidates.append(parent)
    return candidates


def _manifest(root: Path) -> Dict[str, Any]:
    path = root / "yiai-mcp.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise McpImportError(
            "invalid_yiai_manifest",
            "仓库中的 yiai-mcp.json 不是合法 JSON",
            detail=str(exc),
        ) from exc
    if not isinstance(payload, dict):
        raise McpImportError(
            "invalid_yiai_manifest",
            "yiai-mcp.json 必须是 JSON 对象",
        )
    return payload


def _runtime_for_root(root: Path, requested: str) -> Optional[str]:
    manifest = _manifest(root)
    declared = str(manifest.get("runtime") or "").strip().lower()
    if requested in {"python", "node"}:
        return requested if (
            (requested == "python" and (
                (root / "pyproject.toml").exists()
                or (root / "setup.py").exists()
                or any(root.glob("*.py"))
                or declared == "python"
            ))
            or (requested == "node" and (
                (root / "package.json").exists() or declared == "node"
            ))
        ) else None
    if declared in {"python", "node"}:
        return declared
    if (root / "pyproject.toml").exists() or (root / "setup.py").exists():
        return "python"
    if (root / "package.json").exists():
        return "node"
    if any(root.glob("*.py")):
        return "python"
    return None


def _select_project_root(repo_root: Path, requested: str) -> tuple[Path, str]:
    scored: List[tuple[int, Path, str]] = []
    for root in _candidate_roots(repo_root):
        runtime = _runtime_for_root(root, requested)
        if not runtime:
            continue
        manifest = _manifest(root)
        score = 100 if manifest else 0
        score += 30 if "mcp" in root.name.casefold() else 0
        score += 20 if runtime == "python" and (root / "pyproject.toml").exists() else 0
        score += 20 if runtime == "node" and (root / "package.json").exists() else 0
        score -= len(root.relative_to(repo_root).parts)
        scored.append((score, root, runtime))
    if not scored:
        raise McpImportError(
            "runtime_not_detected",
            "没有识别到 Python 或 Node MCP 项目",
            detail="仓库需要包含 pyproject.toml、setup.py、package.json 或 yiai-mcp.json。",
        )
    _, root, runtime = sorted(scored, key=lambda item: item[0], reverse=True)[0]
    return root, runtime


def _python_entrypoint(project_root: Path) -> tuple[str, str]:
    manifest = _manifest(project_root)
    manifest_entry = str(manifest.get("entrypoint") or "").strip()
    if manifest_entry:
        path = (project_root / manifest_entry).resolve()
        if not path.exists():
            raise McpImportError(
                "entrypoint_not_found",
                "yiai-mcp.json 指定的 Python 入口不存在",
                detail=manifest_entry,
            )
        return "file", str(path)

    pyproject = project_root / "pyproject.toml"
    if pyproject.exists():
        try:
            payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise McpImportError(
                "invalid_pyproject",
                "pyproject.toml 无法解析",
                detail=str(exc),
            ) from exc
        scripts = ((payload.get("project") or {}).get("scripts") or {})
        if isinstance(scripts, dict) and scripts:
            preferred = sorted(
                scripts,
                key=lambda item: (
                    0 if any(key in item.casefold() for key in ("mcp", "server")) else 1,
                    item,
                ),
            )[0]
            return "console_script", preferred

    candidates: List[tuple[int, Path]] = []
    for current, directories, files in os.walk(project_root):
        relative = Path(current).relative_to(project_root)
        if len(relative.parts) > 3:
            directories[:] = []
            continue
        directories[:] = [
            item for item in directories if item not in IGNORED_DIRECTORIES
        ]
        for filename in files:
            if not filename.endswith(".py"):
                continue
            path = Path(current) / filename
            try:
                source = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            signals = sum(
                token in source
                for token in ("FastMCP", "mcp.run(", "stdio_server", "Server(")
            )
            if not signals:
                continue
            name_score = 20 if "mcp" in filename.casefold() else 0
            name_score += 10 if "server" in filename.casefold() else 0
            candidates.append((signals * 20 + name_score - len(relative.parts), path))
    if not candidates:
        raise McpImportError(
            "entrypoint_not_detected",
            "识别到 Python 项目，但没有找到 MCP stdio 启动入口",
            detail="可在仓库根目录增加 yiai-mcp.json 并填写 entrypoint。",
        )
    return "file", str(sorted(candidates, key=lambda item: item[0], reverse=True)[0][1].resolve())


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _venv_script(venv: Path, name: str) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / f"{name}.exe"
    return venv / "bin" / name


def _prepare_python(project_root: Path, steps: List[Dict[str, str]]) -> tuple[str, List[str], str]:
    entry_kind, entrypoint = _python_entrypoint(project_root)
    venv = project_root / ".yiai-venv"
    _run(["uv", "venv", "--system-site-packages", str(venv)], cwd=project_root)
    python = _venv_python(venv)
    _step(steps, "prepare_runtime", "passed", "已创建隔离的 Python 运行环境")

    if (project_root / "pyproject.toml").exists() or (project_root / "setup.py").exists():
        _run(
            ["uv", "pip", "install", "--python", str(python), "-e", str(project_root)],
            cwd=project_root,
        )
        _step(steps, "install_dependencies", "passed", "已安装 Python 项目及依赖")
    elif (project_root / "requirements.txt").exists():
        _run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "-r",
                str(project_root / "requirements.txt"),
            ],
            cwd=project_root,
        )
        _step(steps, "install_dependencies", "passed", "已安装 requirements.txt")
    else:
        _step(
            steps,
            "install_dependencies",
            "passed",
            "仓库没有额外依赖清单，复用系统 MCP 运行库",
        )

    if entry_kind == "console_script":
        script = _venv_script(venv, entrypoint)
        if not script.exists():
            raise McpImportError(
                "console_script_missing",
                "Python 包已安装，但启动脚本没有生成",
                detail=entrypoint,
                steps=steps,
            )
        return str(script), [], entrypoint
    return str(python), [entrypoint], entrypoint


def _read_package_json(project_root: Path) -> Dict[str, Any]:
    try:
        payload = json.loads((project_root / "package.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise McpImportError(
            "invalid_package_json",
            "package.json 无法解析",
            detail=str(exc),
        ) from exc
    if not isinstance(payload, dict):
        raise McpImportError("invalid_package_json", "package.json 必须是 JSON 对象")
    return payload


def _node_entrypoint(project_root: Path, package: Dict[str, Any]) -> tuple[str, List[str], str]:
    manifest = _manifest(project_root)
    manifest_entry = str(manifest.get("entrypoint") or "").strip()
    if manifest_entry:
        path = (project_root / manifest_entry).resolve()
        if not path.exists():
            raise McpImportError(
                "entrypoint_not_found",
                "yiai-mcp.json 指定的 Node 入口不存在",
                detail=manifest_entry,
            )
        return "node", [str(path)], manifest_entry

    bins = package.get("bin") or {}
    if isinstance(bins, str):
        bins = {str(package.get("name") or "mcp"): bins}
    if isinstance(bins, dict) and bins:
        name = sorted(
            bins,
            key=lambda item: (
                0 if any(key in item.casefold() for key in ("mcp", "server")) else 1,
                item,
            ),
        )[0]
        path = (project_root / str(bins[name])).resolve()
        if path.exists():
            return "node", [str(path)], str(bins[name])

    for relative in (
        "dist/index.js",
        "build/index.js",
        "dist/server.js",
        "build/server.js",
        "server.js",
        "index.js",
    ):
        path = (project_root / relative).resolve()
        if path.exists():
            return "node", [str(path)], relative

    scripts = package.get("scripts") or {}
    for name in ("mcp", "start", "serve"):
        if isinstance(scripts, dict) and name in scripts:
            return (
                "npm",
                ["--silent", "--prefix", str(project_root), "run", name],
                f"npm run {name}",
            )
    raise McpImportError(
        "entrypoint_not_detected",
        "识别到 Node 项目，但没有找到可运行的 MCP 入口",
        detail="需要 package.json bin、start/mcp 脚本或常见 dist/build 入口。",
    )


def _prepare_node(project_root: Path, steps: List[Dict[str, str]]) -> tuple[str, List[str], str]:
    if not shutil.which("node") or not shutil.which("npm"):
        raise McpImportError(
            "node_runtime_missing",
            "当前 API 镜像尚未安装 Node/npm",
            detail="需要使用包含 Node 运行时的 YIAI API 镜像。",
            steps=steps,
        )
    package = _read_package_json(project_root)
    _run(
        ["npm", "--silent", "--prefix", str(project_root), "install", "--no-audit", "--no-fund"],
        cwd=project_root,
    )
    _step(steps, "install_dependencies", "passed", "已安装 Node 项目依赖")
    scripts = package.get("scripts") or {}
    if isinstance(scripts, dict) and "build" in scripts:
        _run(
            ["npm", "--silent", "--prefix", str(project_root), "run", "build"],
            cwd=project_root,
        )
        _step(steps, "build_package", "passed", "已完成 Node 项目构建")
    else:
        _step(steps, "build_package", "passed", "仓库不需要额外构建")
    return _node_entrypoint(project_root, package)


def _required_env_names(project_root: Path) -> List[str]:
    for filename in (".env.example", ".env.sample", "example.env"):
        path = project_root / filename
        if not path.exists():
            continue
        names: List[str] = []
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name = stripped.split("=", 1)[0].strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                names.append(name)
        return sorted(set(names))
    return []


def suggest_tool_effect(name: str, description: str) -> str:
    """Suggest an operator template without granting runtime authority."""

    normalized_name = str(name or "").casefold()
    normalized_description = str(description or "").casefold()
    if any(
        signal in normalized_name
        for signal in ("delete_", "remove_", "drop_", "destroy_", "purge_")
    ) or any(signal in normalized_description for signal in ("删除", "销毁", "清空")):
        return "unknown"
    if any(
        signal in normalized_name
        for signal in ("create_", "insert_", "save_", "submit_", "write_", "register_")
    ) or any(
        signal in normalized_description
        for signal in ("创建记录", "新增记录", "写入", "提交记录", "登记")
    ):
        return "create"
    if any(
        signal in normalized_name
        for signal in ("update_", "edit_", "modify_", "patch_", "set_")
    ) or any(signal in normalized_description for signal in ("更新记录", "修改记录")):
        return "update"
    return "read"


def prepare_git_mcp_package(
    git_url: str,
    *,
    requested_name: str = "",
    requested_runtime: str = "auto",
    package_root: Optional[str] = None,
) -> PreparedMcpPackage:
    """Clone, detect and prepare one public Git MCP repository."""

    source_url = _validate_git_url(git_url)
    runtime_request = (requested_runtime or "auto").strip().lower()
    if runtime_request not in {"auto", "python", "node"}:
        raise McpImportError(
            "invalid_runtime",
            "运行类型只能是自动识别、Python 或 Node",
        )

    root = Path(package_root or os.getenv("MCP_PACKAGE_DIR") or DEFAULT_PACKAGE_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:8]
    destination = root / f"{_slug(source_url)}-{source_hash}-{uuid.uuid4().hex[:6]}"
    steps: List[Dict[str, str]] = []

    try:
        _clone_git_repository(source_url, destination)
        _step(steps, "clone_repository", "passed", "Git 仓库已下载到持久化插件目录")
        project_root, runtime = _select_project_root(destination, runtime_request)
        _step(
            steps,
            "detect_runtime",
            "passed",
            f"已识别 {runtime} MCP 项目：{project_root.relative_to(destination) or Path('.')}",
        )
        if runtime == "python":
            command, args, entrypoint = _prepare_python(project_root, steps)
        else:
            command, args, entrypoint = _prepare_node(project_root, steps)
        _step(steps, "detect_entrypoint", "passed", f"启动入口：{entrypoint}")
        return PreparedMcpPackage(
            source_url=source_url,
            source_type="git",
            name=(requested_name or _slug(source_url)).strip(),
            runtime_type=runtime,
            package_path=str(project_root.resolve()),
            command=command,
            args=args,
            detected_entrypoint=entrypoint,
            required_env_names=_required_env_names(project_root),
            steps=steps,
        )
    except McpImportError as exc:
        if not exc.steps:
            exc.steps = steps
        _safe_remove(destination, root)
        raise
    except Exception as exc:
        _safe_remove(destination, root)
        raise McpImportError(
            "unexpected_import_error",
            "MCP 仓库准备失败",
            detail=str(exc),
            steps=steps,
        ) from exc
