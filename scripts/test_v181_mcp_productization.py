"""No-model contract checks for the productized MCP Git import flow."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from app.runtime.mcp_importer import (
    McpImportError,
    _node_entrypoint,
    _python_entrypoint,
    _select_project_root,
    _validate_git_url,
    suggest_tool_effect,
)


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_git_url_contract() -> None:
    ensure(
        _validate_git_url("https://github.com/example/demo-mcp.git")
        == "https://github.com/example/demo-mcp.git",
        "public Git URL accepted",
    )
    try:
        _validate_git_url("https://token@example.com/demo.git")
    except McpImportError as exc:
        ensure(exc.code == "credential_in_git_url", "credentials rejected")
    else:
        raise AssertionError("credential-bearing Git URL must be rejected")


def test_python_detection() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "pyproject.toml").write_text(
            """
[project]
name = "demo-mcp"
version = "0.1.0"
dependencies = ["mcp"]
""".strip(),
            encoding="utf-8",
        )
        server = root / "astronomy_mcp_server.py"
        server.write_text(
            """
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("astronomy")
if __name__ == "__main__":
    mcp.run(transport="stdio")
""".strip(),
            encoding="utf-8",
        )
        project_root, runtime = _select_project_root(root, "auto")
        kind, entrypoint = _python_entrypoint(project_root)
        ensure(runtime == "python", "Python runtime detected")
        ensure(kind == "file", "Python file entrypoint detected")
        ensure(Path(entrypoint).name == server.name, "correct Python entrypoint")


def test_node_detection() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "package.json").write_text(
            json.dumps(
                {
                    "name": "demo-node-mcp",
                    "version": "1.0.0",
                    "scripts": {"start": "node dist/index.js"},
                }
            ),
            encoding="utf-8",
        )
        (root / "dist").mkdir()
        (root / "dist" / "index.js").write_text(
            "console.error('mcp');",
            encoding="utf-8",
        )
        project_root, runtime = _select_project_root(root, "auto")
        command, args, entrypoint = _node_entrypoint(
            project_root,
            json.loads((root / "package.json").read_text(encoding="utf-8")),
        )
        ensure(runtime == "node", "Node runtime detected")
        ensure(command == "node", "Node executable selected")
        ensure(Path(args[0]).name == "index.js", "Node entrypoint detected")
        ensure(entrypoint == "dist/index.js", "Node evidence retained")


def test_tool_effect_suggestions() -> None:
    ensure(suggest_tool_effect("lookup_window", "query data") == "read", "read suggested")
    ensure(
        suggest_tool_effect("create_plan", "create record") == "create",
        "create suggested",
    )
    ensure(
        suggest_tool_effect("update_plan", "modify record") == "update",
        "update suggested",
    )
    ensure(
        suggest_tool_effect("delete_plan", "remove record") == "unknown",
        "destructive defaults to unknown",
    )
    ensure(
        suggest_tool_effect("add", "计算两个数的和。") == "read",
        "arithmetic add is not misclassified as a write",
    )


def test_frontend_contract() -> None:
    source = Path("frontend/index.html").read_text(encoding="utf-8")
    for marker in (
        "从 Git 导入",
        "/api/mcp-servers/import-git",
        "连接并刷新 Tool",
        "只读查询（可直接调用）",
        "新增/写入（必须确认）",
        "高级手动配置",
    ):
        ensure(marker in source, f"frontend marker present: {marker}")
    ensure(
        "await apiPost(`/api/mcp-servers/${createdId}/discover`, {});" in source,
        "manual create automatically discovers tools",
    )


def main() -> None:
    test_git_url_contract()
    test_python_detection()
    test_node_detection()
    test_tool_effect_suggestions()
    test_frontend_contract()
    print("V1.8.1 MCP productization contract checks passed")


if __name__ == "__main__":
    main()
