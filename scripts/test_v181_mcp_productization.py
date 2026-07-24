"""No-model contract checks for the productized MCP Git import flow."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from app.runtime.mcp_importer import (
    McpImportError,
    _node_entrypoint,
    _python_launch_spec,
    _python_entrypoint,
    _select_project_root,
    _validate_git_url,
    suggest_tool_effect,
)
from app.runtime.mcp_executor import build_model_native_read_tools


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


def test_python_runtime_uses_agno_compatible_launcher() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        command, args = _python_launch_spec(
            root,
            "console_script",
            "demo-mcp",
        )
        ensure(command == "uv", "Python MCP uses Agno allowlisted uv launcher")
        ensure(args[:3] == ["run", "--directory", str(root)], "project root retained")
        ensure("--no-sync" in args, "prepared virtualenv is reused")
        ensure(args[-1] == "demo-mcp", "console script retained")


def _model_native_fixture() -> dict:
    return {
        "agents": [
            {
                "agent_id": "chaos_agent",
                "enabled": True,
                "mcp_server_names": ["calculator-server"],
            }
        ],
        "mcp_servers": [
            {
                "server_id": 1,
                "name": "calculator-server",
                "enabled": True,
                "command": "python",
                "args": ["/tmp/calculator.py"],
                "tools": [
                    {
                        "name": "calculate",
                        "description": "Calculate a mathematical expression",
                        "input_schema": {
                            "type": "object",
                            "properties": {"expression": {"type": "string"}},
                            "required": ["expression"],
                        },
                        "tool_metadata": {
                            "effect": "read",
                            "risk_level": "L1",
                            "execution_mode": "model_native",
                            "natural_language_intents": ["精确计算数学表达式"],
                            "trigger_keywords": ["计算", "calculate"],
                            "trigger_mode": "any",
                            "argument_bindings": {},
                            "result_contract": {
                                "success_statuses": ["success"],
                                "non_success_statuses": ["invalid_input"],
                            },
                        },
                        "policy": {
                            "server_id": 1,
                            "server_name": "calculator-server",
                            "tool_name": "calculate",
                            "effect": "read",
                            "risk_level": "L1",
                            "allowed_paths": ["consultation", "extension_acceptance"],
                            "requires_confirmation": False,
                            "enabled": True,
                            "policy_reason": "test fixture",
                        },
                    }
                ],
            }
        ],
    }


def test_model_native_tools_are_message_gated() -> None:
    config = _model_native_fixture()
    ensure(
        build_model_native_read_tools(
            config,
            "chaos_agent",
            "先只回复你被路由到的 Agent 名称，不调用其他能力。",
        )
        == [],
        "route-only prompt must not attach or start a bound MCP server",
    )
    matched = build_model_native_read_tools(
        config,
        "chaos_agent",
        "请调用 calculate 工具计算表达式。",
    )
    ensure(len(matched) == 1, "matching prompt attaches the published MCP tool")


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
    test_python_runtime_uses_agno_compatible_launcher()
    test_model_native_tools_are_message_gated()
    test_node_detection()
    test_tool_effect_suggestions()
    test_frontend_contract()
    print("V1.8.1 MCP productization contract checks passed")


if __name__ == "__main__":
    main()
