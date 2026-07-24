"""Isolated no-model integration check for Git import plus MCP discovery.

Set MCP_IMPORT_FIXTURE_URL to a tiny public/dumb-HTTP Git repository that
contains a stdio FastMCP server.  PROPERTY_DATA_DIR and MCP_PACKAGE_DIR must
point at disposable directories.
"""

from __future__ import annotations

import asyncio
import os

from app.mcp import discover_server_tools
from app.runtime.mcp_importer import prepare_git_mcp_package
from db import property_db


async def main() -> None:
    fixture_url = os.environ["MCP_IMPORT_FIXTURE_URL"]
    expected_runtime = os.getenv("MCP_IMPORT_EXPECTED_RUNTIME", "python")
    expected_tool = os.getenv(
        "MCP_IMPORT_EXPECTED_TOOL",
        "lookup_transit_window",
    )
    property_db.init_db()
    prepared = await asyncio.to_thread(
        prepare_git_mcp_package,
        fixture_url,
        requested_name=f"fixture-{expected_runtime}-mcp",
        requested_runtime="auto",
    )
    server = property_db.create_mcp_server(
        name=prepared.name,
        command=prepared.command,
        args=prepared.args,
        env={},
        description="isolated MCP Git import fixture",
        enabled=True,
        source_type="git",
        source_url=prepared.source_url,
        runtime_type=prepared.runtime_type,
        install_status="discovering",
        package_path=prepared.package_path,
        detected_entrypoint=prepared.detected_entrypoint,
    )
    discovered = await discover_server_tools(server)
    names = {item["name"] for item in discovered}
    assert prepared.runtime_type == expected_runtime
    assert expected_tool in names
    cached = property_db.list_mcp_tools(server_id=int(server["id"]))
    assert {item["name"] for item in cached} == names
    print(
        "V1.8.1 MCP Git import integration passed:",
        prepared.runtime_type,
        sorted(names),
    )


if __name__ == "__main__":
    asyncio.run(main())
