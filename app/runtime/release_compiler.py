"""Compile editable platform configuration into an immutable RuntimeRelease."""

from __future__ import annotations

import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.runtime.contracts import (
    RiskLevel,
    RuntimePath,
    ToolEffect,
    ToolPolicy,
    content_hash,
)
from app.runtime.tool_planner import (
    effective_tool_metadata,
    validate_tool_metadata,
)
from db.property_db import (
    create_runtime_release,
    get_agent_skills,
    get_agent_tools,
    get_agent_knowledge_bindings,
    get_budget_thresholds,
    get_current_runtime_release,
    get_default_model_config,
    get_retrieval_settings,
    list_agents,
    list_knowledge_docs,
    list_mcp_servers,
    list_mcp_tools,
    list_model_configs,
    list_model_prices,
    list_skills,
    next_runtime_release_version,
    publish_runtime_release,
    replace_tool_policies,
)


def _tool_effect(tool: Dict[str, Any]) -> ToolEffect:
    metadata = tool.get("tool_metadata") or {}
    declared = str(metadata.get("effect") or metadata.get("operation") or "").lower()
    effect_source = str(metadata.get("effect_source") or "")
    if (
        declared in {item.value for item in ToolEffect}
        and effect_source
        in {
            "operator_declared",
            "operator_declared_legacy",
            "builtin_compatibility",
        }
    ):
        return ToolEffect(declared)
    return ToolEffect.UNKNOWN


def compile_tool_policy(server: Dict[str, Any], tool: Dict[str, Any]) -> ToolPolicy:
    effect = _tool_effect(tool)
    metadata = tool.get("tool_metadata") or {}
    declared_risk = str(metadata.get("risk_level") or "")
    server_name = str(server.get("name") or "")
    tool_name = str(tool.get("name") or "")
    if effect == ToolEffect.READ:
        return ToolPolicy(
            server_id=server.get("id"),
            server_name=server_name,
            tool_name=tool_name,
            effect=effect,
            risk_level=(
                RiskLevel(declared_risk)
                if declared_risk in {"L0", "L1"}
                else RiskLevel.L1
            ),
            allowed_paths=[RuntimePath.CONSULTATION, RuntimePath.EXTENSION_ACCEPTANCE],
            requires_confirmation=False,
            enabled=bool(server.get("enabled")),
            policy_reason="只读工具可在已发布白名单内自动执行。",
        )
    if effect in {ToolEffect.CREATE, ToolEffect.UPDATE}:
        return ToolPolicy(
            server_id=server.get("id"),
            server_name=server_name,
            tool_name=tool_name,
            effect=effect,
            risk_level=(
                RiskLevel(declared_risk)
                if declared_risk in {"L2", "L3"}
                else RiskLevel.L2
            ),
            allowed_paths=[RuntimePath.CONTROLLED_ACTION, RuntimePath.EXTENSION_ACCEPTANCE],
            requires_confirmation=True,
            enabled=bool(server.get("enabled")),
            policy_reason="写工具只允许生成 Proposal；确认后由 ActionGateway 执行。",
        )
    return ToolPolicy(
        server_id=server.get("id"),
        server_name=server_name,
        tool_name=tool_name,
        effect=effect,
        risk_level=RiskLevel.L3,
        allowed_paths=[],
        requires_confirmation=True,
        enabled=False,
        policy_reason=(
            "V1.8 禁用删除/破坏性工具。"
            if effect == ToolEffect.DELETE
            else "未分类工具默认高风险并拒绝发布到运行时。"
        ),
    )


def _skill_version(skill: Dict[str, Any]) -> str:
    metadata = skill.get("skill_metadata") or {}
    return str(metadata.get("version") or "legacy-1.0.0")


def _skill_reference_snapshots(skill: Dict[str, Any]) -> List[Dict[str, str]]:
    source: Optional[Path] = None
    storage_path = str(skill.get("storage_path") or "").strip()
    if storage_path:
        candidate = Path(storage_path)
        if candidate.exists():
            source = candidate
    if source is None:
        try:
            import skill_storage

            candidate = Path(skill_storage._skill_dir(int(skill["id"])))
            if candidate.exists():
                source = candidate
        except Exception:
            source = None
    references_root = source / "references" if source else None
    if not references_root or not references_root.is_dir():
        return []
    snapshots: List[Dict[str, str]] = []
    consumed = 0
    for reference in sorted(references_root.rglob("*")):
        if not reference.is_file():
            continue
        raw = reference.read_bytes()
        if consumed + len(raw) > 256_000:
            break
        relative = reference.relative_to(references_root).as_posix()
        content = raw.decode("utf-8", errors="replace")
        snapshots.append(
            {
                "path": relative,
                "content": content,
                "content_hash": content_hash(raw.hex()),
            }
        )
        consumed += len(raw)
    return snapshots


def _public_model_config(config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not config:
        return None
    return {
        key: value
        for key, value in dict(config).items()
        if key not in {"api_key"}
    }


def _knowledge_chunk_snapshots(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    content = str(doc.get("content") or "")
    try:
        import rag_chunking

        chunks = rag_chunking.split_text(
            content,
            strategy=doc.get("split_strategy") or "auto",
            chunk_size=int(doc.get("chunk_size") or 512),
            chunk_overlap=int(doc.get("chunk_overlap") or 64),
        )
    except Exception:
        chunks = [content] if content else []
    return [
        {
            "chunk_index": index,
            "content": str(chunk),
            "chunk_hash": content_hash(str(chunk)),
        }
        for index, chunk in enumerate(chunks)
        if str(chunk)
    ]


def _compile_graph() -> Tuple[Dict[str, Any], List[ToolPolicy]]:
    skills = list_skills()
    agents = list_agents()
    docs = list_knowledge_docs()
    servers = list_mcp_servers()
    all_tools = list_mcp_tools()

    skill_nodes = []
    for skill in skills:
        instructions = skill.get("instructions") or ""
        skill_nodes.append(
            {
                "skill_id": int(skill["id"]),
                "name": skill.get("name") or "",
                "description": skill.get("description") or "",
                "version": _skill_version(skill),
                "enabled": bool(skill.get("enabled")),
                "trigger_condition": skill.get("trigger_condition") or "",
                "metadata": skill.get("skill_metadata") or {},
                "content_hash": content_hash(instructions),
                "reference_snapshots": _skill_reference_snapshots(skill),
                # Runtime progressive loading uses the immutable package.  The
                # body remains here only as a compatibility fallback for
                # legacy DB Skills that have not yet been packaged.
                "instructions_fallback": instructions,
            }
        )

    knowledge_nodes = []
    published_doc_ids = []
    for doc in docs:
        if not doc.get("is_indexed") or doc.get("source_type") == "demo_test":
            continue
        doc_id = int(doc["id"])
        published_doc_ids.append(doc_id)
        body = doc.get("content") or ""
        digest = content_hash(body)
        knowledge_nodes.append(
            {
                "knowledge_doc_id": doc_id,
                "title": doc.get("title") or "",
                "category": doc.get("category") or "",
                "document_version": digest[:16],
                "document_hash": digest,
                "index_status": doc.get("index_status") or "unknown",
                "chunk_count": int(doc.get("chunk_count") or 0),
                "chunk_size": int(doc.get("chunk_size") or 512),
                "chunk_overlap": int(doc.get("chunk_overlap") or 64),
                "split_strategy": doc.get("split_strategy") or "auto",
                "chunk_snapshots": _knowledge_chunk_snapshots(doc),
            }
        )

    policies: List[ToolPolicy] = []
    server_nodes = []
    for server in servers:
        tools = [item for item in all_tools if int(item.get("server_id") or 0) == int(server["id"])]
        compiled_tools = []
        for tool in tools:
            runtime_metadata = effective_tool_metadata(
                str(server.get("name") or ""),
                str(tool.get("name") or ""),
                tool.get("tool_metadata") or {},
            )
            effective_tool = {**tool, "tool_metadata": runtime_metadata}
            policy = compile_tool_policy(server, effective_tool)
            if (
                "execution_mode" not in (tool.get("tool_metadata") or {})
                and policy.effect in {ToolEffect.CREATE, ToolEffect.UPDATE}
            ):
                runtime_metadata["execution_mode"] = "proposal"
            policies.append(policy)
            compiled_tools.append(
                {
                    "tool_id": int(tool["id"]),
                    "name": tool.get("name") or "",
                    "description": tool.get("description") or "",
                    "input_schema": tool.get("input_schema") or {},
                    "tool_metadata": runtime_metadata,
                    "policy": policy.model_dump(mode="json"),
                }
            )
        server_nodes.append(
            {
                "server_id": int(server["id"]),
                "name": server.get("name") or "",
                "description": server.get("description") or "",
                "enabled": bool(server.get("enabled")),
                "is_builtin": bool(server.get("is_builtin")),
                "command": server.get("command"),
                "args": server.get("args") or [],
                # Credentials remain deployment-owned.  A RuntimeRelease pins
                # required variable names, never secret values.
                "env_keys": sorted((server.get("env") or {}).keys()),
                "tools": compiled_tools,
            }
        )

    agent_nodes = []
    for agent in agents:
        agent_id = str(agent.get("agent_id") or "")
        bound_skill_ids = [int(item) for item in get_agent_skills(agent_id)]
        bound_servers = [
            str(item.get("tool_name") or "")
            for item in get_agent_tools(agent_id)
            if item.get("tool_name")
        ]
        # V1.7 had no Agent-RAG binding table/UI.  The bootstrap compiler turns
        # its former "all published business docs" behavior into an explicit
        # release-level binding.  Later releases may narrow this list through
        # the V1.8 binding API without changing existing snapshots.
        explicit_knowledge_ids = get_agent_knowledge_bindings(agent_id)
        if agent_id == "router":
            bound_knowledge_ids = []
        elif explicit_knowledge_ids is None:
            bound_knowledge_ids = list(published_doc_ids)
        else:
            bound_knowledge_ids = [
                item for item in explicit_knowledge_ids if item in published_doc_ids
            ]
        agent_nodes.append(
            {
                "agent_id": agent_id,
                "name": agent.get("name") or "",
                "description": agent.get("description") or "",
                "instructions": agent.get("instructions") or "",
                "category": agent.get("category") or "vertical",
                "enabled": bool(agent.get("enabled")),
                "model_id": agent.get("model_id"),
                "skill_ids": bound_skill_ids,
                "mcp_server_names": bound_servers,
                "knowledge_doc_ids": bound_knowledge_ids,
            }
        )

    graph = {
        "schema_version": "1.0",
        "agents": agent_nodes,
        "skills": skill_nodes,
        "knowledge": knowledge_nodes,
        "mcp_servers": server_nodes,
        "bindings": {
            "agent_skill": [
                {"agent_id": agent["agent_id"], "skill_id": skill_id}
                for agent in agent_nodes
                for skill_id in agent["skill_ids"]
            ],
            "agent_mcp": [
                {"agent_id": agent["agent_id"], "server_name": server_name}
                for agent in agent_nodes
                for server_name in agent["mcp_server_names"]
            ],
            "agent_knowledge": [
                {"agent_id": agent["agent_id"], "knowledge_doc_id": doc_id}
                for agent in agent_nodes
                for doc_id in agent["knowledge_doc_ids"]
            ],
        },
        "model_policy": {
            "version": "v1.8",
            "default": _public_model_config(get_default_model_config()),
            "available": [
                _public_model_config(item) for item in list_model_configs()
            ],
        },
        "price_snapshots": list_model_prices(enabled_only=True),
        "budget_policy": get_budget_thresholds(),
        "retrieval_policy": get_retrieval_settings("default") or {},
    }
    return graph, policies


def validate_release_graph(graph: Dict[str, Any], policies: List[ToolPolicy]) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    agents = graph.get("agents") or []
    skills = graph.get("skills") or []
    knowledge = graph.get("knowledge") or []
    servers = graph.get("mcp_servers") or []

    for field, nodes in (("agent", agents), ("skill", skills), ("mcp_server", servers)):
        key = "agent_id" if field == "agent" else "name"
        counts = Counter(str(node.get(key) or "").strip().lower() for node in nodes)
        for value, count in counts.items():
            if not value:
                errors.append({"code": f"{field}_identity_missing", "value": value})
            elif count > 1:
                errors.append({"code": f"{field}_identity_duplicate", "value": value})

    enabled_skill_ids = {int(item["skill_id"]) for item in skills if item.get("enabled")}
    knowledge_ids = {int(item["knowledge_doc_id"]) for item in knowledge}
    server_names = {str(item["name"]) for item in servers if item.get("enabled")}
    for agent in agents:
        if not agent.get("enabled") or agent.get("category") in {"router", "orchestration"}:
            continue
        if not str(agent.get("instructions") or "").strip():
            warnings.append({"code": "agent_instructions_empty", "agent_id": agent.get("agent_id")})
        missing_skills = set(agent.get("skill_ids") or []) - enabled_skill_ids
        missing_docs = set(agent.get("knowledge_doc_ids") or []) - knowledge_ids
        missing_servers = set(agent.get("mcp_server_names") or []) - server_names
        if missing_skills:
            errors.append({"code": "agent_skill_binding_invalid", "agent_id": agent["agent_id"], "ids": sorted(missing_skills)})
        if missing_docs:
            errors.append({"code": "agent_knowledge_binding_invalid", "agent_id": agent["agent_id"], "ids": sorted(missing_docs)})
        if missing_servers:
            errors.append({"code": "agent_mcp_binding_invalid", "agent_id": agent["agent_id"], "names": sorted(missing_servers)})

    for policy in policies:
        if policy.effect == ToolEffect.UNKNOWN:
            warnings.append(
                {
                    "code": "tool_unclassified_disabled",
                    "server_name": policy.server_name,
                    "tool_name": policy.tool_name,
                }
            )

    for server in servers:
        for tool in server.get("tools") or []:
            metadata = tool.get("tool_metadata") or {}
            metadata_errors = validate_tool_metadata(
                metadata,
                tool.get("input_schema") or {},
            )
            for detail in metadata_errors:
                errors.append(
                    {
                        "code": "tool_runtime_metadata_invalid",
                        "server_name": server.get("name"),
                        "tool_name": tool.get("name"),
                        "detail": detail,
                    }
                )
            policy = tool.get("policy") or {}
            effect = str(policy.get("effect") or "unknown")
            if effect in {"create", "update"} and not (
                metadata.get("trigger_keywords") or []
            ):
                warnings.append(
                    {
                        "code": "write_tool_has_no_natural_language_trigger",
                        "server_name": server.get("name"),
                        "tool_name": tool.get("name"),
                        "detail": (
                            "工具仍保留在发布快照，但自然语言不会自动进入写路径；"
                            "请在平台配置 trigger_keywords。"
                        ),
                    }
                )

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "agents": len(agents),
            "skills": len(skills),
            "knowledge_docs": len(knowledge),
            "mcp_servers": len(servers),
            "tool_policies": len(policies),
        },
    }


def compile_runtime_release(created_by: str = "operator") -> Dict[str, Any]:
    graph, policies = _compile_graph()
    validation = validate_release_graph(graph, policies)
    current = get_current_runtime_release()
    version = next_runtime_release_version()
    release_id = f"rr_{version:04d}_{uuid.uuid4().hex[:8]}"
    release = create_runtime_release(
        release_id=release_id,
        version=version,
        config_hash=content_hash(graph),
        config=graph,
        validation=validation,
        parent_release_id=(current or {}).get("release_id"),
        created_by=created_by,
    )
    replace_tool_policies(
        release_id,
        [policy.model_dump(mode="json") for policy in policies],
    )
    return release


def publish_compiled_release(created_by: str = "operator") -> Dict[str, Any]:
    release = compile_runtime_release(created_by=created_by)
    if not (release.get("validation") or {}).get("valid"):
        return release
    return publish_runtime_release(release["release_id"])


def ensure_bootstrap_release() -> Dict[str, Any]:
    current = get_current_runtime_release()
    if current:
        return current
    release = publish_compiled_release(created_by="bootstrap")
    if release.get("status") != "published":
        raise RuntimeError(f"bootstrap RuntimeRelease validation failed: {release.get('validation')}")
    return release
