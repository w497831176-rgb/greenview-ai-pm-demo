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
    if effect in {ToolEffect.CREATE, ToolEffect.UPDATE, ToolEffect.DELETE}:
        return ToolPolicy(
            server_id=server.get("id"),
            server_name=server_name,
            tool_name=tool_name,
            effect=effect,
            risk_level=RiskLevel.L3,
            allowed_paths=[],
            requires_confirmation=False,
            enabled=False,
            policy_reason=(
                "MCP 永久只读：create/update/delete 不进入确认或执行路径；"
                "业务写入只能使用显式内部服务 Action。"
            ),
        )
    return ToolPolicy(
        server_id=server.get("id"),
        server_name=server_name,
        tool_name=tool_name,
        effect=effect,
        risk_level=RiskLevel.L3,
        allowed_paths=[],
        requires_confirmation=False,
        enabled=False,
        policy_reason="未分类 MCP 工具默认高风险并拒绝发布到运行时。",
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
            if policy.effect != ToolEffect.READ:
                # Retain the declaration for operator visibility, but remove
                # every runtime planner/confirmation execution mode.
                runtime_metadata["execution_mode"] = "disabled"
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
                "domain_scope": agent.get("domain_scope") or "property",
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


_AGENT_FIELD_LABELS = {
    "name": "名称",
    "description": "描述",
    "instructions": "Prompt / Instructions",
    "model_id": "模型",
    "enabled": "启用状态",
    "category": "Agent 类型",
    "domain_scope": "业务域",
}

_SKILL_FIELD_LABELS = {
    "name": "名称",
    "description": "描述",
    "version": "版本",
    "trigger_condition": "触发条件",
    "enabled": "启用状态",
    "content_hash": "内容哈希",
}

_KNOWLEDGE_FIELD_LABELS = {
    "title": "文档名称",
    "category": "分类",
    "document_version": "文档版本",
    "index_status": "索引状态",
    "chunk_size": "Chunk 大小",
    "chunk_overlap": "Chunk 重叠",
    "split_strategy": "切片策略",
}

_MCP_FIELD_LABELS = {
    "name": "名称",
    "description": "描述",
    "enabled": "启用状态",
    "is_builtin": "内置能力",
}

_MCP_TOOL_FIELD_LABELS = {
    "description": "描述",
    "effect": "读写类型",
    "risk_level": "风险级别",
    "enabled": "运行时启用",
    "requires_confirmation": "需要确认",
    "execution_mode": "执行模式",
}


def _diff_value(value: Any, *, field: str, include_details: bool) -> Any:
    if field == "instructions" and isinstance(value, str):
        if include_details:
            return value
        return {
            "preview": value[:160],
            "length": len(value),
            "truncated": len(value) > 160,
        }
    return value


def _field_changes(
    before: Dict[str, Any],
    after: Dict[str, Any],
    labels: Dict[str, str],
    *,
    include_details: bool,
) -> List[Dict[str, Any]]:
    changes: List[Dict[str, Any]] = []
    for field, label in labels.items():
        old_value = before.get(field)
        new_value = after.get(field)
        if old_value == new_value:
            continue
        changes.append(
            {
                "field": field,
                "label": label,
                "old": _diff_value(
                    old_value,
                    field=field,
                    include_details=include_details,
                ),
                "new": _diff_value(
                    new_value,
                    field=field,
                    include_details=include_details,
                ),
            }
        )
    return changes


def _id_name_map(
    before: List[Dict[str, Any]],
    after: List[Dict[str, Any]],
    id_field: str,
    name_field: str,
) -> Dict[Any, str]:
    result: Dict[Any, str] = {}
    for item in [*before, *after]:
        item_id = item.get(id_field)
        if item_id is None:
            continue
        result[item_id] = str(item.get(name_field) or item_id)
    return result


def _named_binding_changes(
    before_values: List[Any],
    after_values: List[Any],
    names: Dict[Any, str],
    *,
    id_field: str,
) -> Dict[str, List[Dict[str, Any]]]:
    before_set = set(before_values or [])
    after_set = set(after_values or [])

    def rows(values: set[Any]) -> List[Dict[str, Any]]:
        return [
            {id_field: value, "name": names.get(value, str(value))}
            for value in sorted(values, key=lambda item: str(item))
        ]

    return {
        "added": rows(after_set - before_set),
        "removed": rows(before_set - after_set),
    }


def _change_type(
    before: Optional[Dict[str, Any]],
    after: Optional[Dict[str, Any]],
    fields: List[Dict[str, Any]],
    *,
    has_nested_changes: bool = False,
) -> Optional[str]:
    if before is None:
        return "added"
    if after is None:
        return "deleted"
    if bool(before.get("enabled")) and not bool(after.get("enabled")):
        return "disabled"
    if fields or has_nested_changes:
        return "modified"
    return None


def _collection_diff(
    before_items: List[Dict[str, Any]],
    after_items: List[Dict[str, Any]],
    *,
    key_field: str,
    name_field: str,
    labels: Dict[str, str],
    include_details: bool,
) -> List[Dict[str, Any]]:
    before_map = {item.get(key_field): item for item in before_items}
    after_map = {item.get(key_field): item for item in after_items}
    result: List[Dict[str, Any]] = []
    for item_id in sorted(
        set(before_map) | set(after_map),
        key=lambda item: str(item),
    ):
        before = before_map.get(item_id)
        after = after_map.get(item_id)
        fields = _field_changes(
            before or {},
            after or {},
            labels,
            include_details=include_details,
        )
        change_type = _change_type(before, after, fields)
        if not change_type:
            continue
        source = after or before or {}
        result.append(
            {
                "id": item_id,
                "name": str(source.get(name_field) or item_id),
                "change_type": change_type,
                "fields": fields,
            }
        )
    return result


def _mcp_tool_nodes(servers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for server in servers:
        server_name = str(server.get("name") or "")
        for tool in server.get("tools") or []:
            policy = tool.get("policy") or {}
            metadata = tool.get("tool_metadata") or {}
            tool_name = str(tool.get("name") or "")
            result.append(
                {
                    "tool_key": f"{server_name}:{tool_name}",
                    "name": f"{server_name} / {tool_name}",
                    "description": tool.get("description") or "",
                    "effect": policy.get("effect") or "unknown",
                    "risk_level": policy.get("risk_level") or "L3",
                    "enabled": bool(policy.get("enabled")),
                    "requires_confirmation": bool(
                        policy.get("requires_confirmation")
                    ),
                    "execution_mode": metadata.get("execution_mode")
                    or "unknown",
                }
            )
    return result


def _flatten_policy(
    value: Any,
    prefix: str = "",
) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    result: Dict[str, Any] = {}
    for key in sorted(value):
        path = f"{prefix}.{key}" if prefix else str(key)
        nested = value[key]
        if isinstance(nested, dict):
            result.update(_flatten_policy(nested, path))
        elif isinstance(nested, list):
            result[path] = {
                "count": len(nested),
                "hash": content_hash(nested),
            }
        else:
            result[path] = nested
    return result


def _policy_diff(
    before: Any,
    after: Any,
    *,
    category: str,
    label: str,
) -> List[Dict[str, Any]]:
    before_flat = _flatten_policy(before or {})
    after_flat = _flatten_policy(after or {})
    result: List[Dict[str, Any]] = []
    for path in sorted(set(before_flat) | set(after_flat)):
        if before_flat.get(path) == after_flat.get(path):
            continue
        result.append(
            {
                "category": category,
                "label": label,
                "field": path,
                "old": before_flat.get(path),
                "new": after_flat.get(path),
            }
        )
    return result


def diff_runtime_configs(
    before: Optional[Dict[str, Any]],
    after: Optional[Dict[str, Any]],
    *,
    include_details: bool = False,
) -> Dict[str, Any]:
    """Return a deterministic, presentation-ready RuntimeRelease diff.

    The lightweight form deliberately omits full configs, Chunk snapshots,
    Skill bodies and MCP schemas. Full Agent instructions are returned only
    when an operator explicitly requests detailed diff data.
    """

    before = before or {}
    after = after or {}
    before_skills = before.get("skills") or []
    after_skills = after.get("skills") or []
    before_knowledge = before.get("knowledge") or []
    after_knowledge = after.get("knowledge") or []
    before_servers = before.get("mcp_servers") or []
    after_servers = after.get("mcp_servers") or []

    skill_names = _id_name_map(
        before_skills,
        after_skills,
        "skill_id",
        "name",
    )
    knowledge_names = _id_name_map(
        before_knowledge,
        after_knowledge,
        "knowledge_doc_id",
        "title",
    )
    server_names = {
        str(item.get("name") or ""): str(item.get("name") or "")
        for item in [*before_servers, *after_servers]
        if item.get("name")
    }

    before_agents = {
        item.get("agent_id"): item for item in before.get("agents") or []
    }
    after_agents = {
        item.get("agent_id"): item for item in after.get("agents") or []
    }
    agent_changes: List[Dict[str, Any]] = []
    for agent_id in sorted(
        set(before_agents) | set(after_agents),
        key=lambda item: str(item),
    ):
        old_agent = before_agents.get(agent_id)
        new_agent = after_agents.get(agent_id)
        fields = _field_changes(
            old_agent or {},
            new_agent or {},
            _AGENT_FIELD_LABELS,
            include_details=include_details,
        )
        capabilities = {
            "skills": _named_binding_changes(
                (old_agent or {}).get("skill_ids") or [],
                (new_agent or {}).get("skill_ids") or [],
                skill_names,
                id_field="skill_id",
            ),
            "knowledge": _named_binding_changes(
                (old_agent or {}).get("knowledge_doc_ids") or [],
                (new_agent or {}).get("knowledge_doc_ids") or [],
                knowledge_names,
                id_field="knowledge_doc_id",
            ),
            "mcp_servers": _named_binding_changes(
                (old_agent or {}).get("mcp_server_names") or [],
                (new_agent or {}).get("mcp_server_names") or [],
                server_names,
                id_field="server_name",
            ),
        }
        has_capability_changes = any(
            values["added"] or values["removed"]
            for values in capabilities.values()
        )
        change_type = _change_type(
            old_agent,
            new_agent,
            fields,
            has_nested_changes=has_capability_changes,
        )
        if not change_type:
            continue
        source = new_agent or old_agent or {}
        agent_changes.append(
            {
                "agent_id": agent_id,
                "name": str(source.get("name") or agent_id),
                "change_type": change_type,
                "fields": fields,
                "capabilities": capabilities,
            }
        )

    skill_changes = _collection_diff(
        before_skills,
        after_skills,
        key_field="skill_id",
        name_field="name",
        labels=_SKILL_FIELD_LABELS,
        include_details=include_details,
    )
    knowledge_changes = _collection_diff(
        before_knowledge,
        after_knowledge,
        key_field="knowledge_doc_id",
        name_field="title",
        labels=_KNOWLEDGE_FIELD_LABELS,
        include_details=include_details,
    )
    mcp_changes = _collection_diff(
        before_servers,
        after_servers,
        key_field="name",
        name_field="name",
        labels=_MCP_FIELD_LABELS,
        include_details=include_details,
    )
    mcp_tool_changes = _collection_diff(
        _mcp_tool_nodes(before_servers),
        _mcp_tool_nodes(after_servers),
        key_field="tool_key",
        name_field="name",
        labels=_MCP_TOOL_FIELD_LABELS,
        include_details=include_details,
    )

    policy_changes = [
        *_policy_diff(
            before.get("model_policy"),
            after.get("model_policy"),
            category="model_policy",
            label="默认模型策略",
        ),
        *_policy_diff(
            before.get("retrieval_policy"),
            after.get("retrieval_policy"),
            category="retrieval_policy",
            label="检索参数",
        ),
        *_policy_diff(
            before.get("budget_policy"),
            after.get("budget_policy"),
            category="budget_policy",
            label="预算策略",
        ),
    ]

    all_entity_changes = [
        *agent_changes,
        *skill_changes,
        *knowledge_changes,
        *mcp_changes,
        *mcp_tool_changes,
    ]
    counts = {
        "added": sum(
            item["change_type"] == "added" for item in all_entity_changes
        ),
        "modified": sum(
            item["change_type"] == "modified" for item in all_entity_changes
        )
        + len(policy_changes),
        "disabled": sum(
            item["change_type"] == "disabled" for item in all_entity_changes
        ),
        "deleted": sum(
            item["change_type"] == "deleted" for item in all_entity_changes
        ),
        "affected_agents": len(agent_changes),
    }
    before_hash = content_hash(before)
    after_hash = content_hash(after)
    has_changes = before_hash != after_hash
    if has_changes and not any(counts.values()):
        policy_changes.append(
            {
                "category": "configuration",
                "label": "其他配置",
                "field": "snapshot",
                "old": before_hash[:12],
                "new": after_hash[:12],
            }
        )
        counts["modified"] = 1

    return {
        "has_changes": has_changes,
        "before_hash": before_hash,
        "after_hash": after_hash,
        "summary": counts,
        "agents": agent_changes,
        "skills": skill_changes,
        "knowledge": knowledge_changes,
        "mcp_servers": mcp_changes,
        "mcp_tools": mcp_tool_changes,
        "policies": policy_changes,
    }


def summarize_release(release: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not release:
        return None
    validation = release.get("validation") or {}
    return {
        "release_id": release.get("release_id"),
        "version": release.get("version"),
        "status": release.get("status"),
        "config_hash": release.get("config_hash"),
        "parent_release_id": release.get("parent_release_id"),
        "created_by": release.get("created_by"),
        "created_at": release.get("created_at"),
        "published_at": release.get("published_at"),
        "superseded_at": release.get("superseded_at"),
        "validation": {
            "valid": bool(validation.get("valid")),
            "errors": validation.get("errors") or [],
            "warnings": validation.get("warnings") or [],
            "counts": validation.get("counts") or {},
        },
    }


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
        if agent.get("domain_scope") not in {"property", "isolated_general"}:
            errors.append(
                {
                    "code": "agent_domain_scope_invalid",
                    "agent_id": agent.get("agent_id"),
                    "domain_scope": agent.get("domain_scope"),
                }
            )
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
            policy = tool.get("policy") or {}
            effect = str(policy.get("effect") or "unknown")
            if effect != ToolEffect.READ.value:
                if server.get("enabled"):
                    errors.append(
                        {
                            "code": "enabled_mcp_tool_must_be_read_only",
                            "server_name": server.get("name"),
                            "tool_name": tool.get("name"),
                            "effect": effect,
                            "detail": (
                                "MCP is permanently read-only; configure business "
                                "writes as explicit internal service Actions."
                            ),
                        }
                    )
                # Non-read MCP metadata is intentionally not interpreted as a
                # planner or Proposal contract, even on a disabled server.
                continue
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


def _prepare_runtime_candidate() -> Dict[str, Any]:
    graph, policies = _compile_graph()
    return {
        "config": graph,
        "policies": policies,
        "config_hash": content_hash(graph),
        "validation": validate_release_graph(graph, policies),
        "current": get_current_runtime_release(),
    }


def preview_runtime_release(
    created_by: str = "operator",
    *,
    include_details: bool = False,
) -> Dict[str, Any]:
    """Compile and validate current Draft configuration without persistence."""

    candidate = _prepare_runtime_candidate()
    current = candidate["current"]
    release_diff = diff_runtime_configs(
        (current or {}).get("config") or {},
        candidate["config"],
        include_details=include_details,
    )
    has_changes = bool(
        not current or candidate["config_hash"] != current.get("config_hash")
    )
    validation = candidate["validation"]
    can_publish = bool(validation.get("valid") and has_changes)
    return {
        "created_by": created_by,
        "config_hash": candidate["config_hash"],
        "has_changes": has_changes,
        "can_publish": can_publish,
        "block_reason": (
            None
            if can_publish
            else ("no_changes" if not has_changes else "validation_failed")
        ),
        "validation": validation,
        "diff": release_diff,
        "current_release": summarize_release(current),
        "persisted": False,
        "effective_on": "new_session",
        "existing_sessions": "keep_pinned_snapshot",
    }


def _persist_candidate(
    candidate: Dict[str, Any],
    *,
    created_by: str,
) -> Dict[str, Any]:
    current = candidate.get("current")
    version = next_runtime_release_version()
    release_id = f"rr_{version:04d}_{uuid.uuid4().hex[:8]}"
    release = create_runtime_release(
        release_id=release_id,
        version=version,
        config_hash=candidate["config_hash"],
        config=candidate["config"],
        validation=candidate["validation"],
        parent_release_id=(current or {}).get("release_id"),
        created_by=created_by,
    )
    replace_tool_policies(
        release_id,
        [
            policy.model_dump(mode="json")
            for policy in candidate.get("policies") or []
        ],
    )
    return release


def compile_runtime_release(created_by: str = "operator") -> Dict[str, Any]:
    """Persist one explicit draft.

    UI preview/validation does not call this function. It remains available for
    bootstrap and compatibility contracts that intentionally need a draft.
    """

    return _persist_candidate(
        _prepare_runtime_candidate(),
        created_by=created_by,
    )


def publish_current_runtime_config(
    created_by: str = "operator",
) -> Dict[str, Any]:
    """Validate and publish exactly once, or return a non-persistent block."""

    candidate = _prepare_runtime_candidate()
    current = candidate["current"]
    if current and candidate["config_hash"] == current.get("config_hash"):
        return {
            "release": summarize_release(current),
            "published": False,
            "created": False,
            "reason": "no_changes",
            "has_changes": False,
            "validation": candidate["validation"],
        }
    if not candidate["validation"].get("valid"):
        return {
            "release": None,
            "published": False,
            "created": False,
            "reason": "validation_failed",
            "has_changes": True,
            "validation": candidate["validation"],
            "diff": diff_runtime_configs(
                (current or {}).get("config") or {},
                candidate["config"],
            ),
        }
    release = _persist_candidate(candidate, created_by=created_by)
    published = publish_runtime_release(release["release_id"])
    return {
        "release": published,
        "published": True,
        "created": True,
        "reason": "published",
        "has_changes": True,
        "validation": candidate["validation"],
    }


def publish_compiled_release(created_by: str = "operator") -> Dict[str, Any]:
    result = publish_current_runtime_config(created_by=created_by)
    release = dict(result.get("release") or {})
    if not release:
        release["validation"] = result.get("validation") or {}
        release["status"] = "draft"
    release["_publish_result"] = {
        "published": result.get("published"),
        "created": result.get("created"),
        "reason": result.get("reason"),
    }
    return release


def ensure_bootstrap_release() -> Dict[str, Any]:
    current = get_current_runtime_release()
    if current:
        return current
    release = publish_compiled_release(created_by="bootstrap")
    if release.get("status") != "published":
        raise RuntimeError(f"bootstrap RuntimeRelease validation failed: {release.get('validation')}")
    return release
