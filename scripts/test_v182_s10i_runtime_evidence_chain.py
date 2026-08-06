"""S10-I offline behavior checks for the converged runtime evidence chain.

This script deliberately exercises the real RuntimeCoordinator, retrieval
planner/merge, citation renderer, EvidenceLedger persistence and SSE protocol.
Only the lowest-level Provider output and MCP transport are deterministic
fixtures.  RuntimeRelease resolution, pgvector storage/indexing/retrieval and
Agno Agent assembly stay real.  It must run in fresh, isolated data stores
inside containers started with ``--network none``.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The test data boundary is established before importing any app/db module.
_TEMP_ROOT = tempfile.TemporaryDirectory(prefix="yiai-s10i-evidence-chain-")
_DATA_DIR = (Path(_TEMP_ROOT.name) / "property-data").resolve()
_FORBIDDEN_DATA_ROOTS = (
    Path("/app/data"),
    Path("/volume3/docker/agno-demo-os"),
    ROOT,
)
if any(
    _DATA_DIR == item.resolve() or item.resolve() in _DATA_DIR.parents
    for item in _FORBIDDEN_DATA_ROOTS
):
    raise RuntimeError(f"unsafe S10-I test data directory: {_DATA_DIR}")
os.environ["PROPERTY_DATA_DIR"] = str(_DATA_DIR)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# A second disposable --network none container exposes pgvector only through
# this shared Unix socket.  Refuse TCP/default compose hosts so this behavior
# test can never touch a production or developer database by accident.
_PG_SOCKET_DIR = Path(
    os.environ.get("S10I_PG_SOCKET_DIR", "/tmp/yiai-s10i-pg")
).resolve()
if _PG_SOCKET_DIR == Path("/tmp") or Path("/tmp") not in _PG_SOCKET_DIR.parents:
    raise RuntimeError(f"unsafe S10-I pgvector socket directory: {_PG_SOCKET_DIR}")
_PG_HOST = quote(str(_PG_SOCKET_DIR), safe="")
os.environ["DB_HOST"] = _PG_HOST
os.environ["DB_PORT"] = "5432"
os.environ["DB_USER"] = "s10i"
os.environ["DB_PASS"] = "s10i-offline"
os.environ["DB_DATABASE"] = "s10i"
os.environ["POSTGRES_URL"] = (
    "postgresql+psycopg://s10i:s10i-offline@/s10i?host=" + _PG_HOST
)
for _provider_key in (
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "PARALLEL_API_KEY",
):
    os.environ[_provider_key] = ""

from db import property_db as db  # noqa: E402

db.init_db()

import rag_embeddings  # noqa: E402
import rag_indexer  # noqa: E402
import rag_store  # noqa: E402
import rag_chunking  # noqa: E402
from agno.agent import Agent as AgnoAgent  # noqa: E402
from agno.models.base import Model  # noqa: E402
from agno.models.response import ModelResponse  # noqa: E402
from app.runtime import agent_factory as agent_factory_module  # noqa: E402
from app.runtime import coordinator as coordinator_module  # noqa: E402
from app.runtime import mcp_executor as mcp_executor_module  # noqa: E402
from app.runtime.citation_renderer import (  # noqa: E402
    build_run_evidence_bundle,
    render_bundle_citations,
)
from app.runtime.contracts import (  # noqa: E402
    EvidenceItem,
    EvidenceSet,
    ToolEffect,
    ToolInvocation,
    content_hash,
)
from app.runtime.coordinator import (  # noqa: E402
    KNOWLEDGE_INSUFFICIENT_RESPONSE,
    RuntimeCoordinator,
)
from app.runtime.evidence_ledger import project_evidence_for_trace  # noqa: E402
from app.runtime.mcp_executor import _evaluate_read_tool_result  # noqa: E402
from app.runtime.snapshot_resolver import resolve_snapshot  # noqa: E402


QUERY = (
    "请依据《物业维修服务承诺》说明紧急维修登记和到场时限，"
    "同时查询上海天气及最近维修工单；只读、不写入、不转人工。"
)
TARGET_CHUNK = (
    "物业客服中心在5分钟内完成工单登记并通知工程人员，"
    "工程人员30分钟内到场处置。"
)
SESSION_ID = "s10i-runtime-evidence-chain"
USER_ID = "s10i-offline-owner"
WORK_ORDER = db.get_work_order("WO-20260710-001")
if not WORK_ORDER:
    raise RuntimeError("fresh S10-I fixture is missing seeded work order")
WORK_ORDER_ID = str(WORK_ORDER["id"])
WORK_ORDER_STATUS = str(WORK_ORDER["status"])


def _check(condition: Any, label: str, detail: Any = None) -> None:
    if not condition:
        raise AssertionError(f"{label}: {detail!r}")
    print(f"PASS {label}")


def _provider_attempt_count() -> int:
    with sqlite3.connect(str(db.DB_PATH)) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) FROM model_calls
            WHERE record_kind = 'provider_attempt'
            """
        ).fetchone()
    return int(row[0] if row else 0)


def _policy(server_name: str, tool_name: str) -> Dict[str, Any]:
    return {
        "server_name": server_name,
        "tool_name": tool_name,
        "effect": "read",
        "risk_level": "L1",
        "allowed_paths": ["consultation"],
        "requires_confirmation": False,
        "enabled": True,
        "policy_reason": "S10-I offline read-only fixture",
    }


SKILL_INSTRUCTIONS = (
    "先核对本轮发布证据，再向业主分别说明制度依据与只读查询结果；"
    "不得创建草稿、Proposal或业务写入。"
)
SNAPSHOT_CONFIG: Dict[str, Any] = {
    "agents": [
        {
            "agent_id": "router",
            "name": "路由 Agent",
            "category": "orchestration",
            "enabled": True,
            "model_id": "deepseek-v4-flash",
            "instructions": "只做A/B/C语义路由。",
        },
        {
            "agent_id": "maintenance",
            "name": "维修 Agent",
            "description": "维修承诺、天气影响和维修工单只读查询",
            "category": "vertical",
            "domain_scope": "property",
            "enabled": True,
            "model_id": "deepseek-v4-flash",
            "instructions": "只根据本轮不可变证据合同回答。",
            "skill_ids": [8],
            "knowledge_doc_ids": [1],
            "mcp_server_names": ["weather-server", "workorder-server"],
        },
    ],
    "skills": [
        {
            "skill_id": 8,
            "name": "维修响应说明",
            "description": "根据发布证据解释维修服务并组合只读结果",
            "instructions_fallback": SKILL_INSTRUCTIONS,
            "trigger_condition": "服务承诺",
            "enabled": True,
            "version": "1.0.0",
            "content_hash": content_hash(SKILL_INSTRUCTIONS),
            "reference_snapshots": [],
            "metadata": {
                "contract_version": "1.0",
                "version": "1.0.0",
                "positive_triggers": ["服务承诺", "紧急维修"],
                "negative_triggers": [],
                "priority": 80,
                "conflict_group": "maintenance-guidance",
                "composable": False,
                "always_on": False,
            },
        }
    ],
    "knowledge": [
        {
            "knowledge_doc_id": 1,
            "title": "物业维修服务承诺",
            "category": "维修服务",
            "document_version": "v27-doc1",
            "document_hash": content_hash(
                {"title": "物业维修服务承诺", "content": TARGET_CHUNK}
            ),
            "chunk_snapshots": [
                {
                    "chunk_index": 1,
                    "content": TARGET_CHUNK,
                    "chunk_hash": content_hash(TARGET_CHUNK),
                }
            ],
        }
    ],
    "mcp_servers": [
        {
            "server_id": 101,
            "name": "weather-server",
            "description": "天气只读查询",
            "enabled": True,
            "command": "fixture-weather",
            "args": [],
            "env_keys": [],
            "tools": [
                {
                    "name": "get_current_weather",
                    "description": "查询指定城市当前天气",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                    "policy": _policy(
                        "weather-server", "get_current_weather"
                    ),
                    "tool_metadata": {
                        "result_contract": {
                            "success_statuses": ["success"],
                            "non_success_statuses": [
                                "not_found",
                                "timeout",
                                "upstream_error",
                            ],
                            "evidence_units": {"wind_level": "级"},
                        }
                    },
                }
            ],
        },
        {
            "server_id": 102,
            "name": "workorder-server",
            "description": "维修工单只读查询",
            "enabled": True,
            "command": "fixture-workorder",
            "args": [],
            "env_keys": [],
            "tools": [
                {
                    "name": "get_my_recent_work_orders",
                    "description": "查询我的最近维修工单",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 20,
                                "default": 5,
                            }
                        },
                        "additionalProperties": False,
                    },
                    "policy": _policy(
                        "workorder-server", "get_my_recent_work_orders"
                    ),
                }
            ],
        },
    ],
    "retrieval_policy": {
        "top_k": 5,
        "keyword_weight": 0.3,
        "semantic_weight": 0.7,
        "rrf_k": 60,
        "enable_rerank": False,
        "score_threshold": 0.0,
        "context_threshold": 0.2,
        "semantic_context_threshold": 0.67,
    },
    "model_policy": {
        "version": "s10i-offline-fixture",
        "default": {
            "model_id": "deepseek-v4-flash",
            "provider": "deepseek",
            "model_params": {"use_thinking": True},
        },
        "available": [],
    },
    "price_snapshots": [],
}
RELEASE_ID = "rr_s10i_offline_fixture"


def _prepare_real_runtime_snapshot() -> Any:
    """Index the real seeded document and resolve a real published snapshot."""

    document = db.get_knowledge_doc(1)
    _check(bool(document), "临时SQLite包含真实物业维修服务承诺")
    body = str(document.get("content") or "")
    chunks = rag_chunking.split_text(
        body,
        strategy=document.get("split_strategy") or "auto",
        chunk_size=int(document.get("chunk_size") or 512),
        chunk_overlap=int(document.get("chunk_overlap") or 64),
    )
    _check(
        any(
            re.search(r"5\s*分钟", chunk)
            and re.search(r"30\s*分钟", chunk)
            for chunk in chunks
        ),
        "真实切片包含5分钟登记和30分钟到场",
    )

    # This is the production pgvector boundary, pointed at a disposable Unix
    # socket shared by two --network none containers.
    rag_embeddings._use_fallback = True
    rag_store.init_vector_store()
    _check(rag_indexer.index_document(1, force=True), "真实RAG indexer写入临时pgvector")
    stored_chunks = rag_store.list_chunks_for_doc(1)
    _check(len(stored_chunks) == len(chunks), "真实RAG store保留全部切片", len(stored_chunks))
    _check(
        any(
            re.search(r"5\s*分钟", str(item.get("content") or ""))
            and re.search(r"30\s*分钟", str(item.get("content") or ""))
            for item in stored_chunks
        ),
        "临时pgvector包含目标事实切片",
    )

    config = copy.deepcopy(SNAPSHOT_CONFIG)
    knowledge = config["knowledge"][0]
    document_hash = content_hash(body)
    knowledge.update(
        {
            "document_version": document_hash[:16],
            "document_hash": document_hash,
            "chunk_count": len(chunks),
            "chunk_size": int(document.get("chunk_size") or 512),
            "chunk_overlap": int(document.get("chunk_overlap") or 64),
            "split_strategy": document.get("split_strategy") or "auto",
            "chunk_snapshots": [
                {
                    "chunk_index": index,
                    "content": chunk,
                    "chunk_hash": content_hash(chunk),
                }
                for index, chunk in enumerate(chunks)
            ],
        }
    )
    config_hash = content_hash(config)
    db.create_runtime_release(
        release_id=RELEASE_ID,
        version=1,
        config_hash=config_hash,
        config=config,
        validation={
            "valid": True,
            "errors": [],
            "warnings": [],
            "counts": {
                "agents": len(config.get("agents") or []),
                "skills": len(config.get("skills") or []),
                "knowledge_docs": len(config.get("knowledge") or []),
                "mcp_servers": len(config.get("mcp_servers") or []),
            },
        },
        created_by="s10i-offline-test",
    )
    published = db.publish_runtime_release(RELEASE_ID)
    _check(published.get("status") == "published", "真实RuntimeRelease已发布到临时指针")
    snapshot = resolve_snapshot(SESSION_ID)
    _check(snapshot.release_id == RELEASE_ID, "真实snapshot_resolver固定发布版本")
    _check(snapshot.snapshot_hash == config_hash, "真实snapshot_resolver保留配置哈希")
    persisted = db.get_run_config_snapshot(SESSION_ID)
    _check(
        bool(persisted and persisted.get("snapshot_id") == snapshot.snapshot_id),
        "真实RunConfigSnapshot已落临时SQLite",
    )
    return snapshot


class _DeterministicProviderModel(Model):
    """Lowest-level offline Provider output; the Agno Agent remains real."""

    calls: Dict[str, int] = {"router": 0, "vertical": 0}
    vertical_prompt: str = ""

    def __init__(self, model_id: str):
        super().__init__(
            id=model_id,
            name=f"offline-{model_id}",
            provider="s10i-offline-provider-output",
            retries=0,
        )

    @staticmethod
    def _message_text(kwargs: Dict[str, Any]) -> str:
        return "\n".join(
            str(getattr(message, "content", "") or "")
            for message in kwargs.get("messages") or []
        )

    @classmethod
    def _vertical_response(cls, prompt: str) -> str:
        cls.vertical_prompt = prompt
        blocks = re.finditer(
            r"- (?P<evidence_id>ev_[^\s|]+)\s*\|\s*物业维修服务承诺\s*\|"
            r"[^\n]*\n(?P<content>.*?)(?=\n- ev_|\n\[只读Tool证据\]|\Z)",
            prompt,
            flags=re.S,
        )
        target = next(
            (
                match
                for match in blocks
                if re.search(r"5\s*分钟", match.group("content"))
                and re.search(r"30\s*分钟", match.group("content"))
            ),
            None,
        )
        if target is None:
            raise AssertionError("real Agno Agent did not receive the target RAG evidence")
        evidence_id = target.group("evidence_id")
        _check("物业客服中心" in target.group("content"), "Agno Agent可见真实RAG快照")

        expected_tool_facts = {
            "$.data.temperature_c": "29℃",
            "$.data.condition": "多云",
            "$.data.humidity_pct": "70%",
            "$.data.wind_direction": "东风",
            "$.data.wind_level": "3级",
            "$.data.items[0].id": WORK_ORDER_ID,
            "$.data.items[0].status": WORK_ORDER_STATUS,
        }
        for json_path, display_value in expected_tool_facts.items():
            _check(
                json_path in prompt and display_value in prompt,
                f"Agno Agent可见ToolEvidence：{json_path}={display_value}",
            )

        return (
            "根据《物业维修服务承诺》，物业客服中心在5分钟内完成工单登记并通知工程人员，"
            f"工程人员30分钟内到场处置。[[evidence:{evidence_id}]]\n\n"
            "上海天气为29℃、多云，湿度70%，东风3级；"
            "演示固定样例，不代表真实实时天气。\n\n"
            f"最近维修工单为{WORK_ORDER_ID}，状态为{WORK_ORDER_STATUS}。"
        )

    @classmethod
    def _response(cls, kwargs: Dict[str, Any]) -> ModelResponse:
        prompt = cls._message_text(kwargs)
        if "decision_schema" in prompt and "A_SAFETY_HANDOFF" in prompt:
            cls.calls["router"] += 1
            content = json.dumps(
                {
                    "lane": "B_PROPERTY_GOVERNED",
                    "business_intent": "readonly_property_evidence",
                    "reason": "用户需要物业制度与两项只读业务查询。",
                },
                ensure_ascii=False,
            )
        elif "物业维修服务承诺" in prompt and "[只读Tool证据]" in prompt:
            cls.calls["vertical"] += 1
            content = cls._vertical_response(prompt)
        else:
            raise AssertionError(f"unexpected Agno Provider prompt: {prompt[:500]}")
        return ModelResponse(role="assistant", content=content)

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        del args
        return self._response(kwargs)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        del args
        return self._response(kwargs)

    def invoke_stream(self, *args: Any, **kwargs: Any):
        del args
        yield self._response(kwargs)

    async def ainvoke_stream(self, *args: Any, **kwargs: Any):
        del args
        yield self._response(kwargs)

    def _parse_provider_response(
        self,
        response: Any,
        response_format: Any = None,
    ) -> ModelResponse:
        del response_format
        return response if isinstance(response, ModelResponse) else ModelResponse(content=response)

    def _parse_provider_response_delta(self, response_delta: Any) -> ModelResponse:
        return (
            response_delta
            if isinstance(response_delta, ModelResponse)
            else ModelResponse(content=response_delta)
        )


def _build_offline_provider_model(model_id: str, **_kwargs: Any) -> Model:
    return _DeterministicProviderModel(model_id)


class _FixtureFunction:
    def __init__(self, entrypoint):
        self.entrypoint = entrypoint


class _FixtureMCPTools:
    """MCP transport fixture; production planning, policy and parsing stay real."""

    opened: List[str] = []
    closed: List[str] = []

    def __init__(self, **kwargs: Any):
        self.name = str(kwargs.get("name") or "")
        self.functions: Dict[str, Any] = {}

    async def __aenter__(self):
        self.__class__.opened.append(self.name)

        async def weather(**arguments: Any):
            _check(arguments == {"city": "上海"}, "天气Tool参数由真实Planner生成")
            return {
                "structured_content": {
                    "status": "success",
                    "data": {
                        "city": "上海",
                        "temperature_c": 29,
                        "condition": "多云",
                        "humidity_pct": 70,
                        "wind_direction": "东风",
                        "wind_level": 3,
                        "notice": "演示固定样例，不代表真实实时天气",
                    },
                }
            }

        async def recent_orders(**arguments: Any):
            _check(arguments == {"limit": 5}, "工单Tool参数由真实Planner生成")
            return {
                "structured_content": {
                    "status": "success",
                    "data": {
                        "items": [
                            {
                                "id": WORK_ORDER_ID,
                                "status": WORK_ORDER_STATUS,
                            }
                        ]
                    },
                }
            }

        if self.name == "weather-server":
            self.functions = {
                "get_current_weather": _FixtureFunction(weather),
            }
        elif self.name == "workorder-server":
            self.functions = {
                "get_my_recent_work_orders": _FixtureFunction(recent_orders),
            }
        else:
            raise AssertionError(f"unexpected MCP server: {self.name}")
        return self

    async def close(self):
        self.__class__.closed.append(self.name)


def _parse_sse(frame: str) -> Dict[str, Any]:
    event = "message"
    data_lines: List[str] = []
    for line in str(frame).splitlines():
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())
    if not data_lines:
        raise AssertionError(f"SSE frame has no data: {frame!r}")
    payload = json.loads("\n".join(data_lines))
    if not isinstance(payload, dict):
        raise AssertionError(f"SSE payload is not an object: {payload!r}")
    return {"event": event, "data": payload}


def _tool_invocation(
    result: Any,
    *,
    invocation_id: str,
    arguments: Optional[Dict[str, Any]] = None,
    result_contract: Optional[Dict[str, Any]] = None,
    server_name: str = "fixture-server",
    tool_name: str = "fixture-tool",
) -> tuple[ToolInvocation, str]:
    business_status, summary, tool_evidence = _evaluate_read_tool_result(
        result,
        result_contract or {"success_statuses": ["success"]},
        invocation_id=invocation_id,
        server_name=server_name,
        tool_name=tool_name,
    )
    invocation = ToolInvocation(
        invocation_id=invocation_id,
        server_name=server_name,
        tool_name=tool_name,
        effect=ToolEffect.READ,
        arguments=dict(arguments or {}),
        discovery_status="success",
        transport_status="success",
        invocation_status="success",
        business_status=business_status,
        result_summary=summary,
        tool_evidence=tool_evidence,
    )
    return invocation, summary


def _validate_answer_with_tool(
    answer: str,
    invocation: ToolInvocation,
) -> tuple[List[Dict[str, Any]], Any]:
    bundle = build_run_evidence_bundle(
        EvidenceSet(query="只读工具事实核验", retrieval_status="not_requested"),
        tool_invocations=[invocation],
    )
    _, _, violations, final_bundle = render_bundle_citations(answer, bundle)
    return violations, final_bundle


def test_tool_evidence_behavior() -> None:
    humidity, _ = _tool_invocation(
        {"structured_content": {"status": "success", "data": {"humidity_pct": 70}}},
        invocation_id="tool-humidity-70",
        arguments={"threshold": 70},
        server_name="weather-fixture",
        tool_name="weather-read",
    )
    _check(humidity.tool_evidence is not None, "成功结果生成ToolEvidence")
    violations, final_bundle = _validate_answer_with_tool("当前湿度70%。", humidity)
    _check(not violations, "humidity_pct=70支持70%", violations)
    _check(len(final_bundle.tool_evidence_links) == 1, "合法Tool事实形成证据链接")

    for answer in ("预计70分钟。", "费用70元。", "当前湿度71%。"):
        violations, _ = _validate_answer_with_tool(answer, humidity)
        _check(bool(violations), f"湿度事实不支持越单位或改值：{answer}", violations)

    result_65, _ = _tool_invocation(
        {"structured_content": {"status": "success", "data": {"humidity_pct": 65}}},
        invocation_id="tool-result-65",
        arguments={"humidity_pct": 70},
    )
    violations, _ = _validate_answer_with_tool("当前湿度65%。", result_65)
    _check(not violations, "结果65支持65%", violations)
    violations, _ = _validate_answer_with_tool("当前湿度70%。", result_65)
    _check(bool(violations), "请求参数70不能冒充结果证据", violations)

    for status in ("failed", "not_found", "timeout"):
        failed, _ = _tool_invocation(
            {
                "structured_content": {
                    "status": status,
                    "data": {"humidity_pct": 70},
                }
            },
            invocation_id=f"tool-{status}",
        )
        _check(failed.tool_evidence is None, f"{status}不生成ToolEvidence")

    long_result, summary = _tool_invocation(
        {
            "structured_content": {
                "status": "success",
                "data": {"padding": "x" * 700, "humidity_pct": 70},
            }
        },
        invocation_id="tool-truncated-summary",
    )
    _check("humidity_pct" not in summary, "页面摘要确实在500字符处截断")
    violations, _ = _validate_answer_with_tool("当前湿度70%。", long_result)
    _check(not violations, "摘要截断不影响结构化事实核验", violations)

    calculator, _ = _tool_invocation(
        {"result": 84},
        invocation_id="tool-calculator-84",
        server_name="calculator-server",
        tool_name="calculate",
    )
    violations, _ = _validate_answer_with_tool("计算结果为84元。", calculator)
    _check(not violations, "calculator result=84兼容既有计算证据", violations)


def _projection_item(index: int, content: str) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"ev_projection_{index}",
        knowledge_id="projection-doc",
        knowledge_version="v1",
        document_id="projection-doc",
        document_version="v1",
        document_hash=content_hash("projection-doc"),
        chunk_id=f"projection-chunk-{index}",
        chunk_index=index,
        chunk_hash=content_hash(content),
        content_snapshot=content,
        retrieval_score=0.9 - index / 100,
        retrieval_mode="fixture",
        title="投影测试文档",
    )


def test_bundle_projection_behavior() -> None:
    items = [
        _projection_item(1, "规则甲要求先登记后核验。"),
        _projection_item(2, "规则乙说明结果必须留痕。"),
        _projection_item(3, "规则丙说明未采用证据不得冒充依据。"),
    ]
    evidence = EvidenceSet(
        items=items,
        query="说明规则甲。",
        retrieval_status="completed",
    )
    bundle = build_run_evidence_bundle(evidence)
    answer = f"规则甲要求先登记后核验。[[evidence:{items[0].evidence_id}]]"
    _, citations, violations, final_bundle = render_bundle_citations(answer, bundle)
    _check(not violations and len(citations) == 1, "Renderer真实采用1条候选")
    projection = project_evidence_for_trace(
        {
            "run_evidence_bundle": final_bundle.model_dump(mode="json"),
            "citation_links": [item.model_dump(mode="json") for item in citations],
        }
    )
    _check(
        projection["counts"]["retrieved_rag_candidates"] == 3
        and projection["counts"]["adopted_rag"] == 1
        and projection["counts"]["unused_rag"] == 2,
        "Ledger投影候选3/采用1/未采用2",
        projection["counts"],
    )

    blocked_answer = answer + "\n当前湿度71%。"
    _, blocked_citations, blocked_violations, blocked_bundle = (
        render_bundle_citations(blocked_answer, bundle)
    )
    _check(bool(blocked_violations), "安全门禁真实拦截无依据数值")
    blocked_projection = project_evidence_for_trace(
        {
            "run_evidence_bundle": blocked_bundle.model_dump(mode="json"),
            "citation_links": [
                item.model_dump(mode="json") for item in blocked_citations
            ],
            "contract_violations": blocked_violations,
        }
    )
    _check(
        blocked_projection["counts"]["adopted_rag"] == 0
        and blocked_projection["counts"]["unused_rag"] == 3,
        "交付被拦截时全部候选仍显示为未采用",
        blocked_projection["counts"],
    )


async def test_real_coordinator_evidence_chain() -> None:
    baseline_attempts = _provider_attempt_count()
    _check(baseline_attempts == 0, "临时账本Provider attempt基线为0")

    snapshot = _prepare_real_runtime_snapshot()
    _DeterministicProviderModel.calls = {"router": 0, "vertical": 0}
    _DeterministicProviderModel.vertical_prompt = ""
    _FixtureMCPTools.opened = []
    _FixtureMCPTools.closed = []
    events: List[Dict[str, Any]] = []
    with patch.object(
        coordinator_module,
        "build_model",
        side_effect=_build_offline_provider_model,
    ), patch.object(
        agent_factory_module,
        "build_model",
        side_effect=_build_offline_provider_model,
    ), patch.object(
        mcp_executor_module, "MCPTools", _FixtureMCPTools
    ):
        assembly = agent_factory_module.build_agent_from_snapshot(
            snapshot,
            "maintenance",
            QUERY,
        )
        _check(isinstance(assembly.agent, AgnoAgent), "真实AgentFactory装配Agno Agent")
        _check(
            {item.skill_id for item in assembly.activated_skills} == {8},
            "真实AgentFactory预装载Skill 8",
        )
        async for frame in RuntimeCoordinator().stream(
            QUERY,
            SESSION_ID,
            USER_ID,
        ):
            events.append(_parse_sse(frame))

    _check(bool(events), "真实RuntimeCoordinator产生SSE事件")
    _check(events[0]["event"] == "start", "SSE以start开始")
    _check(events[-1]["event"] == "done", "SSE以done终态结束")
    done = events[-1]["data"]
    _check(done.get("status") == "complete", "组合链路done.status=complete", done)
    _check(done.get("runtime_path") == "consultation", "组合链路保持只读咨询路径")

    lane = next(item["data"] for item in events if item["event"] == "lane")
    route = next(item["data"] for item in events if item["event"] == "route")
    _check(lane.get("lane") == "B_PROPERTY_GOVERNED", "真实链路进入B路", lane)
    _check(route.get("current_agent_id") == "maintenance", "真实链路选择维修Agent", route)

    final = next(item["data"] for item in events if item["event"] == "final")
    answer = str(final.get("content") or "")
    for expected in (
        "5分钟",
        "30分钟",
        "29℃",
        "多云",
        "湿度70%",
        "东风3级",
        "演示固定样例，不代表真实实时天气",
        WORK_ORDER_ID,
        WORK_ORDER_STATUS,
    ):
        _check(expected in answer, f"最终回答保留：{expected}", answer)
    _check(
        not answer.startswith(KNOWLEDGE_INSUFFICIENT_RESPONSE),
        "组合链路不是知识不足兜底",
    )
    _check(len(done.get("citations") or []) == 1, "最终回答采用真实RAG引用")

    skill_ids = {
        int(item.get("skill_id"))
        for item in done.get("activated_skills") or []
        if item.get("skill_id") is not None
    }
    _check(skill_ids == {8}, "真实AgentFactory激活Skill 8", skill_ids)

    mcp_calls = done.get("mcp_calls") or []
    _check(len(mcp_calls) == 2, "真实ToolPlanner只执行两只读Tool", mcp_calls)
    _check(
        {
            (item.get("server_name"), item.get("tool_name"))
            for item in mcp_calls
        }
        == {
            ("weather-server", "get_current_weather"),
            ("workorder-server", "get_my_recent_work_orders"),
        },
        "两只读Tool名称与发布绑定一致",
        mcp_calls,
    )
    _check(
        all(
            item.get("effect") == "read"
            and item.get("invocation_status") == "success"
            and item.get("business_status") == "success"
            and item.get("tool_evidence")
            for item in mcp_calls
        ),
        "两只读Tool均成功并冻结ToolEvidence",
        mcp_calls,
    )
    _check(
        sorted(_FixtureMCPTools.opened)
        == sorted(_FixtureMCPTools.closed)
        == ["weather-server", "workorder-server"],
        "MCP传输fixture完整打开并关闭",
    )

    decision_summary = done.get("decision_summary") or {}
    _check(decision_summary.get("agent", {}).get("status") == "selected", "Agent判定已选择")
    _check(decision_summary.get("skill", {}).get("status") == "selected", "Skill判定已选择")
    _check(decision_summary.get("rag", {}).get("status") == "selected", "RAG判定已选择")
    _check(
        decision_summary.get("rag", {}).get("retrieval_status") == "completed",
        "RAG使用真实检索结果而非Snapshot兜底",
        decision_summary.get("rag"),
    )
    _check(decision_summary.get("tool", {}).get("status") == "selected", "Tool判定已选择")
    _check(decision_summary.get("handoff", {}).get("status") == "skipped", "Handoff未触发")

    trace_id = str(done.get("trace_id") or "")
    stored = db.get_evidence_ledger(trace_id)
    _check(bool(stored), "真实EvidenceLedger已持久化", trace_id)
    ledger = dict(stored["ledger"])
    _check(ledger.get("contract_violations") == [], "组合链路contract_violations=0", ledger.get("contract_violations"))
    retrieved = list(ledger.get("retrieval_evidence") or [])
    _check(bool(retrieved), "Ledger保留真实RAG候选")
    target_candidate = next(
        (
            item
            for item in retrieved
            if re.search(r"5\s*分钟", str(item.get("content_snapshot") or ""))
            and re.search(r"30\s*分钟", str(item.get("content_snapshot") or ""))
        ),
        None,
    )
    _check(bool(target_candidate), "真实检索合并召回5/30分钟目标分片")
    _check(bool(target_candidate.get("subquery")), "目标候选保留subquery")
    named_scope = target_candidate.get("named_document_scope") or {}
    _check(
        named_scope.get("matched") is True
        and any(
            int(item.get("document_id") or -1) == 1
            for item in named_scope.get("documents") or []
        ),
        "目标候选保留命名文档作用域",
        named_scope,
    )
    _check(
        target_candidate.get("retrieval_path") == "named_document_clause_hybrid"
        and any(
            match.get("retrieval_path") == "named_document_clause_hybrid"
            and set(match.get("retrieval_sources") or [])
            & {"keyword", "semantic"}
            for match in target_candidate.get("retrieval_matches") or []
        ),
        "目标候选来自真实混合检索而非Snapshot兜底",
        target_candidate.get("retrieval_matches"),
    )
    bundle = ledger.get("run_evidence_bundle") or {}
    skill_evidence = list(bundle.get("skill_evidence") or [])
    _check(
        [int(item.get("skill_id")) for item in skill_evidence] == [8],
        "Bundle冻结Skill 8证据",
        skill_evidence,
    )
    skill_body = str(skill_evidence[0].get("content_snapshot") or "")
    expected_preloaded_body = f"# 维修响应说明\n\n{SKILL_INSTRUCTIONS}"
    _check(
        skill_body.strip() == expected_preloaded_body,
        "Bundle Skill content_snapshot等于Agno预加载正文",
        {"bundle": skill_body, "expected": expected_preloaded_body},
    )
    _check(
        skill_body in _DeterministicProviderModel.vertical_prompt,
        "Bundle Skill正文与Agno Agent实际预加载正文一致",
        {
            "skill_body": skill_body,
            "agent_prompt_has_body": (
                skill_body in _DeterministicProviderModel.vertical_prompt
            ),
        },
    )
    _check(len(bundle.get("tool_evidence") or []) == 2, "Ledger Bundle保留两条ToolEvidence")
    _check(len(bundle.get("tool_evidence_links") or []) == 2, "最终回答链接两条ToolEvidence")
    cited_id = str((done.get("citations") or [])[0].get("evidence_id") or "")
    _check(
        bundle.get("validated_rag_evidence_ids") == [cited_id],
        "Bundle冻结经校验RAG evidence_id",
        bundle.get("validated_rag_evidence_ids"),
    )
    linked_tool_ids = [
        str(item.get("evidence_id") or "")
        for item in bundle.get("tool_evidence_links") or []
    ]
    _check(
        bundle.get("delivered_evidence_ids") == [cited_id, *linked_tool_ids],
        "Bundle冻结最终交付的RAG与Tool evidence_id",
        bundle.get("delivered_evidence_ids"),
    )
    _check(bundle.get("violations") == [], "Bundle无证据合同违规", bundle.get("violations"))

    projection = project_evidence_for_trace(ledger)
    _check(
        projection["counts"]["retrieved_rag_candidates"] == len(retrieved)
        and projection["counts"]["adopted_rag"] == 1
        and projection["counts"]["unused_rag"] == len(retrieved) - 1,
        "真实Trace投影准确区分采用与未采用RAG候选",
        projection["counts"],
    )
    _check(
        projection["counts"]["tool_evidence_links"] == 2,
        "真实Trace投影展示两条Tool证据链接",
        projection["counts"],
    )

    _check(ledger.get("action_proposals") == [], "未生成Proposal")
    _check(ledger.get("action_receipts") == [], "未生成Receipt")
    _check(ledger.get("handoff_events") == [], "未产生Handoff事件")
    _check(db.get_work_order_draft(SESSION_ID) is None, "未生成维修Draft")
    session = db.get_chat_session(SESSION_ID) or {}
    _check(session.get("handoff_status") == "none", "会话未转人工", session)
    no_write_eval = next(
        (
            item
            for item in ledger.get("evaluation_results") or []
            if item.get("case") == "consultation_no_write"
        ),
        None,
    )
    _check(bool(no_write_eval and no_write_eval.get("passed")), "只读无写入合同通过", no_write_eval)

    final_attempts = _provider_attempt_count()
    _check(final_attempts == baseline_attempts, "组合链路Provider attempt增量0")
    _check(
        _DeterministicProviderModel.calls == {"router": 1, "vertical": 1},
        "仅最低层本地Provider输出各执行一次",
    )


def main() -> None:
    try:
        _check(
            all(not os.environ.get(key) for key in (
                "DEEPSEEK_API_KEY",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "GOOGLE_API_KEY",
                "GEMINI_API_KEY",
                "PARALLEL_API_KEY",
            )),
            "Provider Key全部为空",
        )
        test_tool_evidence_behavior()
        test_bundle_projection_behavior()
        asyncio.run(test_real_coordinator_evidence_chain())
        print("S10-I runtime evidence chain passed: Provider attempts 0")
    finally:
        _TEMP_ROOT.cleanup()


if __name__ == "__main__":
    main()
