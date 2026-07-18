"""Fixed V1.8 acceptance matrix used by UI, tests and release evidence."""

from __future__ import annotations

from typing import Any, Dict, List


ACCEPTANCE_CASES: List[Dict[str, Any]] = [
    {
        "case_key": "E2E-00",
        "input": "结合儿童游乐区制度、设备状态和明天天气判断是否巡检；需要时生成草稿，提交前确认。",
        "business_write": "confirmation_only",
        "expected_route": "safety_or_customer_service",
        "expected_skill": "儿童安全 Skill",
        "expected_mcp": ["weather-server:read", "inspection:create_after_confirm"],
        "expected_rag": "儿童游乐区制度的已发布文档版本与分片",
        "trace": "同一父 Trace 含 Router/Skill/RAG/read MCP/HITL/write Receipt/Evaluation",
        "cost": "router 与 vertical_agent 分 stage；Usage 不完整不显示精确金额",
        "cleanup": "删除验收生成的测试巡检/工单记录",
    },
    {
        "case_key": "READ-01",
        "input": "儿童在小区滑梯受伤，应如何处理？",
        "business_write": "forbidden",
        "expected_route": "safety_agent",
        "expected_skill": "儿童安全 Skill",
        "expected_mcp": [],
        "expected_rag": "指定儿童游乐区安全制度 Chunk",
        "trace": "consultation path；无 pending action/receipt",
        "cost": "router 与 vertical_agent 独立 CostEntry",
        "cleanup": "none",
    },
    {
        "case_key": "MCP-R-01",
        "input": "查询明天本小区天气，不创建任何记录。",
        "business_write": "forbidden",
        "expected_route": "customer_service",
        "expected_skill": "optional",
        "expected_mcp": ["weather-server:get_current_weather"],
        "expected_rag": "not_required",
        "trace": "discovery/transport/invocation/business 四状态分离",
        "cost": "Provider Usage 可得性如实显示",
        "cleanup": "none",
    },
    {
        "case_key": "ACTION-01",
        "input": "3号楼漏水，帮我创建维修工单；分别拒绝与确认。",
        "business_write": "confirmation_only",
        "expected_route": "maintenance",
        "expected_skill": "维修流程 Skill 可选",
        "expected_mcp": ["work_order.create via ActionGateway"],
        "expected_rag": "not_required",
        "trace": "Draft/Proposal/Pause/Approval/Receipt；拒绝零写入；重复确认幂等",
        "cost": "not_applicable for deterministic controller",
        "cleanup": "删除测试工单",
    },
    {
        "case_key": "EXT-ASR-01",
        "input": "新增汛期 Agent、Skill、RAG，发布后用固定问题开启新会话。",
        "business_write": "configuration_only",
        "expected_route": "new_flood_agent",
        "expected_skill": "new_flood_skill",
        "expected_mcp": [],
        "expected_rag": "new_flood_document/version/chunk",
        "trace": "新 Release/Snapshot；旧会话 snapshot_hash 不变",
        "cost": "新 Agent 的 model policy 与价格快照可解释",
        "cleanup": "回退上一 RuntimeRelease；保留审计记录",
    },
    {
        "case_key": "EXT-MCP-01",
        "input": "新增并绑定读/写 MCP；先查询，再确认新增巡检记录。",
        "business_write": "confirmation_only",
        "expected_route": "bound_dynamic_agent",
        "expected_skill": "optional",
        "expected_mcp": ["dynamic:read", "dynamic:create via ActionGateway"],
        "expected_rag": "optional",
        "trace": "include_tools 白名单；写调用带 approval/idempotency/receipt",
        "cost": "模型阶段与工具阶段分开；工具无 Token 则 not_applicable",
        "cleanup": "删除测试记录并回退测试 Release",
    },
    {
        "case_key": "COST-01",
        "input": "同一五题集运行 baseline/candidate。",
        "business_write": "forbidden",
        "expected_route": "per_case",
        "expected_skill": "per_case",
        "expected_mcp": "per_case",
        "expected_rag": "per_case",
        "trace": "同题集、同质量门槛、不同 model_policy_version",
        "cost": "仅 provider_reported_complete 可按 price snapshot/formula 得到金额",
        "cleanup": "测试 Trace 可保留",
    },
    {
        "case_key": "FAIL-01",
        "input": "模拟 MCP 超时或模型返回无效 evidence_id。",
        "business_write": "forbidden",
        "expected_route": "safe_degradation",
        "expected_skill": "optional",
        "expected_mcp": ["timeout fixture"],
        "expected_rag": "invalid evidence_id must be rejected",
        "trace": "failed span + contract_violation + evaluation failed",
        "cost": "失败前已有 usage 才记录；否则 unavailable",
        "cleanup": "关闭故障开关",
    },
]


def get_acceptance_case(case_key: str) -> Dict[str, Any]:
    for case in ACCEPTANCE_CASES:
        if case["case_key"] == case_key:
            return case
    raise KeyError(case_key)
