"""Fixed V1.8 acceptance matrix used by UI, tests and release evidence."""

from __future__ import annotations

from typing import Any, Dict, List


ACCEPTANCE_CASES: List[Dict[str, Any]] = [
    {
        "case_key": "HIT-01",
        "input": "只咨询：结合《物业维修服务承诺》和本小区天气，判断明天下午是否适合上门检查漏水。",
        "business_write": "forbidden",
        "expected_route": "maintenance",
        "expected_skill": "维修工单处理",
        "expected_mcp": ["weather-server:get_current_weather", "weather-server:get_weather_advice"],
        "expected_rag": "绑定的《物业维修服务承诺》已发布文档版本与分片",
        "trace": "同一父 Trace 含 Router/垂直 Agent/Skill/RAG/read MCP/Citation/Evaluation/Cost",
        "cost": "router 与 vertical_agent 分 stage；Usage 不完整不显示精确金额",
        "cleanup": "none",
    },
    {
        "case_key": "HANDOFF-01",
        "input": "分别测试 AI 因高风险或强情绪判断转人工，以及业主明确说“转人工”。",
        "business_write": "forbidden",
        "expected_route": "per_scenario",
        "expected_skill": "optional",
        "expected_mcp": [],
        "expected_rag": "not_required",
        "trace": "两种 handoff 触发原因、状态、上下文包、Trace 与 Evaluation 可区分",
        "cost": "触发前模型调用按实际 Usage 记录；人工阶段不伪造 Token",
        "cleanup": "关闭测试转接",
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
        "case_key": "EXT-OFFDOMAIN-01",
        "input": "新增“系外行星观测”垂直 Agent、Skill、读 MCP、写 MCP 与 RAG，显式绑定并发布；用固定问题开启新会话。",
        "business_write": "configuration_only",
        "expected_route": "new_exoplanet_agent",
        "expected_skill": "new_exoplanet_skill",
        "expected_mcp": ["astronomy-server:read", "astronomy-server:create via Proposal"],
        "expected_rag": "GVX-42 文档/version/chunk；Agent scope 在 Top-K 前生效",
        "trace": "新 Release/Snapshot；动态 Router/Skill/ToolPlan/scoped RAG 来自同一快照；旧会话 snapshot_hash 不变",
        "cost": "新 Agent 的 model policy 与价格快照可解释",
        "cleanup": "回退上一 RuntimeRelease；删除测试 Agent、Skill、RAG；保留审计记录",
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
        "case_key": "BADCASE-01",
        "input": "自动制造一条 Evaluation/契约失败，并对另一条真实回答提交业主差评。",
        "business_write": "configuration_review_only",
        "expected_route": "per_source_trace",
        "expected_skill": "root_cause_dependent",
        "expected_mcp": "root_cause_dependent",
        "expected_rag": "root_cause_dependent",
        "trace": "自动/手动两来源关联原 Trace；根因、方案、应用、复测、验证、关闭动作完整",
        "cost": "AI 专家分析和复测模型调用独立计费；确定性状态迁移 not_applicable",
        "cleanup": "关闭或删除测试 Badcase 与临时草稿",
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
