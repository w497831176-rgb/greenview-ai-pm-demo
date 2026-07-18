"""Deterministic human-copilot policy for the owner chat.

This module deliberately does not call a model.  A human takeover changes who
is accountable for the next action, so the decision must be explainable,
stable, and auditable instead of depending on a model's wording.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


HANDOFF_STATUS_LABELS = {
    "none": "AI 正常服务",
    "requested": "等待人工领取",
    "active": "人工处理中",
    "waiting_user": "等待业主补充",
    "resolved": "人工已给出处理结果",
    "closed": "人工协同已关闭",
    "cancelled": "人工协同已取消",
}

# The state model is intentionally small.  It describes responsibility, not a
# generic ticket workflow.
HANDOFF_TRANSITIONS = {
    "none": {"requested"},
    "requested": {"active", "cancelled"},
    "active": {"waiting_user", "resolved"},
    "waiting_user": {"active", "cancelled"},
    "resolved": {"closed", "active"},
    "closed": {"requested"},
    "cancelled": {"requested"},
}


def is_transition_allowed(current: str, target: str) -> bool:
    """Return whether a responsibility-state transition is legal."""
    return target in HANDOFF_TRANSITIONS.get(current or "none", set())


def _contains_any(text: str, terms: Iterable[str]) -> List[str]:
    return [term for term in terms if term in text]


def evaluate_handoff_policy(
    message: str,
    *,
    mcp_calls: Optional[List[Dict[str, Any]]] = None,
    explicit_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Return an explainable collaboration decision for one owner message.

    L0: AI can answer directly.
    L1: AI prepares a draft and must obtain the owner's confirmation before a
        state-changing action (the work-order workflow implements this).
    L2: AI may prepare facts, but should not make the final business decision.
    L3: a human must take accountability now; the service creates a handoff.

    L2 is intentionally a recommendation rather than an automatic transfer:
    the current demo has no identity/authorisation system, so it must not
    pretend to execute financial or permission decisions on a person's behalf.
    """
    text = (message or "").strip().lower()
    if explicit_reason:
        return {
            "level": "L3",
            "reason_code": "owner_requested",
            "reason": explicit_reason,
            "queue": "property_service",
            "should_request_handoff": True,
            "human_task": "由工作人员接管后确认诉求、给出处理结论并记录结果。",
            "matched_signals": ["owner_requested"],
        }

    explicit_terms = ["转人工", "人工客服", "找人工", "人工处理", "我要人工", "人工介入", "接人工", "人工服务"]
    matched = _contains_any(text, explicit_terms)
    if matched:
        return {
            "level": "L3",
            "reason_code": "owner_requested",
            "reason": "业主明确要求人工服务",
            "queue": "property_service",
            "should_request_handoff": True,
            "human_task": "由工作人员接管后确认诉求、给出处理结论并记录结果。",
            "matched_signals": matched,
        }

    safety_terms = ["燃气泄漏", "煤气泄漏", "起火", "火灾", "触电", "电梯困人", "人身受伤", "有人受伤"]
    matched = _contains_any(text, safety_terms)
    if matched:
        return {
            "level": "L3",
            "reason_code": "safety_emergency",
            "reason": "检测到人身或公共安全风险，需要人工承担处置责任",
            "queue": "emergency_dispatch",
            "should_request_handoff": True,
            "human_task": "核实现场风险、通知应急责任人，并在会话中记录实际处置结果。",
            "matched_signals": matched,
        }

    escalation_terms = ["投诉到住建", "投诉到媒体", "起诉", "律师函", "仲裁", "报警", "曝光"]
    matched = _contains_any(text, escalation_terms)
    if matched:
        return {
            "level": "L3",
            "reason_code": "dispute_escalation",
            "reason": "检测到争议升级或外部投诉风险，需要人工确认口径与处理边界",
            "queue": "customer_relations",
            "should_request_handoff": True,
            "human_task": "核实事实与授权边界，确定正式回复和后续处理人。",
            "matched_signals": matched,
        }

    decision_terms = ["退款", "赔偿", "减免", "退费", "重复扣费", "多扣", "开门权限", "门禁权限", "个人信息", "身份证", "银行卡"]
    matched = _contains_any(text, decision_terms)
    if matched:
        return {
            "level": "L2",
            "reason_code": "human_approval_required",
            "reason": "涉及资金、权限或个人信息，AI 只能整理事实，不能代替人工作出决定",
            "queue": "property_service",
            "should_request_handoff": False,
            "human_task": "在人工核验身份、规则与授权后作出决定；AI 可提供事实和草稿。",
            "matched_signals": matched,
        }

    failed_calls = [
        call for call in (mcp_calls or [])
        if str(call.get("status") or "").lower() in {"failed", "error"}
    ]
    if failed_calls:
        return {
            "level": "L2",
            "reason_code": "tool_failure",
            "reason": "关键工具调用失败，AI 不应把未核验结果表述为事实",
            "queue": "property_service",
            "should_request_handoff": False,
            "human_task": "人工核验缺失信息；如需接管，由业主或工作人员明确发起。",
            "matched_signals": [str(call.get("tool_name") or "unknown_tool") for call in failed_calls],
        }

    confirmation_terms = ["创建工单", "提交报修", "确认创建", "确认报修"]
    matched = _contains_any(text, confirmation_terms)
    if matched:
        return {
            "level": "L1",
            "reason_code": "owner_confirmation_required",
            "reason": "创建工单会写入业务数据，先由 AI 收集并展示草稿，再取得业主确认",
            "queue": "maintenance",
            "should_request_handoff": False,
            "human_task": "无需人工接管；等待业主确认后由受控工单流程执行。",
            "matched_signals": matched,
        }

    return {
        "level": "L0",
        "reason_code": "ai_direct",
        "reason": "低风险信息咨询，可由 AI 在证据与工具边界内直接处理",
        "queue": None,
        "should_request_handoff": False,
        "human_task": "无需人工接管。",
        "matched_signals": [],
    }


def handoff_policy_summary(policy: Dict[str, Any]) -> str:
    """Short, UI-safe explanation stored in the handoff package."""
    return f"{policy.get('level', 'L0')} · {policy.get('reason') or '未提供原因'}"
