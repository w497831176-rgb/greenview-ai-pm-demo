"""Authoritative, session-scoped repair work-order workflow.

The language model can explain and collect information, but it never decides
whether a real work order has been written.  This module is the control plane:
it keeps a per-session draft, validates the required fields, and writes exactly
once only after an explicit confirmation.
"""

import re
from typing import Any, Dict, List, Optional

from app.runtime.action_gateway import ActionGateway
from db.property_db import (
    delete_work_order_draft,
    get_action_receipt_by_idempotency_key,
    get_latest_action_proposal,
    get_pending_action_proposal,
    get_work_order_draft,
    save_work_order_draft,
)


DEFAULT_ROOM_ID = "3-2-1201"
DEFAULT_OWNER_NAME = "王先生"
ACTION_TYPE = "work_order.create"
action_gateway = ActionGateway()


def _room_id(text: str) -> str:
    match = re.search(r"(\d{1,2})\s*[-#—]\s*(\d{1,2})\s*[-#—]\s*(\d{3,4})", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    match = re.search(r"(\d{1,2})\s*[栋幢号楼]\s*(\d{1,2})\s*[单元]?\s*(\d{3,4})", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return ""


def _phone(text: str) -> str:
    match = re.search(r"1[3-9]\d{9}", text)
    return match.group(0) if match else ""


def _contact_name(text: str) -> str:
    match = re.search(r"(?:我是|联系人[：:]?)\s*([\u4e00-\u9fa5]{1,4}(?:先生|女士)?)", text)
    return match.group(1) if match else ""


def _urgency(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ("紧急", "立刻", "马上", "水漫", "爆管", "漏电", "燃气泄漏", "火灾")):
        return "紧急"
    if any(word in lowered for word in ("高优先", "很严重", "严重", "尽快处理")):
        return "高"
    if any(word in lowered for word in ("中等", "一般", "不急")):
        return "中"
    if any(word in lowered for word in ("低优先", "有空", "不着急")):
        return "低"
    return ""


def _appointment(text: str, urgency: str) -> str:
    if urgency == "紧急":
        return "尽快（紧急报修）"
    match = re.search(r"(?:今天|明天|后天|今晚|上午|下午|傍晚|周[一二三四五六日末天]|\d{1,2}月\d{1,2}日).{0,12}(?:上门|维修|处理)?", text)
    if match:
        return match.group(0).strip()
    if any(word in text for word in ("尽快", "稍后", "有空时")):
        return "尽快"
    return ""


def _issue_type(text: str) -> str:
    if any(word in text for word in ("水", "漏", "渗", "滴", "管", "下水道", "马桶", "龙头", "水槽")):
        return "水电"
    if any(word in text for word in ("电", "灯", "跳闸", "插座", "开关", "线路")):
        return "水电"
    if any(word in text for word in ("电梯", "梯")):
        return "公区"
    if any(word in text for word in ("门", "窗", "玻璃", "锁")):
        return "门窗"
    return "其他"


def _issue_description(text: str) -> str:
    """Extract the reported problem, never the bare create command itself."""

    candidate = (text or "").strip()
    candidate = re.sub(
        r"^(?:我要|我想|请|帮我|麻烦|需要)?"
        r"(?:马上|现在|直接|尽快)?"
        r"(?:报修|创建(?:一张|一个)?(?:维修)?工单|提交(?:一张|一个)?(?:维修)?工单)"
        r"[\s，,。:：-]*",
        "",
        candidate,
        count=1,
        flags=re.IGNORECASE,
    ).strip()
    if not candidate or candidate in {
        "工单",
        "维修工单",
        "帮我创建工单",
        "创建工单",
        "提交工单",
        "我要报修",
    }:
        return ""
    # A field-only follow-up must not accidentally become the fault
    # description while the workflow is collecting missing values.
    compact = re.sub(r"\s+", "", candidate)
    phone = _phone(candidate)
    room = _room_id(candidate)
    if (
        (phone and compact in {phone, f"电话{phone}", f"联系电话{phone}"})
        or (room and compact in {room, f"房号{room}", f"我是{room}"})
        or re.fullmatch(r"(?:紧急|高|中|低)", compact)
        or re.fullmatch(
            r"(?:尽快|今天|明天|后天|今晚|上午|下午|晚上|"
            r"\d{1,2}月\d{1,2}日)(?:上门|维修|处理)?",
            compact,
        )
    ):
        return ""
    return candidate


def is_explicit_work_order_request(message: str) -> bool:
    """Return True only for an affirmative *command* to start a repair draft.

    A repair-related noun is not authority to start a state-changing workflow.
    Questions such as "报修前要准备什么" and explicit instructions such as
    "本轮不要创建工单" must remain normal Agent/RAG/MCP conversations.
    """
    text = (message or "").strip()
    compact = re.sub(r"\s+", "", text)
    if not compact or _has_creation_negation(compact):
        return False

    affirmative_patterns = (
        r"(?:我要|我想|请|帮我|麻烦|需要)(?:马上|现在|直接|尽快)?(?:报修|创建(?:维修)?工单|提交(?:维修)?工单|安排(?:师傅|维修|上门)|派师傅|上门维修|修一下)",
        r"(?:创建|提交)(?:一张|一个)?(?:维修)?工单",
        r"(?:请|帮我)安排(?:师傅|维修|上门)",
    )
    return any(re.search(pattern, compact, flags=re.IGNORECASE) for pattern in affirmative_patterns)


def _has_creation_negation(text: str) -> bool:
    """Recognise explicit non-creation intent before the draft controller runs."""
    return bool(re.search(
        r"(?:不要|不需要|无需|暂不|先不|仅|只)(?:马上|现在|本轮|先)?(?:创建|提交|生成|新建)?(?:真实|正式)?(?:维修)?(?:工单|报修)?",
        text,
        flags=re.IGNORECASE,
    )) and any(token in text for token in (
        "不要创建", "不创建", "不要新建", "不新建", "暂不", "先不",
        "无需", "不需要", "仅咨询", "只咨询", "仅查询", "只查询",
    ))


def is_cancel_request(message: str) -> bool:
    normalized = (message or "").strip()
    compact = re.sub(r"[\s，。！？、,.!?；;：:]+", "", normalized)
    if any(phrase in compact for phrase in ("不要取消", "别取消")):
        return False
    return any(word in compact for word in (
        "取消报修",
        "取消工单草稿",
        "取消这个工单",
        "取消该工单",
        "不报修了",
        "先不用报修",
        "拒绝创建",
        "拒绝提交",
        "不要创建",
        "不要提交",
        "别创建",
        "别提交",
        "不创建了",
        "不提交了",
    )) or compact in {"拒绝", "取消", "不提交", "不创建"}


def is_confirmation(message: str) -> bool:
    normalized = (message or "").strip().lower()
    return bool(re.search(r"^(确认创建|确认提交|确认|同意创建|好的.*创建|就.*创建|创建.*)$", normalized))


def _is_draft_follow_up(
    message: str,
    existing: Optional[Dict[str, Any]] = None,
) -> bool:
    """Only let concise field-completion messages advance an existing draft.

    A stale draft must never hijack a later request for weather, RAG evidence,
    work-order enquiry or an unrelated consultation in the same session.
    """
    text = (message or "").strip()
    compact = re.sub(r"\s+", "", text)
    if not text or _has_creation_negation(compact):
        return False
    if is_confirmation(text) or is_cancel_request(text) or is_explicit_work_order_request(text):
        return True
    if len(text) > 90:
        return False
    # These are concrete fields the draft needs; generic "漏水" or "报修"
    # alone intentionally do not continue the workflow.
    appointment_follow_up = bool(re.search(
        r"(?:预约|上门时间|可上门|今天(?:上午|下午|晚上)|明天(?:上午|下午|晚上)|\d{1,2}月\d{1,2}日)",
        text,
    ))
    # A bare priority such as "紧急" is a valid field answer.  Do not treat a
    # normal sentence merely mentioning an urgent repair as draft completion.
    explicit_urgency = bool(re.fullmatch(r"(?:紧急|高|中|低)", compact))
    room_follow_up = len(compact) <= 24 and bool(_room_id(text))
    contact_follow_up = "联系人" in text and bool(_contact_name(text))
    issue_follow_up = bool(
        existing
        and not str(existing.get("issue_desc") or "").strip()
        and _issue_description(text)
    )
    return bool(
        _phone(text)
        or room_follow_up
        or contact_follow_up
        or appointment_follow_up
        or explicit_urgency
        or issue_follow_up
    )


def _missing(draft: Dict[str, Any]) -> List[str]:
    labels = {
        "issue_desc": "维修问题描述",
        "urgency": "紧急程度",
        "contact_phone": "联系电话",
        "appointment_time": "预约上门时间",
    }
    return [label for key, label in labels.items() if not str(draft.get(key) or "").strip()]


def _summary(draft: Dict[str, Any]) -> str:
    return (
        f"房号：{draft.get('room_id') or DEFAULT_ROOM_ID}\n"
        f"问题类型：{draft.get('issue_type') or '其他'}\n"
        f"问题描述：{draft.get('issue_desc') or '未提供'}\n"
        f"紧急程度：{draft.get('urgency') or '未提供'}\n"
        f"联系人：{draft.get('contact_name') or DEFAULT_OWNER_NAME}\n"
        f"联系电话：{draft.get('contact_phone') or '未提供'}\n"
        f"预约上门：{draft.get('appointment_time') or '未提供'}"
    )


def _result(action: str, reply: str, draft: Optional[Dict[str, Any]], **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "handled": True,
        "action": action,
        "reply": reply,
        "draft": draft or {},
        "missing_fields": _missing(draft or {}),
        "route_reason": "报修会话处于维修工单流程，交由维修 Agent 持续处理。",
    }
    payload.update(extra)
    return payload


def advance_work_order_workflow(
    session_id: str,
    message: str,
    trace_id: Optional[str] = None,
    release_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Advance a repair draft without relying on a model tool call.

    Returns ``None`` for ordinary chat.  Every non-None result is authoritative
    and can be safely rendered by the chat API without claiming a fictitious
    work order.
    """
    existing = get_work_order_draft(session_id)
    pending = get_pending_action_proposal(session_id, ACTION_TYPE)

    if (existing or pending) and is_cancel_request(message):
        if pending:
            action_gateway.reject(
                pending["proposal_id"],
                actor=f"owner:{session_id}",
                comment="用户取消待确认工单",
            )
        delete_work_order_draft(session_id)
        return _result(
            "rejected",
            "已取消本次待确认的报修草稿，未创建正式工单。需要时您可以重新告诉我报修问题。",
            {},
            proposal_id=(pending or {}).get("proposal_id"),
        )

    if not existing:
        if is_confirmation(message):
            latest = get_latest_action_proposal(session_id, ACTION_TYPE)
            if latest and latest.get("status") == "committed":
                receipt = get_action_receipt_by_idempotency_key(latest["idempotency_key"]) or {}
                actual_id = receipt.get("resource_id")
                return _result(
                    "idempotent_replay",
                    f"该工单已提交成功，工单号：{actual_id}。本次重复确认未再次写库。",
                    {},
                    proposal_id=latest.get("proposal_id"),
                    receipt=receipt,
                    work_order_id=actual_id,
                )
        if not is_explicit_work_order_request(message):
            return None
    elif not _is_draft_follow_up(message, existing):
        # Preserve the unconfirmed draft, but do not turn every subsequent
        # consultation into a stateful ticket interaction.
        return None

    if existing and is_confirmation(message):
        missing = _missing(existing)
        if missing:
            return _result(
                "confirmation_blocked",
                "暂不能创建正式工单，因为还缺少：" + "、".join(missing) + "。请补充后再确认创建。",
                existing,
            )
        proposal = action_gateway.propose(
            session_id=session_id,
            action_type=ACTION_TYPE,
            payload={
                "room_id": existing["room_id"],
                "issue_type": existing["issue_type"],
                "issue_desc": existing["issue_desc"],
                "urgency": existing["urgency"],
                "contact_name": existing.get("contact_name") or DEFAULT_OWNER_NAME,
                "contact_phone": existing["contact_phone"],
                "appointment_time": existing["appointment_time"],
            },
            trace_id=trace_id,
            release_id=release_id,
        )
        if proposal.status == "pending_confirmation":
            proposal = action_gateway.approve(
                proposal.proposal_id,
                actor=f"owner:{session_id}",
                comment="用户明确确认创建维修工单",
            )
        receipt = action_gateway.execute(proposal.proposal_id)
        if not receipt.may_claim_success:
            return _result(
                "failed",
                "工单提交未成功，后端没有返回已提交 Receipt；草稿仍保留，请稍后重试或转人工。",
                existing,
                proposal_id=proposal.proposal_id,
                receipt=receipt.model_dump(mode="json"),
                error_summary=receipt.error_summary,
            )
        delete_work_order_draft(session_id)
        actual_id = receipt.resource_id
        return _result(
            "committed",
            f"正式维修工单已创建成功，工单号：{actual_id}。维修人员将按“{existing['appointment_time']}”安排处理，请保持电话畅通。",
            existing,
            work_order_id=actual_id,
            proposal_id=proposal.proposal_id,
            receipt=receipt.model_dump(mode="json"),
        )

    base = dict(existing or {})
    issue_desc = base.get("issue_desc") or _issue_description(message)
    urgency = _urgency(message) or base.get("urgency") or ""
    issue_type = (
        _issue_type(issue_desc)
        if not str(base.get("issue_desc") or "").strip()
        else (base.get("issue_type") or _issue_type(issue_desc))
    )
    draft = {
        "room_id": _room_id(message) or base.get("room_id") or DEFAULT_ROOM_ID,
        "issue_type": issue_type,
        "issue_desc": issue_desc,
        "urgency": urgency,
        "contact_name": _contact_name(message) or base.get("contact_name") or DEFAULT_OWNER_NAME,
        "contact_phone": _phone(message) or base.get("contact_phone") or "",
        "appointment_time": _appointment(message, urgency) or base.get("appointment_time") or "",
    }
    save_work_order_draft(session_id=session_id, **draft)
    missing = _missing(draft)
    if missing:
        ask_now = missing[:2]
        return _result(
            "draft_updated",
            "我已记录为待确认的维修工单草稿（尚未创建正式工单）。\n\n"
            + _summary(draft)
            + "\n\n请先补充："
            + "、".join(ask_now)
            + "。"
            + ("其余信息收齐后，我会请您确认创建。" if len(missing) > len(ask_now) else ""),
            draft,
        )
    proposal = action_gateway.propose(
        session_id=session_id,
        action_type=ACTION_TYPE,
        payload=draft,
        trace_id=trace_id,
        release_id=release_id,
    )
    return _result(
        "awaiting_confirmation",
        "维修工单草稿已完整，尚未创建正式工单。请核对：\n\n"
        + _summary(draft)
        + "\n\n如信息无误，请回复“确认创建”。",
        draft,
        proposal_id=proposal.proposal_id,
    )
