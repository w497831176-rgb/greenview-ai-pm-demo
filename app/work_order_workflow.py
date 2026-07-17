"""Authoritative, session-scoped repair work-order workflow.

The language model can explain and collect information, but it never decides
whether a real work order has been written.  This module is the control plane:
it keeps a per-session draft, validates the required fields, and writes exactly
once only after an explicit confirmation.
"""

import re
import time
from typing import Any, Dict, List, Optional

from db.property_db import (
    create_work_order,
    delete_work_order_draft,
    get_work_order_draft,
    save_work_order_draft,
)


DEFAULT_ROOM_ID = "3-2-1201"
DEFAULT_OWNER_NAME = "王先生"


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


def is_explicit_work_order_request(message: str) -> bool:
    lowered = (message or "").lower()
    return any(word in lowered for word in (
        "报修", "创建工单", "创建维修工单", "提交工单", "安排维修", "安排师傅", "派师傅", "上门维修", "我要修", "帮我修",
    ))


def is_cancel_request(message: str) -> bool:
    return any(word in (message or "") for word in ("取消报修", "取消工单草稿", "不报修了", "先不用报修"))


def is_confirmation(message: str) -> bool:
    normalized = (message or "").strip().lower()
    return bool(re.search(r"^(确认创建|确认|同意创建|好的.*创建|就.*创建|创建.*)$", normalized))


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


def advance_work_order_workflow(session_id: str, message: str) -> Optional[Dict[str, Any]]:
    """Advance a repair draft without relying on a model tool call.

    Returns ``None`` for ordinary chat.  Every non-None result is authoritative
    and can be safely rendered by the chat API without claiming a fictitious
    work order.
    """
    existing = get_work_order_draft(session_id)

    if existing and is_cancel_request(message):
        delete_work_order_draft(session_id)
        return _result(
            "cancelled",
            "已取消本次待确认的报修草稿，未创建正式工单。需要时您可以重新告诉我报修问题。",
            {},
        )

    if not existing and not is_explicit_work_order_request(message):
        return None

    if existing and is_confirmation(message):
        missing = _missing(existing)
        if missing:
            return _result(
                "confirmation_blocked",
                "暂不能创建正式工单，因为还缺少：" + "、".join(missing) + "。请补充后再确认创建。",
                existing,
            )
        work_order_id = f"WO-{time.strftime('%Y%m%d')}-{int(time.time() * 1000) % 1000000:06d}"
        created = create_work_order(
            work_order_id=work_order_id,
            room_id=existing["room_id"],
            issue_type=existing["issue_type"],
            issue_desc=existing["issue_desc"],
            urgency=existing["urgency"],
            contact_name=existing.get("contact_name") or DEFAULT_OWNER_NAME,
            contact_phone=existing["contact_phone"],
            appointment_time=existing["appointment_time"],
            status="pending",
            session_id=session_id,
        )
        delete_work_order_draft(session_id)
        actual_id = (created or {}).get("id") or work_order_id
        return _result(
            "created",
            f"正式维修工单已创建成功，工单号：{actual_id}。维修人员将按“{existing['appointment_time']}”安排处理，请保持电话畅通。",
            existing,
            work_order_id=actual_id,
        )

    base = dict(existing or {})
    issue_desc = base.get("issue_desc") or message.strip()
    urgency = _urgency(message) or base.get("urgency") or ""
    draft = {
        "room_id": _room_id(message) or base.get("room_id") or DEFAULT_ROOM_ID,
        "issue_type": base.get("issue_type") or _issue_type(issue_desc),
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
    return _result(
        "awaiting_confirmation",
        "维修工单草稿已完整，尚未创建正式工单。请核对：\n\n"
        + _summary(draft)
        + "\n\n如信息无误，请回复“确认创建”。",
        draft,
    )
