"""Work order creation tools with V1.4.3 session draft + explicit confirmation.

The backend enforces a two-stage handshake:
1. First complete request in a session -> save a per-session draft, return summary
   and ask the user to reply "确认创建".
2. User replies "确认创建" with a valid draft for the same session -> create the
   real work order exactly once.
"""

import contextvars
import json
import re
import time
import traceback
from typing import Any, Optional

from agno.tools import Toolkit
from db.property_db import (
    create_work_order as db_create_work_order,
    delete_work_order_draft,
    get_work_order_draft,
    save_work_order_draft,
)

# Populated by app/chat.py around each agent turn.
_work_order_session_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "work_order_session_id", default=None
)
_work_order_user_message: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "work_order_user_message", default=None
)


def set_work_order_context(session_id: Optional[str], user_message: Optional[str]) -> None:
    """Set the context variables used by create_work_order for confirmation gating."""
    _work_order_session_id.set(session_id)
    _work_order_user_message.set(user_message)


def _extract_urgency(desc: str) -> str:
    lowered = desc.lower()
    if any(k in lowered for k in ["爆管", " flooding", "水漫", "漏电", "着火", "燃气泄漏", "电梯困人"]):
        return "urgent"
    if any(k in lowered for k in ["滴水", "渗水", "发霉", "裂缝", "门锁坏", "灯不亮", "堵塞"]):
        return "normal"
    return "low"


def _extract_issue_type(desc: str) -> str:
    lowered = desc.lower()
    if any(k in lowered for k in ["水", "漏", "渗", "滴", "管", "下水道", "马桶", "龙头", "水槽"]):
        return "plumbing"
    if any(k in lowered for k in ["电", "灯", "跳闸", "插座", "开关", "线路"]):
        return "electrical"
    if any(k in lowered for k in ["门锁", "门把", "窗户", "玻璃", "把手", "柜门", "合页"]):
        return "lock_door_window"
    if any(k in lowered for k in ["电梯", "梯"]):
        return "elevator"
    if any(k in lowered for k in ["空调", "暖气", "制冷", "通风", "新风"]):
        return "hvac"
    return "general_repair"


def _extract_room_id(text: str, default: str = "3-2-1201") -> str:
    # Match patterns like 3-2-1201, 3栋2单元1201, 3号楼2单元1201室 etc.
    pattern = r"(\d{1,2})\s*[-#—]\s*(\d{1,2})\s*[-#—]\s*(\d{3,4})"
    match = re.search(pattern, text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    # Chinese pattern: 3栋2单元1201
    pattern2 = r"(\d{1,2})\s*[栋幢]\s*(\d{1,2})\s*[单元]\s*(\d{3,4})"
    match2 = re.search(pattern2, text)
    if match2:
        return f"{match2.group(1)}-{match2.group(2)}-{match2.group(3)}"
    return default


def _extract_contact_name(text: str) -> str:
    # Look for patterns like "我是王先生" or "联系人：王先生"
    patterns = [
        r"我是([\u4e00-\u9fa5]{1,4})(先生|女士)?",
        r"联系人[：:]\s*([\u4e00-\u9fa5]{1,4})(先生|女士)?",
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            return match.group(1) + (match.group(2) or "")
    return ""


def _extract_contact_phone(text: str) -> str:
    match = re.search(r"1[3-9]\d{9}", text)
    return match.group(0) if match else ""


def _user_confirmed_creation(message: Optional[str]) -> bool:
    if not message:
        return False
    lowered = message.strip().lower()
    confirm_patterns = [
        r"^确认创建",
        r"^确认",
        r"^好的.*创建",
        r"^同意创建",
        r"^就.*创建",
        r"^创建.*",
    ]
    return any(re.search(p, lowered) for p in confirm_patterns)


class WorkOrderTools(Toolkit):
    def __init__(self):
        super().__init__(name="work_order_tools")
        self.register(self.create_work_order)

    def create_work_order(
        self,
        room_id: Optional[str] = None,
        issue_desc: str = "",
        issue_type: Optional[str] = None,
        urgency: Optional[str] = None,
        contact_name: Optional[str] = None,
        contact_phone: Optional[str] = None,
        appointment_time: Optional[str] = None,
    ) -> str:
        """Create or confirm a work order for the current chat session.

        Two-stage handshake:
        - On the first complete request in a session, only a draft is saved and
          the tool returns a summary asking the user to reply "确认创建".
        - When the user replies "确认创建" and a valid draft exists for this
          session, the real work order is created exactly once.
        """
        session_id = _work_order_session_id.get()
        user_message = _work_order_user_message.get()

        if not issue_desc:
            return "错误：请描述具体的维修问题，我才能帮您记录。"

        # Derive missing fields from the user's message when needed.
        merged_text = f"{user_message or ''} {issue_desc}"
        room_id = room_id or _extract_room_id(merged_text)
        issue_type = issue_type or _extract_issue_type(issue_desc)
        urgency = urgency or _extract_urgency(issue_desc)
        contact_name = contact_name or _extract_contact_name(merged_text)
        contact_phone = contact_phone or _extract_contact_phone(merged_text)

        # Stage 1: no session id -> cannot enforce confirmation, fall back to draft only.
        if not session_id:
            return (
                "【待确认工单草稿】\n"
                f"房号：{room_id}\n"
                f"问题类型：{issue_type}\n"
                f"问题描述：{issue_desc}\n"
                f"紧急度：{urgency}\n"
                f"联系人：{contact_name or '未提供'}\n"
                f"电话：{contact_phone or '未提供'}\n"
                "请回复「确认创建」以生成正式工单。"
            )

        existing_draft = get_work_order_draft(session_id)
        user_confirmed = _user_confirmed_creation(user_message)

        # Stage 2: user confirmed and a draft exists -> create the real work order once.
        if user_confirmed and existing_draft:
            # Prevent duplicate creation if the draft was already consumed.
            try:
                work_order_id = f"WO{int(time.time() * 1000)}"
                result = db_create_work_order(
                    work_order_id=work_order_id,
                    room_id=existing_draft["room_id"],
                    issue_type=existing_draft["issue_type"],
                    issue_desc=existing_draft["issue_desc"],
                    urgency=existing_draft["urgency"],
                    contact_name=existing_draft.get("contact_name") or "",
                    contact_phone=existing_draft.get("contact_phone") or "",
                    appointment_time=existing_draft.get("appointment_time") or "",
                    status="pending",
                    session_id=session_id,
                )
                delete_work_order_draft(session_id)
                return (
                    f"工单已创建成功，工单号：{result.get('id') if result else work_order_id}。"
                    "工作人员会尽快处理。"
                )
            except Exception as exc:
                traceback.print_exc()
                return f"工单创建失败：{exc}。请稍后重试或联系人工。"

        # Stage 3: no confirmation yet -> save/update draft and ask for confirmation.
        save_work_order_draft(
            session_id=session_id,
            room_id=room_id,
            issue_type=issue_type,
            issue_desc=issue_desc,
            urgency=urgency,
            contact_name=contact_name or "",
            contact_phone=contact_phone or "",
            appointment_time=appointment_time or "",
        )

        return (
            "【待确认工单草稿】\n"
            f"房号：{room_id}\n"
            f"问题类型：{issue_type}\n"
            f"问题描述：{issue_desc}\n"
            f"紧急度：{urgency}\n"
            f"联系人：{contact_name or '未提供'}\n"
            f"电话：{contact_phone or '未提供'}\n"
            "请核对以上信息，回复「确认创建」后我将为您生成正式工单。"
        )
