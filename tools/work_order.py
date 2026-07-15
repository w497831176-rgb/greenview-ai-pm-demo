"""
Work Order Tools
================

Agno Toolkit wrapper around the property work order database.
"""

from datetime import datetime
from typing import Optional

from agno.tools import Toolkit

from db.property_db import create_work_order as db_create_work_order
from db.property_db import get_work_order as db_get_work_order
from db.property_db import list_work_orders as db_list_work_orders


class WorkOrderTools(Toolkit):
    """Create and query property maintenance work orders."""

    def __init__(self):
        super().__init__(name="work_order_tools")

    def create_work_order(
        self,
        room_id: str,
        issue_type: str,
        issue_desc: str,
        urgency: str = "中",
        contact_name: Optional[str] = None,
        contact_phone: Optional[str] = None,
        appointment_time: Optional[str] = None,
    ) -> str:
        """Create a new maintenance work order and return the order ID."""
        today = datetime.now().strftime("%Y%m%d")
        existing = db_list_work_orders(date_prefix=today)
        seq = len(existing) + 1
        work_order_id = f"WO-{today}-{seq:03d}"

        order = db_create_work_order(
            work_order_id=work_order_id,
            room_id=room_id,
            issue_type=issue_type,
            issue_desc=issue_desc,
            urgency=urgency,
            contact_name=contact_name or "业主",
            contact_phone=contact_phone or "",
            appointment_time=appointment_time or "",
        )
        return (
            f"工单已创建，工单号：{order['id']}，"
            f"房号：{order['room_id']}，问题类型：{order['issue_type']}，"
            f"紧急程度：{order['urgency']}，当前状态：{order['status']}。"
        )

    def query_work_order(self, work_order_id: str) -> str:
        """Query a single work order by ID."""
        order = db_get_work_order(work_order_id)
        if not order:
            return f"未找到工单 {work_order_id}。"
        return (
            f"工单 {order['id']}：房号 {order['room_id']}，"
            f"问题：{order['issue_desc']}，状态：{order['status']}，"
            f"创建时间：{order['created_at']}。"
        )
