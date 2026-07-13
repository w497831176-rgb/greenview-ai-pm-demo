"""
Work Order API
==============

REST endpoints for listing, creating, and updating work orders.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from db.property_db import (
    create_work_order as db_create_work_order,
    get_work_order as db_get_work_order,
    list_work_orders as db_list_work_orders,
    update_work_order_status as db_update_work_order_status,
)

router = APIRouter(prefix="/api/work-orders", tags=["work-orders"])


class WorkOrderCreate(BaseModel):
    room_id: str
    issue_type: str
    issue_desc: str
    urgency: str = "中"
    contact_name: Optional[str] = None
    contact_phone: Optional[str] = None
    appointment_time: Optional[str] = None


class WorkOrderStatusUpdate(BaseModel):
    status: str
    assigned_to: Optional[str] = None
    completion_note: Optional[str] = None


@router.get("")
async def list_orders(
    status: Optional[str] = Query(None),
    room_id: Optional[str] = Query(None),
    limit: int = Query(100),
):
    """List work orders with optional filters."""
    orders = db_list_work_orders(status=status, room_id=room_id, limit=limit)
    return {"work_orders": orders, "count": len(orders)}


@router.get("/{work_order_id}")
async def get_order(work_order_id: str):
    """Get a single work order by ID."""
    order = db_get_work_order(work_order_id)
    if not order:
        raise HTTPException(status_code=404, detail="not found")
    return {"work_order": order}


@router.post("")
async def create_order(request: WorkOrderCreate):
    """Create a new work order."""
    from datetime import datetime

    today = datetime.now().strftime("%Y%m%d")
    existing = db_list_work_orders(date_prefix=today)
    seq = len(existing) + 1
    work_order_id = f"WO-{today}-{seq:03d}"

    order = db_create_work_order(
        work_order_id=work_order_id,
        room_id=request.room_id,
        issue_type=request.issue_type,
        issue_desc=request.issue_desc,
        urgency=request.urgency,
        contact_name=request.contact_name or "业主",
        contact_phone=request.contact_phone or "",
        appointment_time=request.appointment_time or "",
    )
    return {"work_order": order}


@router.patch("/{work_order_id}")
async def update_order(work_order_id: str, request: WorkOrderStatusUpdate):
    """Update work order status / assignee / note."""
    order = db_update_work_order_status(
        work_order_id=work_order_id,
        status=request.status,
        assigned_to=request.assigned_to,
        completion_note=request.completion_note,
    )
    if not order:
        raise HTTPException(status_code=404, detail="not found")
    return {"work_order": order}
