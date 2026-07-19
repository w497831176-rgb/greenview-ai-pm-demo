"""Backend-owned confirmed action execution with idempotent receipts."""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, Optional

from app.runtime.contracts import (
    ActionProposal,
    ActionReceipt,
    RiskLevel,
    content_hash,
)
from db.property_db import (
    create_action_proposal,
    create_work_order,
    get_action_proposal,
    get_action_receipt_by_idempotency_key,
    get_runtime_release,
    now_cn,
    record_action_approval,
    save_action_receipt,
)


ActionHandler = Callable[[Dict[str, Any], str], Dict[str, Any]]


class ActionGateway:
    """The only V1.8 component allowed to commit model-proposed writes."""

    def __init__(self):
        self._handlers: Dict[str, ActionHandler] = {
            "work_order.create": self._create_work_order,
        }

    def register(self, action_type: str, handler: ActionHandler) -> None:
        self._handlers[action_type] = handler

    def propose(
        self,
        session_id: str,
        action_type: str,
        payload: Dict[str, Any],
        trace_id: Optional[str] = None,
        release_id: Optional[str] = None,
        risk_level: RiskLevel = RiskLevel.L2,
    ) -> ActionProposal:
        parameter_hash = content_hash(payload)
        idempotency_key = content_hash(
            {
                "session_id": session_id,
                "action_type": action_type,
                "parameter_hash": parameter_hash,
            }
        )
        row = create_action_proposal(
            proposal_id=f"proposal_{uuid.uuid4().hex}",
            session_id=session_id,
            trace_id=trace_id,
            release_id=release_id,
            action_type=action_type,
            risk_level=risk_level.value,
            payload=payload,
            idempotency_key=idempotency_key,
        )
        return self._proposal_contract(row)

    def approve(
        self,
        proposal_id: str,
        actor: str,
        comment: Optional[str] = None,
    ) -> ActionProposal:
        row = record_action_approval(proposal_id, "approved", actor, comment)
        return self._proposal_contract(row)

    def reject(
        self,
        proposal_id: str,
        actor: str,
        comment: Optional[str] = None,
    ) -> ActionProposal:
        row = record_action_approval(proposal_id, "rejected", actor, comment)
        return self._proposal_contract(row)

    def execute(self, proposal_id: str) -> ActionReceipt:
        proposal = get_action_proposal(proposal_id)
        if not proposal:
            raise ValueError("action proposal not found")
        existing = get_action_receipt_by_idempotency_key(proposal["idempotency_key"])
        if existing:
            return self._receipt_contract(existing)
        if proposal.get("status") != "approved":
            raise PermissionError("action proposal is not approved")
        handler = self._handlers.get(proposal["action_type"])
        if not handler:
            return self._failed_receipt(
                proposal,
                f"no backend action handler registered for {proposal['action_type']}",
            )
        try:
            result = handler(proposal["payload"], proposal["session_id"])
            resource_id = str(result.get("resource_id") or "")
            if not resource_id:
                return self._failed_receipt(
                    proposal,
                    "backend handler returned no committed resource id",
                )
            return self._committed_receipt(proposal, result)
        except Exception as exc:
            return self._failed_receipt(proposal, str(exc))

    async def execute_async(self, proposal_id: str) -> ActionReceipt:
        """Execute a registered backend action or a published MCP write."""

        proposal = get_action_proposal(proposal_id)
        if not proposal:
            raise ValueError("action proposal not found")
        existing = get_action_receipt_by_idempotency_key(proposal["idempotency_key"])
        if existing:
            return self._receipt_contract(existing)
        if proposal.get("status") != "approved":
            raise PermissionError("action proposal is not approved")
        if not str(proposal.get("action_type") or "").startswith("mcp."):
            return self.execute(proposal_id)
        try:
            release = get_runtime_release(str(proposal.get("release_id") or ""))
            if not release:
                raise RuntimeError("proposal RuntimeRelease no longer exists")
            payload = proposal.get("payload") or {}
            from app.runtime.mcp_executor import invoke_confirmed_write

            result = await invoke_confirmed_write(
                snapshot_config=release.get("config") or {},
                agent_id=str(payload.get("agent_id") or ""),
                server_name=str(payload.get("server_name") or ""),
                tool_name=str(payload.get("tool_name") or ""),
                arguments=payload.get("arguments") or {},
            )
            return self._committed_receipt(proposal, result)
        except Exception as exc:
            return self._failed_receipt(proposal, str(exc))

    def _committed_receipt(
        self,
        proposal: Dict[str, Any],
        result: Dict[str, Any],
    ) -> ActionReceipt:
        resource_id = str(result.get("resource_id") or "")
        if not resource_id:
            return self._failed_receipt(
                proposal,
                "backend handler returned no committed resource id",
            )
        saved = save_action_receipt(
            receipt_id=f"receipt_{uuid.uuid4().hex}",
            proposal_id=proposal["proposal_id"],
            idempotency_key=proposal["idempotency_key"],
            status="committed",
            result=result,
            resource_type=str(result.get("resource_type") or proposal["action_type"]),
            resource_id=resource_id,
        )
        return self._receipt_contract(saved)

    def _failed_receipt(
        self,
        proposal: Dict[str, Any],
        error_summary: str,
    ) -> ActionReceipt:
        saved = save_action_receipt(
            receipt_id=f"receipt_{uuid.uuid4().hex}",
            proposal_id=proposal["proposal_id"],
            idempotency_key=proposal["idempotency_key"],
            status="failed",
            result={},
            error_summary=error_summary[:500],
        )
        return self._receipt_contract(saved)

    @staticmethod
    def _proposal_contract(row: Dict[str, Any]) -> ActionProposal:
        return ActionProposal(
            proposal_id=row["proposal_id"],
            session_id=row["session_id"],
            trace_id=row.get("trace_id"),
            release_id=row.get("release_id"),
            action_type=row["action_type"],
            risk_level=RiskLevel(row["risk_level"]),
            payload=row["payload"],
            parameter_hash=content_hash(row["payload"]),
            idempotency_key=row["idempotency_key"],
            status=row["status"],
        )

    @staticmethod
    def _receipt_contract(row: Dict[str, Any]) -> ActionReceipt:
        return ActionReceipt(
            receipt_id=row["receipt_id"],
            proposal_id=row["proposal_id"],
            idempotency_key=row["idempotency_key"],
            status=row["status"],
            resource_type=row.get("resource_type"),
            resource_id=row.get("resource_id"),
            result=row.get("result") or {},
            committed_at=row.get("committed_at"),
            error_summary=row.get("error_summary"),
        )

    @staticmethod
    def _create_work_order(payload: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        required = (
            "room_id",
            "issue_type",
            "issue_desc",
            "urgency",
            "contact_phone",
            "appointment_time",
        )
        missing = [field for field in required if not str(payload.get(field) or "").strip()]
        if missing:
            raise ValueError("work order payload missing: " + ", ".join(missing))
        work_order_id = (
            f"WO-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        )
        created = create_work_order(
            work_order_id=work_order_id,
            room_id=str(payload["room_id"]),
            issue_type=str(payload["issue_type"]),
            issue_desc=str(payload["issue_desc"]),
            urgency=str(payload["urgency"]),
            contact_name=str(payload.get("contact_name") or "业主"),
            contact_phone=str(payload["contact_phone"]),
            appointment_time=str(payload["appointment_time"]),
            status="待派单",
            session_id=session_id,
        )
        actual_id = str((created or {}).get("id") or "")
        if not actual_id:
            raise RuntimeError("work order database did not return an id")
        return {
            "resource_type": "work_order",
            "resource_id": actual_id,
            "work_order": created,
            "committed_at": now_cn(),
        }
