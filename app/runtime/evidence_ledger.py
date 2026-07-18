"""Single source of truth for runtime evidence, Trace, UI and evaluation."""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.runtime.contracts import RunEvidenceLedger, RunState
from db.property_db import get_evidence_ledger, save_evidence_ledger


class EvidenceLedger:
    def __init__(
        self,
        trace_id: str,
        session_id: str,
        config_snapshot: Dict[str, Any],
        release_id: Optional[str],
        config_hash: Optional[str],
        runtime_path: str,
    ):
        self.release_id = release_id
        self.config_hash = config_hash
        self.runtime_path = runtime_path
        existing = get_evidence_ledger(trace_id)
        if existing:
            self.contract = RunEvidenceLedger.model_validate(existing["ledger"])
        else:
            self.contract = RunEvidenceLedger(
                trace_id=trace_id,
                session_id=session_id,
                config_snapshot=config_snapshot,
            )
            self.persist("running")

    def set(self, field: str, value: Any) -> None:
        if field not in self.contract.model_fields:
            raise ValueError(f"unknown evidence ledger field: {field}")
        setattr(self.contract, field, value)

    def append(self, field: str, value: Dict[str, Any]) -> None:
        if field not in self.contract.model_fields:
            raise ValueError(f"unknown evidence ledger field: {field}")
        collection = getattr(self.contract, field)
        if not isinstance(collection, list):
            raise ValueError(f"evidence ledger field is not appendable: {field}")
        collection.append(value)

    def violation(self, code: str, detail: str, **metadata: Any) -> None:
        self.contract.contract_violations.append(
            {"code": code, "detail": detail, "metadata": metadata}
        )

    def capture_state(self, state: RunState) -> None:
        self.contract.route_decision = (
            state.route_decision.model_dump(mode="json")
            if state.route_decision
            else None
        )
        self.contract.activated_skills = [
            item.model_dump(mode="json") for item in state.activated_skills
        ]
        self.contract.retrieval_evidence = [
            item.model_dump(mode="json") for item in state.retrieval_evidence.items
        ]
        self.contract.tool_invocations = [
            item.model_dump(mode="json") for item in state.tool_invocations
        ]
        self.contract.action_proposals = [
            item.model_dump(mode="json") for item in state.pending_actions
        ]
        self.contract.approval_events = [
            item.model_dump(mode="json") for item in state.approval_events
        ]
        self.contract.action_receipts = [
            item.model_dump(mode="json") for item in state.action_receipts
        ]
        self.contract.model_calls = list(state.model_calls)
        self.contract.citation_links = [
            item.model_dump(mode="json") for item in state.citations
        ]
        self.contract.cost_entries = [
            item.model_dump(mode="json") for item in state.cost_entries
        ]

    def persist(self, status: str) -> Dict[str, Any]:
        return save_evidence_ledger(
            trace_id=self.contract.trace_id,
            session_id=self.contract.session_id,
            ledger=self.contract.model_dump(mode="json"),
            release_id=self.release_id,
            config_hash=self.config_hash,
            runtime_path=self.runtime_path,
            status=status,
        )


def evidence_payload_for_ui(trace_id: str) -> Optional[Dict[str, Any]]:
    """Return the stored historical snapshots; never re-query the live index."""
    row = get_evidence_ledger(trace_id)
    return row["ledger"] if row else None
