"""Single source of truth for runtime evidence, Trace, UI and evaluation."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.runtime.contracts import RunEvidenceBundle, RunEvidenceLedger, RunState
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
        if state.lane_decision:
            lane_payload = state.lane_decision.model_dump(mode="json")
            existing_explanation = (self.contract.lane_decision or {}).get(
                "explanation"
            )
            if existing_explanation:
                lane_payload["explanation"] = existing_explanation
            self.contract.lane_decision = lane_payload
        else:
            self.contract.lane_decision = None
        self.contract.answer_contract = (
            state.answer_contract.model_dump(mode="json")
            if state.answer_contract
            else None
        )
        self.contract.route_decision = (
            state.route_decision.model_dump(mode="json")
            if state.route_decision
            else None
        )
        self.contract.capability_decision = (
            state.capability_decision.model_dump(mode="json")
            if state.capability_decision
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
        if state.evidence_bundle is not None:
            self.contract.run_evidence_bundle = state.evidence_bundle.model_dump(
                mode="json"
            )
        else:
            # Non-consultation paths still receive the same contract shape.
            # They normally have no RAG/Tool evidence, but a committed Receipt
            # must never remain available only through a parallel flat list.
            tool_evidence = [
                item.tool_evidence
                for item in state.tool_invocations
                if item.tool_evidence is not None
            ]
            delivered_ids = [item.evidence_id for item in state.citations]
            self.contract.run_evidence_bundle = RunEvidenceBundle(
                retrieved_rag_candidates=state.retrieval_evidence,
                skill_evidence=list(self.contract.skill_evidence),
                tool_evidence=tool_evidence,
                committed_receipts=[
                    item
                    for item in state.action_receipts
                    if item.may_claim_success
                ],
                validated_rag_evidence_ids=delivered_ids,
                delivered_evidence_ids=delivered_ids,
            ).model_dump(mode="json")

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


def _list_of_dicts(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _rag_candidates(
    bundle: Optional[Dict[str, Any]], ledger: Dict[str, Any]
) -> List[Dict[str, Any]]:
    if bundle is not None:
        evidence_set = bundle.get("retrieved_rag_candidates") or {}
        raw_items = (
            evidence_set.get("items")
            if isinstance(evidence_set, dict)
            else evidence_set
        )
    else:
        raw_items = ledger.get("retrieval_evidence")

    candidates: List[Dict[str, Any]] = []
    for raw in _list_of_dicts(raw_items):
        item = dict(raw)
        item.setdefault("doc_title", item.get("title") or "")
        item.setdefault("doc_id", item.get("document_id"))
        item.setdefault("content", item.get("content_snapshot") or "")
        candidates.append(item)
    return candidates


def _normalize_number(value: Any) -> Optional[str]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        normalized = format(Decimal(str(value)).normalize(), "f")
    except (InvalidOperation, TypeError, ValueError):
        return None
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _field_unit(field_name: str) -> Optional[str]:
    """Infer only explicit, generic unit semantics from a structured field name."""

    name = re.sub(r"[^a-z0-9]+", "_", str(field_name or "").lower()).strip("_")
    parts = set(name.split("_"))
    if "pct" in parts or "percent" in parts or "percentage" in parts:
        return "%"
    if "celsius" in parts or name.endswith("_deg_c") or name.endswith("_temperature_c"):
        return "℃"
    if "minutes" in parts or "minute" in parts or name.endswith("_mins"):
        return "分钟"
    if "yuan" in parts or "cny" in parts or name.endswith("_rmb"):
        return "元"
    return None


def _flatten_scalars(value: Any, path: str = "$") -> Iterable[Tuple[str, str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if isinstance(child, (dict, list)):
                yield from _flatten_scalars(child, child_path)
            else:
                yield child_path, str(key), child
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            if isinstance(child, (dict, list)):
                yield from _flatten_scalars(child, child_path)


_DISPLAY_VALUE = re.compile(
    r"^\s*(?P<number>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>%|℃|°C|分钟|元)\s*$",
    re.IGNORECASE,
)


def _display_value(value: Any) -> Optional[Tuple[str, str]]:
    match = _DISPLAY_VALUE.match(str(value or ""))
    if not match:
        return None
    unit = match.group("unit")
    if unit.lower() == "°c":
        unit = "℃"
    number = _normalize_number(match.group("number"))
    return (number, unit) if number is not None else None


def _legacy_tool_matches(
    values: Iterable[Any], tool_invocations: Any
) -> List[Dict[str, Any]]:
    """Project a legacy validator miss from successful result JSON only.

    Request arguments are deliberately ignored. A truncated or non-JSON summary
    is not promoted into result evidence.
    """

    expected = {
        parsed
        for parsed in (_display_value(value) for value in values)
        if parsed is not None
    }
    if not expected:
        return []

    matches: List[Dict[str, Any]] = []
    for invocation in _list_of_dicts(tool_invocations):
        if (
            invocation.get("effect") != "read"
            or invocation.get("transport_status") != "success"
            or invocation.get("invocation_status") != "success"
            or invocation.get("business_status") != "success"
        ):
            continue
        try:
            payload = json.loads(str(invocation.get("result_summary") or ""))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        for json_path, field_name, raw_value in _flatten_scalars(payload):
            unit = _field_unit(field_name)
            number = _normalize_number(raw_value)
            if unit is None or number is None or (number, unit) not in expected:
                continue
            matches.append(
                {
                    "invocation_id": invocation.get("invocation_id"),
                    "server_name": invocation.get("server_name"),
                    "tool_name": invocation.get("tool_name"),
                    "json_path": json_path,
                    "normalized_value": number,
                    "unit": unit,
                }
            )
    return matches


def _legacy_violation_projection(
    violation: Dict[str, Any], ledger: Dict[str, Any]
) -> Dict[str, Any]:
    projected = dict(violation)
    if projected.get("code") != "ungrounded_critical_value":
        return projected
    metadata = projected.get("metadata")
    values = metadata.get("values") if isinstance(metadata, dict) else None
    if not isinstance(values, list) or not values:
        return projected
    matches = _legacy_tool_matches(values, ledger.get("tool_invocations"))
    matched_values = {
        (item.get("normalized_value"), item.get("unit")) for item in matches
    }
    expected_values = {
        parsed
        for parsed in (_display_value(value) for value in values)
        if parsed is not None
    }
    if not expected_values or matched_values != expected_values:
        return projected
    return {
        "code": "tool_result_value_not_recognized_by_validation",
        "source_code": "ungrounded_critical_value",
        "detail": (
            "成功只读 Tool 的结构化结果已包含该数值，但旧版回答校验未识别该 "
            "Tool 结果证据。"
        ),
        "values": list(values),
        "tool_result_evidence": matches,
        "historical_projection": True,
    }


def _delivery_was_withheld(ledger: Dict[str, Any]) -> bool:
    return any(
        item.get("case") == "knowledge_evidence_gate"
        and item.get("decision") == "rejected_insufficient"
        for item in _list_of_dicts(ledger.get("evaluation_results"))
    )


def _violation_json_key(item: Dict[str, Any]) -> str:
    try:
        return json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(item)


def _violation_semantic_key(item: Dict[str, Any]) -> str:
    """Normalize the same violation stored in Bundle and legacy ledger shapes."""

    metadata = item.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}

    def field(name: str) -> Any:
        value = item.get(name)
        return metadata.get(name) if value is None else value

    evidence_ids = field("evidence_ids")
    if isinstance(evidence_ids, list):
        evidence_ids = sorted(str(value) for value in evidence_ids)
    values = field("values")
    if isinstance(values, list):
        values = sorted(str(value) for value in values)
    identity = {
        "code": str(item.get("code") or ""),
        "values": values or [],
        "evidence_id": field("evidence_id"),
        "evidence_ids": evidence_ids or [],
        "claim_context": field("claim_context"),
        "marker": field("marker"),
        "index": field("index"),
        "json_path": field("json_path"),
        "semantic_type": field("semantic_type"),
    }
    return _violation_json_key(identity)


def _merge_violations(
    primary: Iterable[Dict[str, Any]], fallback: Iterable[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    primary_semantic: set[str] = set()
    seen_exact: set[str] = set()
    for raw in primary:
        item = dict(raw)
        exact_key = _violation_json_key(item)
        if exact_key in seen_exact:
            continue
        seen_exact.add(exact_key)
        primary_semantic.add(_violation_semantic_key(item))
        result.append(item)
    for raw in fallback:
        item = dict(raw)
        if _violation_semantic_key(item) in primary_semantic:
            continue
        exact_key = _violation_json_key(item)
        if exact_key in seen_exact:
            continue
        seen_exact.add(exact_key)
        result.append(item)
    return result


def project_evidence_for_trace(ledger: Dict[str, Any]) -> Dict[str, Any]:
    """Build the canonical read-only Trace projection from frozen ledger JSON."""

    source = dict(ledger or {})
    raw_bundle = source.get("run_evidence_bundle")
    bundle = dict(raw_bundle) if isinstance(raw_bundle, dict) else None
    candidates = _rag_candidates(bundle, source)
    candidate_ids = {
        str(item.get("evidence_id"))
        for item in candidates
        if item.get("evidence_id")
    }

    if bundle is not None:
        delivered_ids = {
            str(item)
            for item in bundle.get("delivered_evidence_ids") or []
            if item
        }
        tool_evidence = {
            str(item.get("evidence_id")): item
            for item in _list_of_dicts(bundle.get("tool_evidence"))
            if item.get("evidence_id")
        }
        tool_links = []
        for raw_link in _list_of_dicts(bundle.get("tool_evidence_links")):
            link = dict(raw_link)
            evidence = tool_evidence.get(str(link.get("evidence_id") or ""))
            if evidence is not None:
                link["tool_evidence"] = dict(evidence)
            tool_links.append(link)
        withheld = _list_of_dicts(bundle.get("withheld"))
        violations = _merge_violations(
            _list_of_dicts(bundle.get("violations")),
            _list_of_dicts(source.get("contract_violations")),
        )
        projection_source = "run_evidence_bundle"
    else:
        delivered_ids = {
            str(item.get("evidence_id"))
            for item in _list_of_dicts(source.get("citation_links"))
            if item.get("evidence_id")
        }
        tool_links = []
        violations = [
            _legacy_violation_projection(item, source)
            for item in _list_of_dicts(source.get("contract_violations"))
        ]
        withheld = []
        if violations and _delivery_was_withheld(source):
            withheld = [
                {
                    "code": item.get("code"),
                    "reason": item.get("detail"),
                    "values": item.get("values")
                    or (item.get("metadata") or {}).get("values")
                    or [],
                    "delivery_status": "withheld_before_user_delivery",
                    "content_available": False,
                    "historical_projection": True,
                }
                for item in violations
            ]
        projection_source = "legacy_flat_ledger"

    adopted_ids = candidate_ids & delivered_ids
    adopted: List[Dict[str, Any]] = []
    unused: List[Dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        if str(item.get("evidence_id") or "") in adopted_ids:
            item["adoption_status"] = "adopted"
            adopted.append(item)
        else:
            item["adoption_status"] = "unused"
            unused.append(item)

    return {
        "projection_version": "run_evidence_projection_v1",
        "projection_source": projection_source,
        "retrieved_rag_candidates": candidates,
        "adopted_rag": adopted,
        "unused_rag": unused,
        "tool_evidence_links": tool_links,
        "withheld": withheld,
        "violation": violations,
        "counts": {
            "retrieved_rag_candidates": len(candidates),
            "adopted_rag": len(adopted),
            "unused_rag": len(unused),
            "tool_evidence_links": len(tool_links),
            "withheld": len(withheld),
            "violation": len(violations),
        },
    }
