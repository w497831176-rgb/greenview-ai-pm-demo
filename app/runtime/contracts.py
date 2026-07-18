"""Machine-enforced V1.8 runtime contracts."""

from __future__ import annotations

import hashlib
import json
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    raw = value if isinstance(value, str) else canonical_json(value)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def stable_id(prefix: str, value: Any) -> str:
    return f"{prefix}_{content_hash(value)[:20]}"


class RuntimePath(str, Enum):
    CONSULTATION = "consultation"
    CONTROLLED_ACTION = "controlled_action"
    EXTENSION_ACCEPTANCE = "extension_acceptance"


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class ToolEffect(str, Enum):
    READ = "read"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


class UsageSource(str, Enum):
    PROVIDER_REPORTED_COMPLETE = "provider_reported_complete"
    PROVIDER_REPORTED_TOTAL_ONLY = "provider_reported_total_only"
    LOCAL_ESTIMATE = "local_estimate"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


class ImmutableModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RouteDecision(ImmutableModel):
    candidates: List[str] = Field(default_factory=list)
    selected_agent_id: str
    reason: str
    confidence: Optional[float] = None
    required_capability_types: List[str] = Field(default_factory=list)


class SkillActivation(ImmutableModel):
    skill_id: int
    version: str
    content_hash: str
    name: str
    match_reason: str
    loaded_resources: List[str] = Field(default_factory=list)


class ToolPolicy(ImmutableModel):
    server_id: Optional[int] = None
    server_name: str
    tool_name: str
    effect: ToolEffect
    risk_level: RiskLevel
    allowed_paths: List[RuntimePath] = Field(default_factory=list)
    requires_confirmation: bool = False
    enabled: bool = True
    policy_reason: str = ""


class ToolInvocation(ImmutableModel):
    invocation_id: str = Field(default_factory=lambda: f"tool_{uuid.uuid4().hex}")
    server_name: str
    tool_name: str
    effect: ToolEffect
    arguments: Dict[str, Any] = Field(default_factory=dict)
    discovery_status: str = "not_applicable"
    transport_status: str = "not_started"
    invocation_status: str = "not_started"
    business_status: str = "unknown"
    latency_ms: Optional[int] = None
    result_summary: Optional[str] = None
    error_summary: Optional[str] = None
    receipt_id: Optional[str] = None


class EvidenceItem(ImmutableModel):
    evidence_id: str
    knowledge_id: str
    knowledge_version: str
    document_id: str
    document_version: str
    document_hash: str
    chunk_id: str
    chunk_index: int
    chunk_hash: str
    content_snapshot: str
    retrieval_score: Optional[float] = None
    retrieval_mode: str
    title: str = ""


class EvidenceSet(ImmutableModel):
    items: List[EvidenceItem] = Field(default_factory=list)
    query: str = ""
    retrieval_status: str = "not_requested"

    def by_id(self) -> Dict[str, EvidenceItem]:
        return {item.evidence_id: item for item in self.items}


class Citation(ImmutableModel):
    index: int
    evidence_id: str
    label: str
    title: str
    document_id: str
    document_version: str
    chunk_id: str
    chunk_index: int
    content_snapshot: str
    retrieval_score: Optional[float] = None
    retrieval_mode: str


class PriceSnapshot(ImmutableModel):
    price_snapshot_id: Optional[str] = None
    model_id: str
    currency: Optional[str] = None
    effective_date: Optional[str] = None
    input_price_per_1m: Optional[float] = None
    cached_input_price_per_1m: Optional[float] = None
    output_price_per_1m: Optional[float] = None
    reasoning_price_per_1m: Optional[float] = None
    source_note: Optional[str] = None


class CostEntry(ImmutableModel):
    stage: str
    provider: str
    requested_model: Optional[str] = None
    response_model: Optional[str] = None
    model_policy_version: str
    usage_source: UsageSource
    input_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    local_estimate_tokens: Optional[int] = None
    price_snapshot: Optional[PriceSnapshot] = None
    formula: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    availability_note: str


class ApprovalEvent(ImmutableModel):
    proposal_id: str
    decision: str
    actor: str
    parameter_hash: str
    comment: Optional[str] = None
    decided_at: str


class ActionProposal(ImmutableModel):
    proposal_id: str
    session_id: str
    trace_id: Optional[str] = None
    release_id: Optional[str] = None
    action_type: str
    risk_level: RiskLevel
    payload: Dict[str, Any]
    parameter_hash: str
    idempotency_key: str
    status: str = "pending_confirmation"


class ActionReceipt(ImmutableModel):
    receipt_id: str
    proposal_id: str
    idempotency_key: str
    status: str
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    result: Dict[str, Any] = Field(default_factory=dict)
    committed_at: Optional[str] = None
    error_summary: Optional[str] = None

    @property
    def may_claim_success(self) -> bool:
        return self.status == "committed" and bool(self.resource_id)


class RunConfigSnapshot(ImmutableModel):
    snapshot_id: str
    release_id: str
    snapshot_hash: str
    session_id: str
    config: Dict[str, Any]
    created_at: str


class RunState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    trace_id: str
    session_id: str
    snapshot_id: str
    path: RuntimePath
    route_decision: Optional[RouteDecision] = None
    selected_agent: Optional[Dict[str, Any]] = None
    activated_skills: List[SkillActivation] = Field(default_factory=list)
    retrieval_evidence: EvidenceSet = Field(default_factory=EvidenceSet)
    tool_invocations: List[ToolInvocation] = Field(default_factory=list)
    pending_actions: List[ActionProposal] = Field(default_factory=list)
    approval_events: List[ApprovalEvent] = Field(default_factory=list)
    action_receipts: List[ActionReceipt] = Field(default_factory=list)
    model_calls: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    cost_entries: List[CostEntry] = Field(default_factory=list)
    status: RunStatus = RunStatus.CREATED
    next_step: Optional[str] = None


class RunEvidenceLedger(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    session_id: str
    config_snapshot: Dict[str, Any]
    route_decision: Optional[Dict[str, Any]] = None
    activated_skills: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    tool_invocations: List[Dict[str, Any]] = Field(default_factory=list)
    action_proposals: List[Dict[str, Any]] = Field(default_factory=list)
    approval_events: List[Dict[str, Any]] = Field(default_factory=list)
    action_receipts: List[Dict[str, Any]] = Field(default_factory=list)
    model_calls: List[Dict[str, Any]] = Field(default_factory=list)
    citation_links: List[Dict[str, Any]] = Field(default_factory=list)
    cost_entries: List[Dict[str, Any]] = Field(default_factory=list)
    evaluation_results: List[Dict[str, Any]] = Field(default_factory=list)
    contract_violations: List[Dict[str, Any]] = Field(default_factory=list)
