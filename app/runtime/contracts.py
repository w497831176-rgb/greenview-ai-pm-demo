"""Machine-enforced V1.8 runtime contracts."""

from __future__ import annotations

import hashlib
import json
import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

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


class RuntimeLane(str, Enum):
    SAFETY_HANDOFF = "A_SAFETY_HANDOFF"
    PROPERTY_GOVERNED = "B_PROPERTY_GOVERNED"
    ISOLATED_GENERAL = "C_ISOLATED_GENERAL"


class LaneDecisionSource(str, Enum):
    ROUTER_MODEL = "router_model"
    STRUCTURED_STATE = "structured_state"


class LaneReasonCode(str, Enum):
    IMMINENT_SAFETY_RISK = "imminent_safety_risk"
    PROPERTY_SERVICE_REQUIRED = "property_service_required"
    NON_PROPERTY_GENERAL = "non_property_general"
    UNSAFE_NON_PROPERTY_REQUEST = "unsafe_non_property_request"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class RequestKind(str, Enum):
    EMERGENCY = "emergency"
    FACT = "fact"
    REALTIME_READ = "realtime_read"
    STATE_CHANGE = "state_change"
    GENERAL = "general"
    UNSAFE_REQUEST = "unsafe_request"
    AMBIGUOUS = "ambiguous"


class AllowedDomain(str, Enum):
    SAFETY = "safety"
    PROPERTY = "property"
    ISOLATED_GENERAL = "isolated_general"
    NONE = "none"


class ResponseMode(str, Enum):
    HUMAN_HANDOFF = "human_handoff"
    EMERGENCY_HANDOFF = "emergency_handoff"
    CLARIFY_ONLY = "clarify_only"
    GROUNDED_ANSWER = "grounded_answer"
    REALTIME_READ = "realtime_read"
    CONTROLLED_WRITE = "controlled_write"
    SAFE_GENERAL = "safe_general"
    SAFE_REFUSAL = "safe_refusal"


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
    PROVIDER_ACTUAL = "provider_actual"
    ESTIMATED = "estimated"
    PROVIDER_REPORTED_COMPLETE = "provider_reported_complete"
    PROVIDER_REPORTED_TOTAL_ONLY = "provider_reported_total_only"
    LOCAL_ESTIMATE = "local_estimate"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


class ImmutableModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class HandoffKind(str, Enum):
    USER_REQUESTED = "user_requested"
    SAFETY_RISK = "safety_risk"


class HandoffExecutionContract(ImmutableModel):
    """Immutable A-lane subtype used by execution and evidence projection."""

    kind: HandoffKind
    reason_code: Literal["user_requested", "safety_risk"]
    queue: Literal["property_service", "emergency"]
    safety_override: bool
    response_mode: ResponseMode


class RouteDecision(ImmutableModel):
    candidates: List[str] = Field(default_factory=list)
    selected_agent_id: str
    reason: str
    confidence: Optional[float] = None
    required_capability_types: List[str] = Field(default_factory=list)


class LaneDecision(ImmutableModel):
    """The Router's coarse domain decision; downstream fields cannot invalidate it."""

    model_config = ConfigDict(frozen=True, extra="ignore")
    lane: RuntimeLane
    business_intent: Optional[str] = Field(default=None, max_length=160)
    reason: Optional[str] = Field(default=None, max_length=160)
    decision_source: LaneDecisionSource = LaneDecisionSource.ROUTER_MODEL


class AnswerContract(ImmutableModel):
    """Deterministic response and evidence boundary derived from LaneDecision."""

    response_mode: ResponseMode
    evidence_required: bool
    evidence_requirements: List[str] = Field(default_factory=list)
    skill_policy: Literal["selected", "skipped"]
    rag_policy: Literal["selected", "skipped"]
    tool_policy: Literal["selected", "skipped"]
    write_policy: Literal["allowed_after_confirmation", "forbidden"]
    handoff_policy: Literal["required", "optional", "skipped"]
    forbidden_claims: List[str] = Field(default_factory=list)
    decision_reason: str = Field(min_length=1, max_length=240)


class SkillCapabilityDecision(ImmutableModel):
    status: Literal["selected", "skipped"]
    reason_code: str = Field(min_length=1)
    details: Dict[str, Any] = Field(default_factory=dict)


class RagCapabilityDecision(ImmutableModel):
    status: Literal["selected", "skipped"]
    reason_code: str = Field(min_length=1)
    details: Dict[str, Any] = Field(default_factory=dict)


class ToolCapabilityDecision(ImmutableModel):
    status: Literal["selected", "skipped"]
    reason_code: str = Field(min_length=1)
    details: Dict[str, Any] = Field(default_factory=dict)


class WriteCapabilityDecision(ImmutableModel):
    status: Literal["required", "not_required"]
    reason_code: str = Field(min_length=1)
    details: Dict[str, Any] = Field(default_factory=dict)


class HandoffCapabilityDecision(ImmutableModel):
    status: Literal["available", "required", "not_required"]
    reason_code: str = Field(min_length=1)
    details: Dict[str, Any] = Field(default_factory=dict)


class CapabilityDecision(ImmutableModel):
    selected_agent_id: Optional[str] = None
    skill: SkillCapabilityDecision
    rag: RagCapabilityDecision
    tool: ToolCapabilityDecision
    write: WriteCapabilityDecision
    handoff: HandoffCapabilityDecision


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


class ToolPlan(ImmutableModel):
    """Deterministic plan compiled from published tool metadata."""

    plan_id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4().hex}")
    agent_id: str
    server_name: str
    tool_name: str
    effect: ToolEffect
    execution_mode: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    missing_required: List[str] = Field(default_factory=list)
    schema_errors: List[str] = Field(default_factory=list)
    match_score: float = 0.0
    match_reason: str = ""
    planner_source: str = "published_tool_metadata"
    result_contract: Dict[str, Any] = Field(default_factory=dict)


class ToolInvocation(ImmutableModel):
    invocation_id: str = Field(default_factory=lambda: f"tool_{uuid.uuid4().hex}")
    plan_id: Optional[str] = None
    server_name: str
    tool_name: str
    effect: ToolEffect
    arguments: Dict[str, Any] = Field(default_factory=dict)
    planner_source: Optional[str] = None
    match_reason: Optional[str] = None
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
    provider_response_model: Optional[str] = None
    thinking_enabled: Optional[bool] = None
    model_policy_version: str
    usage_source: UsageSource
    input_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    input_cache_hit_tokens: Optional[int] = None
    input_cache_miss_tokens: Optional[int] = None
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
    lane_decision: Optional[LaneDecision] = None
    answer_contract: Optional[AnswerContract] = None
    route_decision: Optional[RouteDecision] = None
    capability_decision: Optional[CapabilityDecision] = None
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
    lane_decision: Optional[Dict[str, Any]] = None
    answer_contract: Optional[Dict[str, Any]] = None
    route_decision: Optional[Dict[str, Any]] = None
    capability_decision: Optional[Dict[str, Any]] = None
    activated_skills: List[Dict[str, Any]] = Field(default_factory=list)
    skill_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    tool_invocations: List[Dict[str, Any]] = Field(default_factory=list)
    action_proposals: List[Dict[str, Any]] = Field(default_factory=list)
    approval_events: List[Dict[str, Any]] = Field(default_factory=list)
    handoff_events: List[Dict[str, Any]] = Field(default_factory=list)
    action_receipts: List[Dict[str, Any]] = Field(default_factory=list)
    model_calls: List[Dict[str, Any]] = Field(default_factory=list)
    citation_links: List[Dict[str, Any]] = Field(default_factory=list)
    cost_entries: List[Dict[str, Any]] = Field(default_factory=list)
    evaluation_results: List[Dict[str, Any]] = Field(default_factory=list)
    contract_violations: List[Dict[str, Any]] = Field(default_factory=list)
    system_observations: List[Dict[str, Any]] = Field(default_factory=list)
    badcase_links: List[Dict[str, Any]] = Field(default_factory=list)
