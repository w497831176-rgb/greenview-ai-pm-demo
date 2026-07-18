"""Agno WorkflowFactory surfaces for the three YIAI runtime paths."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from agno.factory import RequestContext
from agno.workflow import OnReject, Workflow, WorkflowFactory
from agno.workflow.step import Step
from agno.workflow.types import StepInput, StepOutput
from pydantic import BaseModel

from app.runtime.coordinator import RuntimeCoordinator
from app.runtime.release_compiler import publish_compiled_release
from app.runtime.snapshot_resolver import resolve_snapshot
from app.settings import agent_db


class RuntimeWorkflowInput(BaseModel):
    path: str = "consultation"
    message: str
    consultation_message: Optional[str] = None
    action_message: Optional[str] = None
    created_by: str = "workflow-operator"


async def _consume_coordinator(
    message: str,
    session_id: str,
    user_id: str,
) -> Dict[str, Any]:
    final: Dict[str, Any] = {}
    answer = ""
    async for event in RuntimeCoordinator().stream(message, session_id, user_id):
        if event.startswith("event: delta"):
            try:
                payload = json.loads(event.split("data: ", 1)[1])
                answer += str(payload.get("content") or "")
            except Exception:
                pass
        elif event.startswith("event: done"):
            final = json.loads(event.split("data: ", 1)[1])
        elif event.startswith("event: error"):
            payload = json.loads(event.split("data: ", 1)[1])
            raise RuntimeError(str(payload.get("error") or "runtime error"))
    return {"answer": answer, "done": final}


def build_runtime_workflow(ctx: RequestContext) -> Workflow:
    raw = ctx.input or {}
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    config = RuntimeWorkflowInput.model_validate(raw)
    session_id = ctx.session_id or f"workflow-{ctx.user_id or 'anonymous'}"
    user_id = ctx.user_id or "workflow-user"
    snapshot = resolve_snapshot(session_id)

    def resolve_step(step_input: StepInput) -> StepOutput:
        del step_input
        return StepOutput(
            step_name="resolve_snapshot",
            content={
                "release_id": snapshot.release_id,
                "snapshot_id": snapshot.snapshot_id,
                "snapshot_hash": snapshot.snapshot_hash,
                "path": config.path,
            },
        )

    async def execute_consultation(step_input: StepInput) -> StepOutput:
        del step_input
        result = await _consume_coordinator(
            config.consultation_message or config.message, session_id, user_id
        )
        return StepOutput(
            step_name="execute_consultation",
            content=result,
            success=result.get("done", {}).get("status") == "complete",
        )

    async def collect_action(step_input: StepInput) -> StepOutput:
        del step_input
        result = await _consume_coordinator(
            config.action_message or config.message, session_id, user_id
        )
        return StepOutput(
            step_name="collect_action_proposal",
            content=result,
            success=result.get("done", {}).get("status") in {"paused", "complete"},
        )

    async def commit_confirmed_action(step_input: StepInput) -> StepOutput:
        del step_input
        result = await _consume_coordinator(
            "确认提交", session_id, user_id
        )
        receipt = (result.get("done") or {}).get("action_receipts") or []
        success = bool(receipt and receipt[-1].get("status") == "committed")
        return StepOutput(
            step_name="commit_confirmed_action",
            content=result,
            success=success,
            error=None if success else "confirmed action produced no committed receipt",
        )

    def publish_extension(step_input: StepInput) -> StepOutput:
        del step_input
        release = publish_compiled_release(created_by=config.created_by)
        published = release.get("status") == "published"
        return StepOutput(
            step_name="publish_extension",
            content={
                "release_id": release.get("release_id"),
                "status": release.get("status"),
                "validation": release.get("validation"),
                "effective_on": "new_session",
            },
            success=published,
            error=None if published else "release validation failed",
        )

    confirmed_action_steps = [
        Step(name="collect_action_proposal", executor=collect_action),
        Step(
            name="commit_confirmed_action",
            executor=commit_confirmed_action,
            requires_confirmation=True,
            confirmation_message="确认执行已持久化的业务 ActionProposal？",
            on_reject=OnReject.cancel,
        ),
    ]
    if config.path == "controlled_action":
        steps = [
            Step(name="resolve_snapshot", executor=resolve_step),
            *confirmed_action_steps,
        ]
    elif config.path == "composite_acceptance":
        # One Agno Workflow run deliberately composes the read-only evidence
        # path and the separately confirmed write path. The child coordinator
        # traces remain independently auditable under the Workflow run.
        steps = [
            Step(name="resolve_snapshot", executor=resolve_step),
            Step(name="execute_consultation", executor=execute_consultation),
            *confirmed_action_steps,
        ]
    elif config.path == "extension_acceptance":
        steps = [
            Step(name="resolve_snapshot", executor=resolve_step),
            Step(name="validate_and_publish", executor=publish_extension),
        ]
    else:
        steps = [
            Step(name="resolve_snapshot", executor=resolve_step),
            Step(name="execute_consultation", executor=execute_consultation),
        ]
    return Workflow(
        id="yiai-runtime",
        name="YIAI V1.8 Runtime",
        description=(
            "Published-snapshot consultation, confirmed action and dynamic "
            "extension workflows."
        ),
        db=agent_db,
        steps=steps,
    )


runtime_workflow_factory = WorkflowFactory(
    id="yiai-runtime",
    db=agent_db,
    factory=build_runtime_workflow,
    input_schema=RuntimeWorkflowInput,
    name="YIAI V1.8 Runtime",
    description="Builds the correct three-path Workflow from a pinned snapshot.",
)
