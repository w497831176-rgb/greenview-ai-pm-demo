"""No-model Agno factory assembly checks for the V1.8 runtime."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TEMP_DIR = tempfile.TemporaryDirectory(
    prefix="yiai-v180-factory-",
    ignore_cleanup_errors=True,
)
os.environ["PROPERTY_DATA_DIR"] = TEMP_DIR.name
os.environ["RUNTIME_ENGINE"] = "v18"

from agno.factory import RequestContext

from db.property_db import init_db


def test_agent_factory_assembly() -> None:
    from app.runtime.agent_factory import build_runtime_agent

    agent = build_runtime_agent(
        RequestContext(
            session_id="factory-agent",
            user_id="contract",
            input={
                "agent_id": "customer_service",
                "message": "物业服务电话是多少？",
            },
        )
    )
    assert agent.id == "customer_service"
    assert agent.model is not None


def test_composite_workflow_assembly() -> None:
    from app.runtime.workflow_factory import build_runtime_workflow

    workflow = build_runtime_workflow(
        RequestContext(
            session_id="factory-workflow",
            user_id="contract",
            input={
                "path": "composite_acceptance",
                "message": "组合验收",
                "consultation_message": "儿童滑梯安全制度是什么？",
                "action_message": "3号楼漏水，请创建工单",
            },
        )
    )
    assert [step.name for step in workflow.steps] == [
        "resolve_snapshot",
        "execute_consultation",
        "collect_action_proposal",
        "commit_confirmed_action",
    ]
    commit_step = workflow.steps[-1]
    assert commit_step.requires_confirmation is True


def main() -> None:
    try:
        init_db()
        from app.runtime.release_compiler import ensure_bootstrap_release

        ensure_bootstrap_release()
        test_agent_factory_assembly()
        print("PASS test_agent_factory_assembly")
        test_composite_workflow_assembly()
        print("PASS test_composite_workflow_assembly")
    finally:
        TEMP_DIR.cleanup()
    print("V1.8 Agno factory no-model contracts passed.")


if __name__ == "__main__":
    main()
