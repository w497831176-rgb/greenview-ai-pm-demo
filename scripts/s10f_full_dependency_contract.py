"""Full-dependency S10-F acceptance helpers.

This module is imported only by the S10-F runner after FastAPI, Agno and the
YIAI modules have loaded against a verified temporary PROPERTY_DATA_DIR.
Every model/runtime boundary is replaced with a deterministic local stub.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


def run_full_dependency_contract(core: Any, check: Callable[[str, object], None]) -> None:
    previous_logging_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    import app.badcases as badcases
    import app.evaluations as evaluations
    import app.main as main_api
    import app.model_configs as model_configs
    import app.observability as observability
    import app.runtime.agent_factory as agent_factory
    import app.runtime.legacy_chat as legacy_chat
    import app.runtime.workflow_factory as workflow_factory
    from db import property_db
    from fastapi import HTTPException
    from fastapi.testclient import TestClient

    if not core.HAS_RUNTIME_DEPENDENCIES:
        raise AssertionError("full dependency contract entered fallback mode")

    counters = {
        "badcase_model": 0,
        "badcase_build_model": 0,
        "retest_runtime": 0,
        "evaluation_runtime": 0,
        "ab_build_model": 0,
        "workflow_runtime": 0,
        "workflow_publish": 0,
        "case_update": 0,
        "case_action": 0,
        "draft_create": 0,
        "owner_chat": 0,
    }
    current_case = {
        "id": 1,
        "status": "pending",
        "title": "S10-F full dependency fixture",
        "description": "deterministic fixture",
        "feedback_reason": "",
        "original_query": "fixture question",
        "ai_response": "",
        "context_json": {},
        "session_id": "fixture-session",
        "category": "other",
    }

    def load_case(_case_id: int) -> dict[str, Any]:
        return dict(current_case)

    def update_case(_case_id: int, **changes: Any) -> dict[str, Any]:
        counters["case_update"] += 1
        return {**current_case, **changes}

    def record_case_action(*_args: Any, **_kwargs: Any) -> None:
        counters["case_action"] += 1

    def create_draft(**kwargs: Any) -> dict[str, Any]:
        counters["draft_create"] += 1
        return {"id": 91, **kwargs}

    async def forbidden_badcase_model(*_args: Any, **_kwargs: Any):
        counters["badcase_model"] += 1
        raise AssertionError("blocked Badcase model boundary was entered")

    def forbidden_badcase_build(*_args: Any, **_kwargs: Any):
        counters["badcase_build_model"] += 1
        raise AssertionError("blocked Badcase build_model boundary was entered")

    async def forbidden_retest(*_args: Any, **_kwargs: Any):
        counters["retest_runtime"] += 1
        raise AssertionError("blocked Badcase retest runtime was entered")

    async def forbidden_evaluation(*_args: Any, **_kwargs: Any):
        counters["evaluation_runtime"] += 1
        raise AssertionError("blocked Evaluation runtime was entered")

    def forbidden_ab_build(*_args: Any, **_kwargs: Any):
        counters["ab_build_model"] += 1
        raise AssertionError("blocked A/B build_model boundary was entered")

    async def forbidden_workflow_runtime(*_args: Any, **_kwargs: Any):
        counters["workflow_runtime"] += 1
        raise AssertionError("blocked Workflow Provider step was entered")

    def safe_workflow_publish(**_kwargs: Any) -> dict[str, Any]:
        counters["workflow_publish"] += 1
        return {
            "release_id": "rr-fixture",
            "status": "published",
            "validation": {"valid": True},
        }

    originals = {
        "badcases_load": badcases._load_case,
        "badcases_get": badcases.db_get_badcase,
        "badcases_update": badcases.db_update_badcase,
        "badcases_record": badcases._record_action,
        "badcases_create_draft": badcases.db_create_knowledge_draft,
        "badcases_enrich": badcases._enrich_badcase,
        "badcases_find_darwin": badcases._find_darwin_skill,
        "badcases_start_darwin": badcases.start_darwin_operation,
        "badcases_persist_darwin": badcases.persist_darwin_operation,
        "badcases_llm": badcases._llm_generate,
        "badcases_build": badcases.build_model,
        "badcases_retest": badcases._consume_chat_stream,
        "badcases_gate": badcases._background_budget_gate,
        "eval_case": evaluations.get_evaluation_case,
        "eval_chat": evaluations._run_real_chat,
        "model_build": model_configs.build_model,
        "obs_conn": observability._get_conn,
        "obs_thresholds": observability.get_budget_thresholds,
        "obs_bounds": observability._period_bounds,
        "obs_gate": observability._background_budget_gate,
        "workflow_gate": workflow_factory._background_budget_gate,
        "workflow_resolve": workflow_factory.resolve_snapshot,
        "workflow_runtime": workflow_factory._consume_coordinator,
        "workflow_publish": workflow_factory.publish_compiled_release,
        "workflow_agent_db": workflow_factory.agent_db,
        "legacy_stream": legacy_chat._stream_agent_response,
        "list_skills": property_db.list_skills,
        "list_mcp": property_db.list_mcp_servers,
    }

    test_client = TestClient(main_api.app, raise_server_exceptions=True)
    ledger_temp = tempfile.TemporaryDirectory(prefix="yiai-s10f-gates-")
    ledger_root = Path(ledger_temp.name)
    ledger_paths: dict[str, Path] = {}

    fixed_bounds = {
        "today": {
            "start": "2026-08-02 00:00:00",
            "end": "2026-08-02 23:59:59",
            "days": 1,
        },
        "last_7_days": {
            "start": "2026-07-27 00:00:00",
            "end": "2026-08-02 23:59:59",
            "days": 7,
        },
        "this_month": {
            "start": "2026-08-01 00:00:00",
            "end": "2026-08-02 23:59:59",
            "days": 2,
        },
    }

    def make_connection(path: Path):
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        return connection

    def configure_ledger(kind: str) -> Path:
        path = ledger_root / f"{kind}.db"
        if not path.exists():
            connection = make_connection(path)
            core.create_schema(connection)
            if kind == "available":
                core.insert_attempt(
                    connection,
                    trace_id="gate-available",
                    created_at="2026-08-02T08:00:00+08:00",
                    calculated_direct_cost=0.000081,
                )
            elif kind == "data_quality_error":
                core.insert_attempt(
                    connection,
                    trace_id="gate-incomplete",
                    created_at="2026-08-02T08:00:00+08:00",
                    hit=None,
                    miss=20,
                    output=30,
                    total=50,
                    calculated_direct_cost=None,
                )
            elif kind == "reconciliation_attention":
                core.insert_attempt(
                    connection,
                    trace_id="gate-price-missing",
                    created_at="2026-08-02T08:00:00+08:00",
                    priced=False,
                    calculated_direct_cost=None,
                )
            elif kind == "hard_limit":
                core.insert_attempt(
                    connection,
                    trace_id="gate-hard-limit",
                    created_at="2026-08-02T08:00:00+08:00",
                    calculated_direct_cost=0.000081,
                )
            connection.commit()
            connection.close()
        ledger_paths[kind] = path

        if kind == "query_failure":
            def fail_connection():
                raise sqlite3.OperationalError("deterministic ledger failure")

            observability._get_conn = fail_connection
        else:
            observability._get_conn = lambda selected=path: make_connection(selected)
        observability._period_bounds = lambda: fixed_bounds
        threshold = 0.00005 if kind == "hard_limit" else 100.0
        observability.get_budget_thresholds = lambda selected=threshold: {
            "daily_threshold_cny": selected,
            "monthly_threshold_cny": selected,
            "per_call_threshold_cny": selected,
        }
        return path

    def provider_attempt_count(path: Path) -> int:
        connection = make_connection(path)
        try:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM model_calls WHERE record_kind='provider_attempt'"
                ).fetchone()[0]
            )
        finally:
            connection.close()

    def runtime_provider_attempt_count() -> int:
        """Count the real accounting DB used by Provider attempt writers."""
        connection = property_db._get_conn()
        try:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM model_calls WHERE record_kind='provider_attempt'"
                ).fetchone()[0]
            )
        finally:
            connection.close()

    def set_case(status: str, category: str = "other") -> None:
        current_case.update(status=status, category=category)

    def paid_http_calls(
        budget_ledger_path: Path,
    ) -> list[tuple[str, Any, bool]]:
        calls: list[tuple[str, Any, bool]] = []

        def append_call(name: str, invoke: Callable[[], Any]) -> None:
            before_budget_ledger = provider_attempt_count(budget_ledger_path)
            before_runtime_ledger = runtime_provider_attempt_count()
            response = invoke()
            calls.append(
                (
                    name,
                    response,
                    provider_attempt_count(budget_ledger_path)
                    == before_budget_ledger
                    and runtime_provider_attempt_count() == before_runtime_ledger,
                )
            )

        set_case("pending")
        append_call(
            "Badcase classify",
            lambda: test_client.post(
                "/api/badcases/1/classify", json={"auto": True}
            ),
        )
        for suffix in ("darwin-fix", "darwin-optimize", "darwin"):
            set_case("classified")
            append_call(
                f"Badcase {suffix}",
                lambda selected=suffix: test_client.post(
                    f"/api/badcases/1/{selected}", json={}
                ),
            )
        set_case("classified", "knowledge_gap")
        append_call(
            "Badcase automatic knowledge",
            lambda: test_client.post(
                "/api/badcases/1/extract-knowledge", json={"auto": True}
            ),
        )
        for suffix in ("retry", "switch-model-retry"):
            set_case("classified")
            append_call(
                f"Badcase {suffix}",
                lambda selected=suffix: test_client.post(
                    f"/api/badcases/1/{selected}",
                    json={"user_message": "fixture question"},
                ),
            )
        set_case("fixing")
        append_call(
            "Badcase retest",
            lambda: test_client.post(
                "/api/badcases/1/retest",
                json={"user_message": "fixture question"},
            ),
        )
        set_case("pending")
        append_call(
            "Badcase tool check",
            lambda: test_client.post("/api/badcases/1/check-tools"),
        )
        append_call(
            "Evaluation run",
            lambda: test_client.post("/api/evaluations/cases/1/run", json={}),
        )
        for path in ("/api/model-configs/ab-test", "/api/models/ab-test"):
            append_call(
                f"A/B {path}",
                lambda selected=path: test_client.post(
                    selected, json={"prompt": "fixture"}
                ),
            )
        return calls

    try:
        badcases._load_case = load_case
        badcases.db_get_badcase = load_case
        badcases.db_update_badcase = update_case
        badcases._record_action = record_case_action
        badcases.db_create_knowledge_draft = create_draft
        badcases._enrich_badcase = lambda case: case
        badcases._find_darwin_skill = lambda: None
        badcases.start_darwin_operation = lambda **_kwargs: None
        badcases.persist_darwin_operation = lambda **_kwargs: None
        badcases._llm_generate = forbidden_badcase_model
        badcases.build_model = forbidden_badcase_build
        badcases._consume_chat_stream = forbidden_retest
        property_db.list_skills = lambda: []
        property_db.list_mcp_servers = lambda: []
        evaluations.get_evaluation_case = lambda _case_id: {
            "id": 1,
            "case_key": "s10f-full",
            "status": "active",
            "user_message": "fixture",
        }
        evaluations._run_real_chat = forbidden_evaluation
        model_configs.build_model = forbidden_ab_build
        workflow_factory._consume_coordinator = forbidden_workflow_runtime

        # Real _background_budget_gate over isolated SQLite evidence.
        expected = {
            "available": ("available", True, None),
            "query_failure": ("unavailable", False, 503),
            "data_quality_error": ("unavailable", False, 503),
            "reconciliation_attention": ("unavailable", False, 503),
            "hard_limit": ("available", False, 403),
        }
        for kind, (budget_status, allowed, http_status) in expected.items():
            path = configure_ledger(kind)
            result = observability._background_budget_gate(f"full-{kind}")
            check(
                f"real temporary-ledger gate {kind}",
                result["budget_status"] == budget_status
                and result["allowed"] is allowed
                and result["http_status"] == http_status,
            )
            if kind == "available":
                continue
            before_provider = provider_attempt_count(path)
            before_runtime_provider = runtime_provider_attempt_count()
            before_models = {
                key: counters[key]
                for key in (
                    "badcase_model",
                    "badcase_build_model",
                    "retest_runtime",
                    "evaluation_runtime",
                    "ab_build_model",
                )
            }
            before_lifecycle = (
                counters["case_update"],
                counters["case_action"],
                counters["draft_create"],
            )
            for name, response, zero_provider_delta in paid_http_calls(path):
                check(
                    f"{name} returns {http_status} for {kind}",
                    response.status_code == http_status,
                )
                check(
                    f"{name} creates zero Provider attempts for {kind}",
                    zero_provider_delta,
                )
            check(
                f"{kind} stops every model/runtime stub",
                all(counters[key] == value for key, value in before_models.items()),
            )
            check(
                f"{kind} does not advance Badcase lifecycle",
                before_lifecycle
                == (
                    counters["case_update"],
                    counters["case_action"],
                    counters["draft_create"],
                ),
            )
            check(
                f"{kind} creates zero Provider attempts",
                provider_attempt_count(path) == before_provider
                and runtime_provider_attempt_count() == before_runtime_provider,
            )

        # Direct AgentOS execution has no product consumer and is always 410.
        def forbidden_budget_gate(_strategy: str):
            raise AssertionError("disabled/read-only AgentOS route queried budget")

        workflow_factory._background_budget_gate = forbidden_budget_gate
        direct_paths = (
            "/agents/runtime-agent/runs",
            "/agents/runtime-agent/runs/run-fixture/continue",
            "/agents/runtime-agent/runs/run-fixture/resume",
            "/workflows/yiai-runtime/runs",
            "/workflows/yiai-runtime/runs/run-fixture/continue",
            "/workflows/yiai-runtime/runs/run-fixture/resume",
            "/eval-runs",
            "/optimize-memories",
        )
        for path in direct_paths:
            response = test_client.post(path, content=b"malformed body")
            check(
                f"unused direct AgentOS surface is stable 410: {path}",
                response.status_code == 410,
            )

        # Construction and read-only history must not consult the budget gate.
        workflow_factory.resolve_snapshot = lambda _session_id: SimpleNamespace(
            release_id="rr-fixture",
            snapshot_id="snapshot-fixture",
            snapshot_hash="hash-fixture",
        )

        class FakeContext:
            def __init__(self, payload: dict[str, Any]):
                self.input = payload
                self.session_id = "workflow-fixture"
                self.user_id = "operator"

        consultation_workflow = workflow_factory.build_runtime_workflow(
            FakeContext({"path": "consultation", "message": "fixture"})
        )
        check(
            "WorkflowFactory construction is budget independent",
            consultation_workflow is not None,
        )
        # Exercise real Agno routes against an isolated SQLite AgentOS DB. The
        # production app uses Postgres, which is deliberately unavailable in
        # this --network none container.
        workflow_factory.publish_compiled_release = safe_workflow_publish
        from agno.db.sqlite import SqliteDb
        from agno.os import AgentOS
        from agno.workflow import WorkflowFactory

        agno_db = SqliteDb(
            db_file=str(ledger_root / "agentos-history.db")
        )
        workflow_factory.agent_db = agno_db
        isolated_factory = WorkflowFactory(
            id="yiai-runtime",
            db=agno_db,
            factory=workflow_factory.build_runtime_workflow,
            input_schema=workflow_factory.RuntimeWorkflowInput,
            name="YIAI S10-F isolated Workflow",
        )
        isolated_os = AgentOS(
            name="YIAI S10-F isolated AgentOS",
            authorization=False,
            scheduler=False,
            tracing=False,
            db=agno_db,
            workflows=[isolated_factory],
        )
        with TestClient(
            isolated_os.get_app(), raise_server_exceptions=True
        ) as isolated_client:
            before_http_publish = counters["workflow_publish"]
            extension_http = isolated_client.post(
                "/workflows/yiai-runtime/runs",
                data={
                    "message": "fixture",
                    "session_id": "s10f-agentos-history",
                    "stream": "false",
                    "factory_input": json.dumps(
                        {
                            "path": "extension_acceptance",
                            "message": "fixture",
                        }
                    ),
                },
            )
            check(
                "real Agno no-model Workflow route executes",
                extension_http.status_code == 200
                and counters["workflow_publish"] == before_http_publish + 1,
            )
            workflow_list = isolated_client.get(
                "/workflows/yiai-runtime/runs",
                params={"session_id": "s10f-agentos-history"},
            )
            check(
                "Agno Workflow run list remains readable under bad budget",
                workflow_list.status_code == 200,
            )
            payload = workflow_list.json()
            runs = (
                payload
                if isinstance(payload, list)
                else payload.get("data")
                or payload.get("runs")
                or payload.get("items")
                or []
            )
            run_id = next(
                (
                    str(item.get("run_id") or item.get("id"))
                    for item in runs
                    if isinstance(item, dict)
                    and (item.get("run_id") or item.get("id"))
                ),
                None,
            )
            if not run_id:
                try:
                    extension_payload = extension_http.json()
                except Exception:
                    extension_payload = {}
                if isinstance(extension_payload, dict):
                    run_id = extension_payload.get("run_id") or extension_payload.get(
                        "id"
                    )
            check("real Agno no-model run has a run id", bool(run_id))
            workflow_detail = isolated_client.get(
                f"/workflows/yiai-runtime/runs/{run_id}",
                params={"session_id": "s10f-agentos-history"},
            )
            check(
                "Agno Workflow run detail remains readable under bad budget",
                workflow_detail.status_code == 200,
            )
        # Provider-capable steps keep the real gate immediately before runtime.
        workflow_factory._background_budget_gate = observability._background_budget_gate
        for kind, expected_status in (
            ("query_failure", 503),
            ("data_quality_error", 503),
            ("reconciliation_attention", 503),
            ("hard_limit", 403),
        ):
            path = configure_ledger(kind)
            workflow = workflow_factory.build_runtime_workflow(
                FakeContext({"path": "consultation", "message": "fixture"})
            )
            paid_step = next(
                step for step in workflow.steps if step.name == "execute_consultation"
            )
            before_runtime = counters["workflow_runtime"]
            before_provider = provider_attempt_count(path)
            before_runtime_provider = runtime_provider_attempt_count()
            try:
                asyncio.run(paid_step.executor(None))
            except HTTPException as exc:
                check(
                    f"Workflow Provider step returns {expected_status} for {kind}",
                    exc.status_code == expected_status,
                )
            else:
                raise AssertionError(f"Workflow Provider step did not block for {kind}")
            check(
                f"Workflow {kind} stops before runtime and Provider attempt",
                counters["workflow_runtime"] == before_runtime
                and provider_attempt_count(path) == before_provider
                and runtime_provider_attempt_count() == before_runtime_provider,
            )

        # extension_acceptance is deterministic and never queries the budget.
        workflow_factory._background_budget_gate = forbidden_budget_gate
        workflow_factory.publish_compiled_release = safe_workflow_publish
        extension = workflow_factory.build_runtime_workflow(
            FakeContext({"path": "extension_acceptance", "message": "fixture"})
        )
        extension_step = next(
            step for step in extension.steps if step.name == "validate_and_publish"
        )
        before_direct_publish = counters["workflow_publish"]
        extension_result = extension_step.executor(None)
        check(
            "non-model extension path executes without the budget gate",
            bool(extension_result.success)
            and counters["workflow_publish"] == before_direct_publish + 1,
        )

        # Manual knowledge save is non-model and remains available.
        badcases._background_budget_gate = forbidden_budget_gate
        set_case("classified", "knowledge_gap")
        manual = test_client.post(
            "/api/badcases/1/extract-knowledge",
            json={"auto": False, "content": "operator supplied knowledge"},
        )
        check(
            "manual knowledge save bypasses background budget",
            manual.status_code == 200 and counters["draft_create"] >= 1,
        )
        badcases._background_budget_gate = originals["badcases_gate"]

        # A healthy real ledger allows a deterministic model stub.
        configure_ledger("available")

        async def safe_badcase_model(*_args: Any, **_kwargs: Any):
            counters["badcase_model"] += 1
            return (
                json.dumps(
                    {
                        "suggested_category": "other",
                        "root_cause_hypothesis": "fixture",
                        "repair_path_suggestion": "ops_only",
                        "priority": "low",
                        "root_cause_domain": "unknown",
                    }
                ),
                {},
            )

        badcases._llm_generate = safe_badcase_model
        set_case("pending")
        before_safe = counters["badcase_model"]
        healthy = test_client.post(
            "/api/badcases/1/classify", json={"auto": True}
        )
        check(
            "healthy budget does not block a safe model stub",
            healthy.status_code == 200
            and counters["badcase_model"] == before_safe + 1,
        )

        # Owner chat HTTP never consults the optional-background gate.
        async def safe_owner_stream(*_args: Any, **_kwargs: Any):
            counters["owner_chat"] += 1
            yield 'event: done\ndata: {"status":"complete"}\n\n'

        legacy_chat._stream_agent_response = safe_owner_stream
        observability._background_budget_gate = forbidden_budget_gate
        owner = test_client.post(
            "/api/chat/stream",
            json={"message": "ordinary owner consultation"},
        )
        check(
            "ordinary owner chat HTTP is outside the background budget gate",
            owner.status_code == 200 and counters["owner_chat"] == 1,
        )
        observability._background_budget_gate = originals["obs_gate"]

        # The standalone history verifier proves import order, mode=ro and
        # byte-for-byte immutability in a fresh full-dependency process.
        with tempfile.TemporaryDirectory(prefix="yiai-s10f-history-") as history_dir:
            history_db = Path(history_dir) / "fixed-history.db"
            connection = make_connection(history_db)
            core.create_schema(connection)
            stages = [
                *("router" for _ in range(5)),
                *("agent_selector" for _ in range(5)),
                *("vertical_agent" for _ in range(3)),
                *("badcase_extract_knowledge" for _ in range(2)),
            ]
            chat_traces = [f"history-chat-{index}" for index in range(1, 6)]
            for index in range(15):
                trace_id = (
                    chat_traces[index % 5]
                    if index < 13
                    else f"history-badcase-{index - 12}"
                )
                is_last = index == 14
                hit = 274 if is_last else 273
                miss = 837 if is_last else 824
                output = 456 if is_last else 447
                reasoning = 347 if is_last else 333
                core.insert_attempt(
                    connection,
                    trace_id=trace_id,
                    stage=stages[index],
                    attempt_key=f"history-{index + 1}",
                    hit=hit,
                    miss=miss,
                    output=output,
                    reasoning=reasoning,
                    total=hit + miss + output,
                    calculated_direct_cost=(
                        0.00208292 if is_last else 0.0017
                    ),
                    created_at=(
                        f"2026-08-01T{8 + index // 2:02d}:"
                        f"{(index % 2) * 30:02d}:00+08:00"
                    ),
                )
            connection.commit()
            connection.close()

            from scripts import verify_v182_s10f_history_readonly as verifier

            before = verifier.database_artifact_fingerprint(history_db)
            inherited_env = os.environ.copy()
            inherited_env["PROPERTY_DATA_DIR"] = (
                "/volume3/docker/agno-demo-os/production-fixture"
            )
            inherited_env["DEEPSEEK_API_KEY"] = ""
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.verify_v182_s10f_history_readonly",
                    str(history_db),
                ],
                cwd=str(core.ROOT),
                env=inherited_env,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            after_subprocess = verifier.database_artifact_fingerprint(history_db)
            check(
                "standalone history verifier passes in full dependencies",
                completed.returncode == 0
                and "PASS: S10-F read-only fixed-history verification"
                in completed.stdout,
            )
            evidence_line = next(
                line for line in completed.stdout.splitlines() if line.startswith("{")
            )
            evidence = json.loads(evidence_line)
            check(
                "history verifier never calls init_db and forces isolated data",
                evidence["readonly_guard"]["init_db_calls"] == 0
                and evidence["readonly_guard"]["property_data_dir_forced"] is True
                and evidence["readonly_guard"]["test_database_created"] is False,
            )
            read_only = verifier._read_only_connection(history_db)
            try:
                query_only = read_only.execute("PRAGMA query_only").fetchone()[0]
                try:
                    read_only.execute("CREATE TABLE forbidden_write(id INTEGER)")
                except sqlite3.OperationalError:
                    write_blocked = True
                else:
                    write_blocked = False
            finally:
                read_only.close()
            after_all_readonly_checks = verifier.database_artifact_fingerprint(
                history_db
            )
            check(
                "history database mtime/hash/sidecars remain unchanged",
                before == after_subprocess == after_all_readonly_checks
                and evidence["readonly_guard"][
                    "database_artifacts_unchanged"
                ]
                is True,
            )
            check(
                "verified SQLite connection is query-only mode=ro",
                query_only == 1 and write_blocked,
            )
            for unsafe in (
                Path("/app/data"),
                Path("/volume3/docker/agno-demo-os/test"),
                history_db.parent,
            ):
                try:
                    verifier.assert_safe_test_data_dir(unsafe, history_db)
                except RuntimeError:
                    rejected = True
                else:
                    rejected = False
                check(f"unsafe verifier data path is rejected: {unsafe}", rejected)

        check(
            "AgentFactory constructor no longer owns a background budget gate",
            not hasattr(agent_factory, "_background_budget_gate"),
        )
    finally:
        badcases._load_case = originals["badcases_load"]
        badcases.db_get_badcase = originals["badcases_get"]
        badcases.db_update_badcase = originals["badcases_update"]
        badcases._record_action = originals["badcases_record"]
        badcases.db_create_knowledge_draft = originals["badcases_create_draft"]
        badcases._enrich_badcase = originals["badcases_enrich"]
        badcases._find_darwin_skill = originals["badcases_find_darwin"]
        badcases.start_darwin_operation = originals["badcases_start_darwin"]
        badcases.persist_darwin_operation = originals["badcases_persist_darwin"]
        badcases._llm_generate = originals["badcases_llm"]
        badcases.build_model = originals["badcases_build"]
        badcases._consume_chat_stream = originals["badcases_retest"]
        badcases._background_budget_gate = originals["badcases_gate"]
        evaluations.get_evaluation_case = originals["eval_case"]
        evaluations._run_real_chat = originals["eval_chat"]
        model_configs.build_model = originals["model_build"]
        observability._get_conn = originals["obs_conn"]
        observability.get_budget_thresholds = originals["obs_thresholds"]
        observability._period_bounds = originals["obs_bounds"]
        observability._background_budget_gate = originals["obs_gate"]
        workflow_factory._background_budget_gate = originals["workflow_gate"]
        workflow_factory.resolve_snapshot = originals["workflow_resolve"]
        workflow_factory._consume_coordinator = originals["workflow_runtime"]
        workflow_factory.publish_compiled_release = originals["workflow_publish"]
        workflow_factory.agent_db = originals["workflow_agent_db"]
        legacy_chat._stream_agent_response = originals["legacy_stream"]
        property_db.list_skills = originals["list_skills"]
        property_db.list_mcp_servers = originals["list_mcp"]
        test_client.close()
        ledger_temp.cleanup()
        logging.disable(previous_logging_disable)
