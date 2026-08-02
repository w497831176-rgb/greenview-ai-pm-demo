"""Static S10-G contract checks for reporting truth and paid-entry gates.

This suite imports no application module, opens no network connection and does
not construct a model.  It complements (but never substitutes for) the
temporary-database behavior suite in ``test_v182_s10f_trace_truthfulness.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKS: list[str] = []


def check(name: str, condition: object) -> None:
    if not condition:
        raise AssertionError(name)
    CHECKS.append(name)


def source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def function_node(relative: str, function_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(source(relative), filename=relative)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return node
    raise AssertionError(f"{relative}:{function_name} is missing")


def call_lines(node: ast.AST, called_name: str) -> list[int]:
    lines: list[int] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        target = child.func
        name = target.id if isinstance(target, ast.Name) else target.attr if isinstance(target, ast.Attribute) else None
        if name == called_name:
            lines.append(child.lineno)
    return sorted(lines)


def direct_call_lines(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    called_name: str,
) -> list[int]:
    """Find calls in one function body without counting nested executors."""
    lines: list[int] = []

    class DirectBodyVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, _child: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, _child: ast.AsyncFunctionDef) -> None:
            return

        def visit_Lambda(self, _child: ast.Lambda) -> None:
            return

        def visit_Call(self, child: ast.Call) -> None:
            target = child.func
            name = (
                target.id
                if isinstance(target, ast.Name)
                else target.attr
                if isinstance(target, ast.Attribute)
                else None
            )
            if name == called_name:
                lines.append(child.lineno)
            self.generic_visit(child)

    visitor = DirectBodyVisitor()
    for statement in node.body:
        visitor.visit(statement)
    return sorted(lines)


def check_gate_before(
    relative: str,
    function_name: str,
    paid_calls: tuple[str, ...],
    *,
    gate_calls: tuple[str, ...] = ("_background_budget_gate", "_enforce_background_budget"),
) -> None:
    node = function_node(relative, function_name)
    guards = sorted(
        line
        for gate_name in gate_calls
        for line in call_lines(node, gate_name)
    )
    check(f"{function_name} has a shared budget gate", bool(guards))
    for paid_call in paid_calls:
        paid_lines = call_lines(node, paid_call)
        check(f"{function_name} reaches {paid_call} only after the gate", bool(paid_lines) and min(guards) < min(paid_lines))


def assert_provider_reporting_contract() -> None:
    observability = source("app/observability.py")
    accounting = source("app/runtime/provider_accounting.py")

    check("formal Usage inconsistency writer field is present", '"provider_usage_inconsistency_reasons"' in accounting)
    check("legacy adjective-form field is not written", '"provider_usage_inconsistent_reasons"' not in accounting)
    formal_index = observability.index('usage.get("provider_usage_inconsistency_reasons")')
    legacy_index = observability.index('usage.get("provider_usage_inconsistent_reasons")')
    check("reader prefers formal field before legacy compatibility", formal_index < legacy_index)

    check("Provider aggregate SQL rejects unassignable timestamps", "AND yiai_time_epoch({alias}.created_at) IS NOT NULL" in observability)
    check("invalid timestamp has an explicit exclusion code", "invalid_timestamp_range_unassignable" in observability)
    check("trends never emits unknown timestamp as a normal bucket", 'if period == "unknown"' in observability and "continue" in observability[observability.index('if period == "unknown"'):observability.index('if period == "unknown"') + 220])

    check("provider actual core completeness helper exists", "def _provider_actual_usage_completeness" in observability)
    for field in (
        "cache_hit_input_tokens",
        "cache_miss_input_tokens",
        "output_tokens",
        "total_tokens",
    ):
        check(f"core Usage completeness includes {field}", field in observability)
    check("incomplete actual Usage has its own anomaly code", "provider_actual_usage_incomplete" in observability)
    check("partial actual cost has a dedicated field", "known_partial_provider_actual_cost_cny" in observability)
    check("complete direct cost requires every actual request priced", '== result["provider_actual_calls"]' in observability)
    check("price and calculated-cost failures are separate", "provider_actual_cost_unavailable_count" in observability)
    check("non-finite or negative cost inputs are rejected", "math.isfinite" in observability and "parsed < 0" in observability)
    check("non-integer Provider Token evidence is rejected safely", "_not_integer" in observability and 'not re.fullmatch(' in observability)
    check("missing Provider request identity has an explicit anomaly", "provider_request_id_unavailable_count" in observability and "provider_request_id_unavailable" in observability)
    check("duplicate Provider request identity is rejected", "duplicate_provider_request_id_count" in observability and "duplicate_provider_request_id" in observability)
    check("Provider request identity uniqueness uses global ledger context", "def _annotate_global_provider_identity_counts" in observability and "provider_request_id_global_occurrences" in observability)
    check("conflicting Provider request identity is rejected", "provider_request_identity_conflict_count" in observability and "provider_request_identity_conflict" in observability)
    check("implicit distinct Provider request identities are compared", "def _provider_request_id_candidates" in observability and "len(provider_request_id_candidates) > 1" in observability)
    check("confirmed Provider metadata contradictions stay visible", "confirmed_provider_attempt_metadata_conflict_count" in observability)
    check("misclassified confirmed Provider evidence stays visible", "confirmed_provider_evidence_misclassified_count" in observability)
    check("missing actual model cannot fall back silently", "provider_actual_model_unverified_count" in observability and 'return "actual_model_unverified"' in observability)
    check("conflicting actual models cannot look verified", "provider_actual_model_conflict_count" in observability and 'return "actual_model_conflict"' in observability)
    check("Trace response sanitizes non-finite historical evidence", "def _json_safe_evidence" in observability and "invalid_non_finite_number:" in observability)
    check("malformed inconsistency reason collections are normalized", "provider_usage_inconsistency_reason_malformed_type" in observability)
    check("malformed cost contracts cannot raise Trace detail errors", "cost_contract_malformed_type" in observability and "isinstance(raw_contract, dict)" in observability)
    check("grouping labels reject unhashable historical values", "def _safe_text_evidence" in observability)
    trace_detail_node = function_node("app/observability.py", "trace_detail")
    check("Trace detail applies JSON-safe evidence conversion", bool(call_lines(trace_detail_node, "_json_safe_evidence")))
    check("Trace detail inherits global Provider identity annotations", bool(call_lines(trace_detail_node, "_fetch_reporting_model_calls")))
    recommendation_node = function_node("app/observability.py", "_single_trace_recommendation")
    check("Trace recommendation validates Token numerics before arithmetic", len(call_lines(recommendation_node, "_optional_int")) >= 3)
    check("group cost completeness inherits data quality", "def _apply_cost_group_quality" in observability)
    top_node = function_node("app/observability.py", "_top_provider_actual_traces")
    check("high-cost Trace list requires normal data quality", bool(call_lines(top_node, "_data_quality_summary")))
    check("distribution and trends Token completeness inherit quality", observability.count('and group_quality["data_quality_status"] == "normal"') >= 1 and observability.count('and period_quality["data_quality_status"] == "normal"') >= 1)
    check("budget checks reporting data quality", 'data_quality_status"] != "normal"' in observability)
    check("budget checks cost completeness", 'get("cost_complete") is not True' in observability)


def assert_paid_entry_gate_order() -> None:
    badcases = "app/badcases.py"
    check_gate_before(badcases, "classify_badcase", ("_llm_generate",))
    check_gate_before(badcases, "darwin_fix", ("_llm_generate",))
    check_gate_before(badcases, "extract_knowledge", ("_llm_generate",))
    check_gate_before(
        badcases,
        "switch_model_retry",
        ("build_model", "_llm_generate"),
    )
    check_gate_before(badcases, "retest_badcase", ("_consume_chat_stream",))
    check_gate_before(badcases, "check_tools_badcase", ("_llm_generate",))

    alias_node = function_node(badcases, "switch_model_retry_alias")
    check("retry alias delegates to the guarded canonical route", bool(call_lines(alias_node, "switch_model_retry")))

    check_gate_before("app/evaluations.py", "run_case", ("_run_real_chat",))
    check_gate_before("app/model_configs.py", "ab_test_models", ("build_model",))
    agent_factory_node = function_node(
        "app/runtime/agent_factory.py", "build_runtime_agent"
    )
    check(
        "AgentFactory construction is budget independent for read-only routes",
        not direct_call_lines(agent_factory_node, "_background_budget_gate"),
    )
    workflow_factory_node = function_node(
        "app/runtime/workflow_factory.py", "build_runtime_workflow"
    )
    check(
        "WorkflowFactory construction is budget independent for read-only routes",
        not direct_call_lines(
            workflow_factory_node, "_enforce_agentos_workflow_budget"
        ),
    )
    workflow_source = source("app/runtime/workflow_factory.py")
    check(
        "WorkflowFactory accepts empty input for bare run-history GETs",
        'message: str = ""' in workflow_source,
    )
    check(
        "snapshot resolution is deferred to an executing Workflow step",
        workflow_source.index("def resolve_step")
        < workflow_source.index("snapshot = resolve_snapshot(session_id)"),
    )

    for execution_function in (
        "execute_consultation",
        "collect_action",
        "commit_confirmed_action",
    ):
        check_gate_before(
            "app/runtime/workflow_factory.py",
            execution_function,
            ("_consume_coordinator",),
            gate_calls=("_enforce_agentos_workflow_budget",),
        )

    publish_extension_node = function_node(
        "app/runtime/workflow_factory.py", "publish_extension"
    )
    check(
        "non-model extension acceptance contains no budget gate",
        not call_lines(
            publish_extension_node, "_enforce_agentos_workflow_budget"
        ),
    )
    check("ordinary owner chat contains no background budget gate", "_background_budget_gate" not in source("app/chat.py") and "_enforce_background_budget" not in source("app/chat.py"))

    main = source("app/main.py")
    check(
        "unused AgentOS direct model middleware exists",
        "_disabled_agentos_direct_surface_middleware" in main,
    )
    check(
        "unused AgentOS direct starts continue and resume are stable 410",
        "agentos_direct_agent_run" in main
        and "agentos_direct_workflow_run" in main
        and 'segments[-1] in {"continue", "resume"}' in main
        and "status_code=410" in main,
    )
    check(
        "AgentOS middleware no longer reads or parses request bodies",
        "await request.body()" not in main
        and "parse_qs" not in main
        and "raw_factory_inputs" not in main,
    )
    check(
        "eval and memory model routes return 410 without a budget query",
        'segments == ["eval-runs"]' in main
        and 'segments == ["optimize-memories"]' in main
        and "_background_budget_gate" not in main,
    )
    check(
        "unused AgentOS model routes expose a stable disabled contract",
        "agentos_builtin_model_surface_disabled" in main
        and "status_code=410" in main,
    )

    frontend = source("frontend/index.html")
    check("frontend surfaces unavailable budget ledger", "预算账本不可核实" in frontend and "budgetUnavailable" in frontend)
    check(
        "Trace detail retains excluded and unresolved Provider attempts",
        "const providerAttempts = Array.isArray(data.provider_model_calls)" in frontend
        and "providerAttempts.map(c => renderProviderAttemptCard" in frontend
        and "unresolvedProviderAttempts" in frontend
        and "这不等于确认请求已发出" in frontend,
    )
    check(
        "Provider attempt Reasoning relation is data-derived",
        "reasoningValue <= outputValue" in frontend
        and "Output证据不足，子集关系未确认" in frontend
        and "Provider Usage关系异常" in frontend,
    )
    check(
        "Trace detail trusts explicit complete-Usage evidence and preserves raw fields",
        "call.provider_actual_usage_complete === true" in frontend
        and "核心字段不完整或异常" in frontend
        and "formatUsageField" in frontend,
    )
    check(
        "misclassified Provider evidence remains visible in logical history",
        "provider_reconciliation_issue_codes" in frontend
        and "Provider对账异常" in frontend
        and "reconciliation.reason" in frontend,
    )


def assert_readonly_verifier_contract() -> None:
    verifier = source("scripts/verify_v182_s10f_history_readonly.py")
    core = source("scripts/test_v182_s10f_trace_truthfulness.py")
    check(
        "history verifier never imports the S10-F test harness",
        "test_v182_s10f_trace_truthfulness" not in verifier,
    )
    check(
        "history verifier forces PROPERTY_DATA_DIR before application import",
        'os.environ["PROPERTY_DATA_DIR"] =' in verifier
        and verifier.index('os.environ["PROPERTY_DATA_DIR"] =')
        < verifier.index(') = _import_observability_without_db_init()'),
    )
    check(
        "history verifier uses only SQLite URI mode=ro",
        'database.as_uri()}?mode=ro' in verifier and "uri=True" in verifier,
    )
    check(
        "history verifier blocks init_db and fingerprints target artifacts",
        "init_db_forbidden_in_readonly_verifier" in verifier
        and "database_artifact_fingerprint" in verifier
        and "verified_database_artifacts_changed" in verifier,
    )
    for protected in (
        "/app/data",
        "/volume3/docker/agno-demo-os",
        "unsafe_test_data_dir:verified_database_parent",
    ):
        check(f"history verifier protects {protected}", protected in verifier)
    check(
        "S10-F test forces a fresh temporary data directory",
        'os.environ["PROPERTY_DATA_DIR"] = str(_TEST_DATA_DIR)' in core
        and 'os.environ.setdefault("PROPERTY_DATA_DIR"' not in core,
    )
    check(
        "full dependency mode rejects fallback and deferred checks",
        "S10F_REQUIRE_FULL_DEPENDENCIES" in core
        and "DEFERRED_CHECKS=0" in core,
    )


def assert_fixed_history_release_contract() -> None:
    release = source("docs/releases/v1.8.2-s10-e-trace-time-range-reconciliation.md")
    for value in (
        "4,096",
        "12,373",
        "6,714",
        "5,009",
        "23,183",
        "0.02588292",
    ):
        check(f"accepted fixed-history value {value} remains documented", value in release)
    check("fixed history still documents fifteen Provider requests", "15" in release and "Provider" in release)
    check("fixed history still distinguishes seven Trace groups", "7" in release and "Trace" in release)
    check("RuntimeRelease remains outside this reporting-only change", "RuntimeRelease" in release and "v27" in release)


def main() -> None:
    assert_provider_reporting_contract()
    assert_paid_entry_gate_order()
    assert_readonly_verifier_contract()
    assert_fixed_history_release_contract()
    print(f"PASS: S10-G source contracts ({len(CHECKS)} checks; behavior not inferred)")


if __name__ == "__main__":
    main()
