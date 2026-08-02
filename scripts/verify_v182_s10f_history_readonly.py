"""Read-only fixed-history verifier for the S10-F reporting contract.

The supplied SQLite ledger is opened only through a ``mode=ro`` URI.  Before
any YIAI application/database module is imported, this process forces a fresh
temporary ``PROPERTY_DATA_DIR`` that is disjoint from the verified database
and every known production data location.  The verifier never initializes or
migrates a database and prints aggregate evidence only.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_AUG1 = {
    "trace_group_count": 7,
    "provider_request_count": 15,
    "input_cache_hit_tokens": 4096,
    "input_cache_miss_tokens": 12373,
    "output_tokens": 6714,
    "reasoning_tokens": 5009,
    "total_tokens": 23183,
    "platform_price_snapshot_direct_cost_cny": 0.02588292,
    "statistics_status": "consistent",
    "data_quality_status": "normal",
}

_PRODUCTION_PATH_MARKERS = (
    "/app/data",
    "/volume3/docker/agno-demo-os",
)


def _normalized_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").rstrip("/").lower()


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def assert_safe_test_data_dir(test_data_dir: Path, database: Path) -> None:
    """Reject any test initialization path that could touch verified data."""
    candidate = test_data_dir.resolve()
    database_parent = database.resolve().parent
    normalized = _normalized_path(candidate)
    if any(
        normalized == marker
        or normalized.startswith(f"{marker}/")
        or marker.startswith(f"{normalized}/")
        for marker in _PRODUCTION_PATH_MARKERS
    ):
        raise RuntimeError("unsafe_test_data_dir:production_path")
    if _paths_overlap(candidate, database_parent):
        raise RuntimeError("unsafe_test_data_dir:verified_database_parent")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def database_artifact_fingerprint(database: Path) -> dict[str, dict[str, Any]]:
    """Fingerprint the ledger and SQLite sidecars without opening them."""
    result: dict[str, dict[str, Any]] = {}
    for suffix in ("", "-wal", "-shm", "-journal"):
        path = Path(f"{database}{suffix}")
        if not path.exists():
            continue
        stat = path.stat()
        result[path.name] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _sha256(path),
        }
    return result


def _import_observability_without_db_init() -> tuple[Any, Any, Any, dict[str, int]]:
    """Import reporting code while making any database init a hard failure."""
    already_imported = [
        name
        for name in sys.modules
        if name in {"app", "db"}
        or name.startswith("app.")
        or name.startswith("db.")
    ]
    if already_imported:
        raise RuntimeError(
            "unsafe_import_order:" + ",".join(sorted(already_imported))
        )

    property_db = importlib.import_module("db.property_db")
    original_init_db = property_db.init_db
    init_db_state = {"calls": 0}

    def forbidden_init_db(*_args: Any, **_kwargs: Any) -> None:
        init_db_state["calls"] += 1
        raise RuntimeError("init_db_forbidden_in_readonly_verifier")

    property_db.init_db = forbidden_init_db
    observability = importlib.import_module("app.observability")
    if init_db_state["calls"]:
        raise RuntimeError("init_db_called_in_readonly_verifier")
    return observability, property_db, original_init_db, init_db_state


def _read_only_connection(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _run_verification(database: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="yiai-s10f-readonly-") as temp_dir:
        test_data_dir = Path(temp_dir).resolve()
        assert_safe_test_data_dir(test_data_dir, database)

        # Assignment is intentional: inherited production values are never
        # accepted, even when the caller set them before starting the process.
        os.environ["PROPERTY_DATA_DIR"] = str(test_data_dir)
        os.environ["DEEPSEEK_API_KEY"] = ""

        before = database_artifact_fingerprint(database)
        (
            observability,
            property_db,
            original_init_db,
            init_db_state,
        ) = _import_observability_without_db_init()
        if Path(property_db.DB_PATH).resolve().parent != test_data_dir:
            raise RuntimeError("PROPERTY_DATA_DIR_not_forced_before_db_import")
        original_get_conn = observability._get_conn
        original_thresholds = observability.get_budget_thresholds
        original_evaluation = observability.evaluation_summary
        original_trace = observability.get_chat_trace

        observability._get_conn = lambda: _read_only_connection(database)
        observability.get_budget_thresholds = lambda: {}
        observability.evaluation_summary = lambda: {}
        observability.get_chat_trace = lambda _trace_id: None
        try:
            aug1 = asyncio.run(
                observability.overview(
                    start="2026-08-01",
                    end="2026-08-01",
                    model_id=None,
                    stage=None,
                    trace_id=None,
                    range_key="custom",
                )
            )
            aug2 = asyncio.run(
                observability.overview(
                    start="2026-08-02",
                    end="2026-08-02",
                    model_id=None,
                    stage=None,
                    trace_id=None,
                    range_key="custom",
                )
            )
        finally:
            observability._get_conn = original_get_conn
            observability.get_budget_thresholds = original_thresholds
            observability.evaluation_summary = original_evaluation
            observability.get_chat_trace = original_trace
            property_db.init_db = original_init_db

        after = database_artifact_fingerprint(database)
        if before != after:
            raise RuntimeError("verified_database_artifacts_changed")
        if Path(property_db.DB_PATH).exists():
            raise RuntimeError("test_database_was_created")

    actual_aug1 = {
        "trace_group_count": aug1["trace_group_count"],
        "provider_request_count": aug1["provider_request_count"],
        "input_cache_hit_tokens": aug1["known_usage"][
            "input_cache_hit_tokens"
        ],
        "input_cache_miss_tokens": aug1["known_usage"][
            "input_cache_miss_tokens"
        ],
        "output_tokens": aug1["known_usage"]["output_tokens"],
        "reasoning_tokens": aug1["known_usage"]["reasoning_tokens"],
        "total_tokens": aug1["total_tokens"],
        "platform_price_snapshot_direct_cost_cny": aug1[
            "platform_price_snapshot_direct_cost_cny"
        ],
        "statistics_status": aug1["statistics_status"],
        "data_quality_status": aug1["data_quality_status"],
    }
    anomaly_counts = {
        key: aug1["data_quality"][key]
        for key in (
            "invalid_timestamp_count",
            "provider_send_unconfirmed_count",
            "orphaned_pending_count",
            "provider_request_id_unavailable_count",
            "provider_request_identity_conflict_count",
            "duplicate_provider_request_id_count",
            "confirmed_provider_attempt_metadata_conflict_count",
            "confirmed_provider_evidence_misclassified_count",
            "provider_actual_model_unverified_count",
            "provider_actual_model_conflict_count",
            "provider_usage_inconsistent_count",
            "provider_actual_usage_incomplete_count",
            "provider_actual_price_missing_count",
            "provider_actual_cost_unavailable_count",
            "unavailable_usage_count",
            "unresolved_reconciliation_count",
        )
    }
    return {
        "aug1": actual_aug1,
        "aug1_anomaly_counts": anomaly_counts,
        "aug2_provider_request_count": aug2["provider_request_count"],
        "readonly_guard": {
            "init_db_calls": init_db_state["calls"],
            "database_artifacts_unchanged": before == after,
            "connection_mode": "ro",
            "property_data_dir_forced": True,
            "test_database_created": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    args = parser.parse_args()
    database = args.database.resolve()
    if not database.is_file():
        raise SystemExit("database file does not exist")

    result = _run_verification(database)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if result["aug1"] != EXPECTED_AUG1:
        raise SystemExit("2026-08-01 fixed history mismatch")
    if any(result["aug1_anomaly_counts"].values()):
        raise SystemExit("2026-08-01 anomaly bucket is not zero")
    if result["aug2_provider_request_count"] != 0:
        raise SystemExit("2026-08-02 Provider request count changed")
    if result["readonly_guard"] != {
        "init_db_calls": 0,
        "database_artifacts_unchanged": True,
        "connection_mode": "ro",
        "property_data_dir_forced": True,
        "test_database_created": False,
    }:
        raise SystemExit("read-only verification guard failed")
    print("PASS: S10-F read-only fixed-history verification")


if __name__ == "__main__":
    main()
