"""Deterministic S10-C two-layer quality and Provider-evidence contract.

No browser, HTTP, model, RuntimeRelease, or database call is made.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.runtime.provider_evidence import (
    capture_provider_response,
    provider_evidence_from_run,
    remember_provider_request,
)


FRONTEND = ROOT / "frontend" / "index.html"


def check_provider_usage_chunk_merge() -> None:
    journal: list[dict] = []
    request_id = "fixture-request-id"

    content_parsed = SimpleNamespace(provider_data={})
    content_chunk = SimpleNamespace(
        id=request_id,
        model="deepseek-fixture-model",
        usage=None,
    )
    capture_provider_response(content_parsed, content_chunk)
    remember_provider_request(journal, provider_evidence_from_run(content_parsed))

    usage_parsed = SimpleNamespace(provider_data={})
    usage_chunk = SimpleNamespace(
        id=request_id,
        model="deepseek-fixture-model",
        usage=SimpleNamespace(
            prompt_cache_hit_tokens=120,
            prompt_cache_miss_tokens=30,
            prompt_tokens=150,
            completion_tokens=20,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=7),
            total_tokens=170,
        ),
    )
    capture_provider_response(usage_parsed, usage_chunk)
    usage_evidence = provider_evidence_from_run(usage_parsed)
    assert usage_evidence["provider_request_id"] == request_id
    remember_provider_request(journal, usage_evidence)

    assert len(journal) == 1, "one streamed request must not split into two rows"
    captured = journal[0]
    assert captured["provider_request_id"] == request_id
    assert captured["provider_response_model"] == "deepseek-fixture-model"
    assert captured["usage"] == {
        "input_cache_hit_tokens": 120,
        "input_cache_miss_tokens": 30,
        "input_tokens": 150,
        "output_tokens": 20,
        "reasoning_tokens": 7,
        "total_tokens": 170,
    }


def check_deepseek_adapter_contract() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "agno==2.6.21" in requirements

    settings_source = (ROOT / "app" / "settings.py").read_text(encoding="utf-8")
    assert "class EvidenceDeepSeek(DeepSeek)" in settings_source
    assert "def _parse_provider_response(" in settings_source
    assert "def _parse_provider_response_delta(" in settings_source
    assert "return EvidenceDeepSeek(" in settings_source

    direct_calls: list[str] = []
    for directory in (ROOT / "app", ROOT / "agents"):
        for path in directory.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            if "chat.completions.create" in source:
                direct_calls.append(str(path.relative_to(ROOT)))
            if "DeepSeek(" in source and path != ROOT / "app" / "settings.py":
                direct_calls.append(str(path.relative_to(ROOT)))
    assert not direct_calls, f"DeepSeek bypasses build_model: {direct_calls}"


def check_two_layer_frontend_contract() -> None:
    html = FRONTEND.read_text(encoding="utf-8")
    checks = {
        "neutral Trace action": ">查看 Trace</button>" in html and "查看Trace证据 ·" not in html,
        "manual Evaluation entry": 'id="evaluation-create"' in html and "人工创建用例" in html,
        "minimal Evaluation fields": all(value in html for value in ("用例名称 *", "用户会怎么问 *", "预期结果至少包含 *")),
        "Evaluation path": all(value in html for value in ("人工创建用例 → 写清预期", "能力节点命中", "PASS / FAIL")),
        "Evaluation advanced evidence": "高级证据：逐节点判定、Trace、Release 与用量" in html,
        "manual Badcase entry": 'id="badcase-create"' in html and "人工创建 Badcase" in html,
        "Badcase four steps": all(value in html for value in ("发现问题", "AI 根因建议", "人工确认修复方案", "单例复测与人工关闭")),
        "Badcase advanced evidence": "高级证据：草稿、Trace、Release、历次复测与审计" in html,
        "AI cannot self-confirm": "建议不等于人工确认根因" in html and "系统不会自动关闭" in html,
        "current RuntimeRelease primary": "当前生效版本 ·" in html and "唯一当前生效" in html,
        "RuntimeRelease counts": all(value in html for value in (">Agent</div>", ">Skill</div>", ">RAG 文档</div>", ">只读 MCP</div>")),
        "history and diff folded": 'id="runtime-history-panel"' in html and "历史版本与 Diff" in html,
    }
    failed = [name for name, passed in checks.items() if not passed]
    assert not failed, f"frontend contract failures: {failed}"


def main() -> None:
    check_provider_usage_chunk_merge()
    check_deepseek_adapter_contract()
    check_two_layer_frontend_contract()
    print("PASS: V1.8.2-S10-C two-layer quality surface and Provider evidence")


if __name__ == "__main__":
    main()
