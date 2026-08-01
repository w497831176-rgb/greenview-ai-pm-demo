"""Deterministic V1.8.2-S9-B/S9-B.1 cost-story and frontend hierarchy checks.

No HTTP, model, RuntimeRelease, or business-data call is made.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from app.observability import (
    _aggregate_model_calls,
    _overview_scope,
    _top_provider_actual_traces,
    _trace_cost_explanation,
)


def actual_call(
    trace_id: str,
    stage: str,
    model: str,
    hit: int,
    miss: int,
    output: int,
    amount: float,
) -> dict:
    local_attempt_id = f"fixture-{trace_id}-{stage}"
    provider_request_id = f"req-{trace_id}-{stage}"
    return {
        "trace_id": trace_id,
        "stage": stage,
        "model_id": model,
        "requested_model": model,
        "provider_response_model": model,
        "thinking_enabled": True,
        "usage_source": "provider_actual",
        "usage_status": "provider_actual",
        "record_kind": "provider_attempt",
        "local_attempt_id": local_attempt_id,
        "provider_request_id": provider_request_id,
        "cost_source": "platform_price_snapshot",
        "usage_normalized": {
            "record_kind": "provider_attempt",
            "include_in_provider_aggregate": True,
            "provider_request_sent": True,
            "local_attempt_id": local_attempt_id,
            "provider_request_id": provider_request_id,
            "requested_model": model,
            "provider_response_model": model,
            "thinking_enabled": True,
            "usage_source": "provider_actual",
            "usage_status": "provider_actual",
            "cost_source": "platform_price_snapshot",
            "input_cache_hit_tokens": hit,
            "input_cache_miss_tokens": miss,
            "output_tokens": output,
        },
        "total_tokens": hit + miss + output,
        "estimated_cost_cny": amount,
        "calculated_direct_cost": amount,
        "status": "success",
        "price_snapshot": {"model_id": model, "currency": "CNY"},
        "cost_formula": "hit*cached_price + miss*input_price + output*output_price",
    }


def main() -> None:
    # 1. Default range is Beijing today 00:00 through current time.
    scope = _overview_scope(None, None, datetime(2026, 7, 28, 12, 34, 56))
    assert scope == {
        "label": "今日",
        "start": "2026-07-28 00:00:00",
        "end": "2026-07-28 12:34:56",
        "timezone": "Asia/Shanghai (UTC+8)",
    }

    flash_calls = [
        actual_call(
            "3c1aad38350c4600",
            "router",
            "deepseek-v4-flash",
            384,
            1845,
            191,
            0.00223468,
        ),
        actual_call(
            "3c1aad38350c4600",
            "vertical_agent",
            "deepseek-v4-flash",
            2176,
            404,
            164,
            0.00077552,
        ),
    ]
    pro_call = actual_call(
        "7a9ef83684304e95",
        "darwin",
        "deepseek-v4-pro",
        2688,
        24,
        1958,
        0.01188720,
    )
    estimated = {
        **actual_call("estimated-trace", "router", "deepseek-v4-flash", 0, 0, 0, 0),
        "record_kind": "legacy",
        "usage_source": "estimated",
        "usage_status": "estimated",
        "cost_source": None,
        "usage_normalized": {
            "record_kind": "legacy",
            "include_in_provider_aggregate": False,
            "provider_request_sent": None,
            "usage_source": "estimated",
            "usage_status": "estimated",
        },
        "total_tokens": 80,
        "estimated_cost_cny": 0.0001,
        "calculated_direct_cost": None,
        "provider_response_model": None,
    }
    unavailable = {
        **actual_call("failed-trace", "router", "deepseek-v4-flash", 0, 0, 0, 0),
        "usage_source": "unavailable",
        "usage_status": "unavailable_provider_error",
        "cost_source": None,
        "usage_normalized": {
            **actual_call("failed-trace", "router", "deepseek-v4-flash", 0, 0, 0, 0)["usage_normalized"],
            "usage_source": "unavailable",
            "usage_status": "unavailable_provider_error",
            "usage_unavailable_reason": "provider_http_error",
            "cost_source": None,
        },
        "total_tokens": 99,
        "estimated_cost_cny": None,
        "calculated_direct_cost": None,
        "provider_response_model": None,
        "status": "failed",
    }

    # 2. provider_actual, estimated, and unavailable are strict separate buckets.
    aggregate = _aggregate_model_calls(flash_calls + [estimated, unavailable])
    assert aggregate["calls"] == 3
    assert aggregate["provider_actual_calls"] == 2
    assert aggregate["provider_actual_cost_cny"] == 0.00301020
    assert aggregate["estimated_calls"] == 0
    assert aggregate["estimated_cost_cny"] == 0
    assert aggregate["unavailable_calls"] == 1
    assert aggregate["excluded_record_count"] == 1
    assert aggregate["input_cache_hit_tokens"] == 2560
    assert aggregate["input_cache_miss_tokens"] == 2249
    assert aggregate["output_tokens"] == 355

    # 3. A zero-model flow stays out of every model-cost bucket.
    no_model_aggregate = _aggregate_model_calls([])
    assert no_model_aggregate["calls"] == 0
    assert no_model_aggregate["unavailable_calls"] == 0
    assert no_model_aggregate["provider_actual_cost_cny"] == 0

    # 4. High-cost ranking includes only fully provider_actual traces.
    ranking = _top_provider_actual_traces(
        flash_calls + [pro_call, estimated, unavailable], limit=5
    )
    assert [item["trace_id"] for item in ranking] == [
        "7a9ef83684304e95",
        "3c1aad38350c4600",
    ]

    # 5. Flash story adds stages exactly and proposes one evidence-based action.
    flash_story = _trace_cost_explanation(flash_calls)
    assert flash_story["summary"] == (
        "本轮产生2次Provider请求（2次Flash）；已取得Provider实际Usage合计5,164 Token；"
        "平台价格快照直接成本¥0.00301020（非DeepSeek最终账单）。"
    )
    assert sum(item["amount_cny"] for item in flash_story["chain"]) == 0.00301020
    assert [item["stage_name"] for item in flash_story["chain"]] == [
        "Router",
        "垂直Agent",
    ]
    assert flash_story["recommendation"]["code"] == "cache_miss_high"

    # 6. Darwin remains Pro even with Thinking enabled and uses output advice.
    pro_story = _trace_cost_explanation([pro_call])
    assert pro_story["summary"] == (
        "本轮产生1次Provider请求（1次Pro）；已取得Provider实际Usage合计4,670 Token；"
        "平台价格快照直接成本¥0.01188720（非DeepSeek最终账单）。"
    )
    assert pro_story["chain"][0]["provider_response_model"] == "deepseek-v4-pro"
    assert pro_story["chain"][0]["thinking_enabled"] is True
    assert pro_story["recommendation"]["code"] == "output_high"

    # 7. No-model and insufficient-evidence wording never invents a zero cost.
    no_model_story = _trace_cost_explanation([])
    assert no_model_story["summary"] == "本轮未调用模型，因此模型Token与费用不适用。"
    assert no_model_story["total_tokens"] is None
    assert no_model_story["cost_scope"]["status"] == "not_applicable"
    failed_story = _trace_cost_explanation([unavailable])
    assert "1次Provider实际Usage不可得" in failed_story["summary"]
    assert "¥0" not in failed_story["summary"]
    assert failed_story["recommendation"]["code"] == "evidence_insufficient"

    # 8. UI has one navigation entry, three default regions, and folded engineering detail.
    source = Path("frontend/index.html").read_text(encoding="utf-8")
    platform_menu = source[source.index("platform: [") : source.index("const ICONS")]
    cost_section = source[source.index("async function renderCostGovernancePage") : source.index("async function renderCostStrategyPage")]
    render_page = cost_section[cost_section.rindex("function renderPage()") : cost_section.index("function bindEvents()")]
    assert platform_menu.count("label: 'Trace 与成本'") == 1
    assert "label: '成本优化策略'" not in platform_menu
    assert "/api/model-configs/ab-test" not in cost_section
    assert "traceParams.set('limit', String(state.pagination.limit))" in cost_section
    assert "traceParams.set('offset', String((state.pagination.page - 1) * state.pagination.limit))" in cost_section
    assert all(label in render_page for label in (
        "renderOverview()",
        "renderTraces()",
        "renderGovernancePrinciples()",
        "高级信息：价格、预算与策略说明",
    ))
    assert all(label not in render_page for label in (
        "renderHighCostTraces()",
        "renderModelChart()",
        "renderStageChart()",
        "renderCostCases()",
        "cg-open-strategy-page",
    ))
    assert all(label in cost_section for label in (
        "一句话结论",
        "['today', '今天']",
        "['yesterday', '昨天']",
        "['last_7_days', '近7天']",
        "['this_month', '本月']",
        "['last_month', '上月']",
        "['custom', '自定义']",
        "需要关注",
        "调用记录",
        "每页20条",
        "高级筛选",
        "我们的成本治理原则",
        "查看逐请求对账证据",
        "1. 本次发生了什么",
        "2. 为什么这样选择",
        "3. 平台直接成本怎么来的",
        "4. 下一步建议",
        "高级证据：逐请求对账、逻辑聚合与历史记录",
        "本轮没有纳入汇总的Provider请求，因此Provider Token与平台价格快照直接成本不适用",
        "系统没有按0或估算冒充actual",
    ))
    assert "$('#cg-run-cost01').addEventListener" not in cost_section
    assert "$('#cg-run-cost02').addEventListener" not in cost_section

    print("PASS: V1.8.2-S9-B/S9-B.1 deterministic cost-story contract (8 groups)")


if __name__ == "__main__":
    main()
