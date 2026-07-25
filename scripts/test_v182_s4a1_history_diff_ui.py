from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend" / "index.html"


def _runtime_source() -> str:
    source = FRONTEND.read_text(encoding="utf-8")
    start = source.index("async function renderRuntimePage")
    end = source.index("async function renderAgentsPage", start)
    return source[start:end]


def test_history_button_names_parent_version() -> None:
    source = _runtime_source()
    assert "与 #${escapeHtml(releaseVersionFromId(release.parent_release_id)" in source
    assert "初始版本（无上版）" in source
    assert ">查看变更</button>" not in source


def test_history_diff_uses_wide_dedicated_modal() -> None:
    source = _runtime_source()
    assert "const openReleaseDiffModal = (data, currentRelease)" in source
    assert "max-w-6xl max-h-[94vh]" in source
    handler_start = source.index("const bindHistoryActions")
    handler_end = source.index("async function openAdvancedAcceptance", handler_start)
    handler = source[handler_start:handler_end]
    assert "openReleaseDiffModal(data, currentRelease)" in handler
    assert "openDrawer(`Release #" not in handler


def test_version_relationship_and_release_metadata_are_explicit() -> None:
    source = _runtime_source()
    assert "Release #${parent.version || '-'} → Release #${release.version || '-'}" in source
    assert "release?.release_id" in source
    assert "发布时间：" in source
    assert "当前运行" in source
    assert "当前运行版本：Release #" in source
    assert "变更结论" in source


def test_agent_fields_use_three_column_comparison() -> None:
    source = _runtime_source()
    assert "md:grid-cols-[160px_minmax(0,1fr)_minmax(0,1fr)]" in source
    assert "<div>变更项</div>" in source
    assert "<div>修改前</div>" in source
    assert "<div>修改后</div>" in source
    assert "A. Agent 参数变化" in source
    assert "item.label || item.field || '参数'" in source


def test_prompt_is_collapsed_by_default() -> None:
    source = _runtime_source()
    assert "item.field === 'instructions'" in source
    assert "<details class=\"group rounded-xl border" in source
    assert "展开完整 Prompt" in source
    assert "收起完整 Prompt" in source
    assert "max-h-80 overflow-auto" in source


def test_agent_capabilities_are_split_into_added_and_removed() -> None:
    source = _runtime_source()
    assert "B. 挂载能力变化" in source
    assert "新增能力" in source
    assert "移除能力" in source
    assert "['Skill', agent.capabilities?.skills]" in source
    assert "['RAG 文档', agent.capabilities?.knowledge]" in source
    assert "['MCP', agent.capabilities?.mcp_servers]" in source
    assert "item.name || item.id" in source


def test_other_engineering_changes_are_secondary_and_collapsed() -> None:
    source = _runtime_source()
    marker = "其他配置变化（${count}项）"
    assert marker in source
    details_start = source.index('<details class="rounded-2xl', source.index(marker) - 300)
    details_end = source.index("</details>", details_start)
    assert " open" not in source[details_start:details_end]
    comparison_start = source.index("const renderHistoricalDiffComparison")
    comparison_end = source.index("const openReleaseDiffModal", comparison_start)
    comparison = source[comparison_start:comparison_end]
    assert comparison.index("Agent 变化") < comparison.index(
        "${renderHistoricalOtherChanges(diff)}"
    )


def test_duplicate_and_initial_release_states_are_not_blank() -> None:
    source = _runtime_source()
    assert "该版本配置与上一版本相同，属于历史重复快照。" in source
    assert "这是初始版本，没有上一个版本可供比较。" in source
    assert "不存在" in source
    assert "已删除" in source


def test_history_diff_remains_lazy_loaded() -> None:
    source = _runtime_source()
    history_diff_path = (
        "${encodeURIComponent(button.dataset.releaseId)}/diff?include_details=true"
    )
    assert "apiGet('/api/runtime/releases/overview')" in source
    assert source.count(history_diff_path) == 1
    handler_start = source.index("const bindHistoryActions")
    handler_end = source.index("async function openAdvancedAcceptance", handler_start)
    assert history_diff_path in source[handler_start:handler_end]
    page_start = source.index("async function renderRuntimePage")
    load_overview_start = source.index("async function loadOverview", page_start)
    assert history_diff_path not in source[load_overview_start:]


def test_s4a1_does_not_expand_runtime_scope() -> None:
    source = _runtime_source()
    assert "/api/runtime/releases/overview" in source
    assert "/api/chat/stream" not in source
    assert "/api/runtime/releases/publish-current-config" in source
    assert "RuntimeRelease 发布中心" in source


if __name__ == "__main__":
    tests = [
        test_history_button_names_parent_version,
        test_history_diff_uses_wide_dedicated_modal,
        test_version_relationship_and_release_metadata_are_explicit,
        test_agent_fields_use_three_column_comparison,
        test_prompt_is_collapsed_by_default,
        test_agent_capabilities_are_split_into_added_and_removed,
        test_other_engineering_changes_are_secondary_and_collapsed,
        test_duplicate_and_initial_release_states_are_not_blank,
        test_history_diff_remains_lazy_loaded,
        test_s4a1_does_not_expand_runtime_scope,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS: {len(tests)}/{len(tests)} S4-A.1 history Diff UI contracts")
