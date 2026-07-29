"""Deterministic contract for the V1.8.2-S10-B.1 demo surface.

Static files only: no browser, API, model, RuntimeRelease, or data mutation.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend" / "index.html"
GUIDE = ROOT / "docs" / "interview" / "S10-B-DEMO-SURFACE-GUIDE.md"


def between(text: str, start: str, end: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[start_index:end_index]


def main() -> None:
    html = FRONTEND.read_text(encoding="utf-8")
    guide = GUIDE.read_text(encoding="utf-8")
    passed: list[str] = []

    def check(name: str, condition: bool) -> None:
        assert condition, name
        passed.append(name)

    # The two business-role menus and routes remain unchanged.
    owner_menu = """owner: [
        { id: 'chat', label: 'AI 助手', icon: 'chat' },
        { id: 'my-orders', label: '我的工单', icon: 'orders' }
      ]"""
    staff_menu = """staff: [
        { id: 'work-orders', label: '工单管理', icon: 'orders' },
        { id: 'handoff-orders', label: '人工接管', icon: 'handoff' }
      ]"""
    check("owner first-level menu unchanged", owner_menu in html)
    check("staff first-level menu unchanged", staff_menu in html)
    for route in ("chat", "my-orders", "work-orders", "handoff-orders"):
        check(f"business route remains reachable: {route}", f"case '{route}':" in html)

    # Legacy platform routes stay implemented even when their entry is folded.
    for route in (
        "agents", "models", "model-ab", "skills", "mcp", "knowledge",
        "runtime", "badcases", "badcase-detail", "evaluations",
        "cost-governance", "cost-strategy",
    ):
        check(f"platform route remains implemented: {route}", f"case '{route}':" in html)

    # Skill status is default-readable; the mutating switch is inside advanced management.
    skills = between(html, "async function renderSkillsPage", "async function editSkill")
    advanced_skill = skills.index("高级管理：修改、预览或删除")
    toggle = skills.index('class="skill-toggle')
    check("Skill status label is visible", "Draft中启用" in skills and "Draft中停用" in skills)
    check("Skill toggle is inside advanced management", toggle > advanced_skill)
    for phrase in ("会修改Skill Draft", "不调用模型", "不自动发布RuntimeRelease", "仅新会话生效", "旧会话保持原Snapshot"):
        check(f"Skill toggle consequence explained: {phrase}", phrase in skills)

    # RAG parameters are reachable only from the advanced management block.
    knowledge = between(html, "async function renderKnowledgePage", "async function renderMcpPage")
    advanced_rag = knowledge.index("高级管理：新增文档、检索设置与调试")
    settings_entry = knowledge.index('data-tab="settings"')
    content_slot = knowledge.index('id="knowledge-content"')
    check("RAG settings removed from default tabs", settings_entry > content_slot and settings_entry > advanced_rag)
    check("RAG Draft save action is explicit", "保存RAG检索Draft" in knowledge)
    for phrase in ("保存会修改Draft", "不调用模型", "不自动发布", "必须发布RuntimeRelease后才影响新会话"):
        check(f"RAG consequence explained: {phrase}", phrase in knowledge)

    # Runtime history keeps comparison visible; mutating or technical operations are folded.
    runtime = between(html, "const renderHistoryRow", "async function openAdvancedAcceptance")
    advanced_version = runtime.index("高级版本操作")
    check("Runtime diff remains the default action", runtime.index("runtime-history-diff") < advanced_version)
    check("Runtime snapshot is folded", runtime.index("runtime-history-config") > advanced_version)
    check("Runtime rollback is folded", runtime.index("runtime-rollback-btn") > advanced_version)
    for phrase in ("修改当前运行版本指针", "不部署Git代码", "仅影响后续新会话", "旧会话保持原Snapshot"):
        check(f"Runtime rollback consequence explained: {phrase}", phrase in html)
    acceptance = between(html, '<details class="rounded-2xl border border-dashed', "requireElement('#runtime-preview-btn'")
    check("Runtime advanced acceptance is folded", "<details" in acceptance and 'id="runtime-advanced-btn"' in acceptance)

    # Model-consuming Badcase actions stay advanced and require an explicit confirmation.
    badcase_surface = between(html, "查看处理草稿与操作入口", "查看技术证据")
    check("Badcase operations stay in folded advanced area", "高级操作" in badcase_surface)
    for phrase in (
        "可能调用模型并产生费用", "调用Pro并产生费用",
        "通常调用Flash并产生费用", "AI分析建议，不等于人工确认根因",
    ):
        check(f"Badcase consequence visible: {phrase}", phrase in html)
    for confirmation in (
        "AI自动分类可能调用Flash并产生费用",
        "Darwin深度分析会调用Pro并产生费用",
        "真实复测会运行真实业务链路",
    ):
        check(f"Badcase confirmation exists: {confirmation}", f"window.confirm('{confirmation}" in html)
    check("No-model Badcase mutations are explained", "不调用模型，但会修改Badcase" in html)

    # Session, handoff, controlled-write, and quick-question wording matches runtime behavior.
    quick = between(html, "window.sendQuickQuestion", "function formatTimeLabel")
    check("Quick question only fills the input", "input.value = text" in quick and "sendChatMessage" not in quick and "dispatchEvent" not in quick)
    check("Owner handoff has visible text", 'id="chat-handoff"' in html and "<span>转人工</span>" in html)
    check("Guide includes new session", "| 业主工作台 | 新建会话 |" in guide)
    check("Guide includes owner handoff", "| 业主工作台 | 转人工 |" in guide)
    check("Guide records first-run Snapshot binding", "首次发送消息时绑定当前Published版本为不可变Snapshot" in guide)
    check("Controlled-write card uses truthful name", "受控写入状态与Receipt回执卡" in guide and "受控写入确认卡" not in guide)
    check("Controlled-write confirmation remains conversational", "确认动作发生在对话中" in guide)

    # Tool names are not mislabeled as MCP Server names; boundaries remain explicit.
    agent_table = between(html, "const isSeededRouter", "async function openAgentModal")
    check("Agent table labels tool values correctly", "已绑定Tool" in agent_table and "<b>MCP Server：</b>${escapeHtml((a.available_mcp_tools" not in agent_table)
    check("MCP Tool ActionGateway boundary remains explicit", "MCP Server负责接入外部能力" in html and "ActionGateway" in html)

    # Trace citations show a safe excerpt when present and a truthful fallback otherwise.
    check("Trace citation displays limited evidence excerpt", "rawExcerpt.slice(0, 160)" in html and "证据摘录" in html)
    check("Trace citation has no-content fallback", "请从回答引用查看证据原文" in html)

    # Runtime version stays dynamic; technical operations remain hidden by default.
    check("RuntimeRelease overview is read dynamically", "loadRuntimeVersionLabel" in html and "/api/runtime/releases/overview" in html)
    check("stale release label removed", "V1.8.1-rc.6" not in html)
    check("v26 is not hard-coded into the frontend", "RuntimeRelease v26" not in html)
    check("Guide does not self-certify complete coverage", "覆盖：100%" not in guide and "覆盖100%" not in guide)

    rows = [line for line in guide.splitlines() if line.startswith("|") and not line.startswith("|---")][1:]
    check("Guide remains near thirty explainable items", 20 <= len(rows) <= 30)

    print({"status": "PASS", "checks": len(passed), "guide_items": len(rows)})


if __name__ == "__main__":
    main()
