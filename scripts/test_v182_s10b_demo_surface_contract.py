"""Deterministic contract for the V1.8.2-S10-B interview demo surface.

This test reads static frontend/document files only. It never opens a browser,
calls an API or model, or changes project data.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend" / "index.html"
GUIDE = ROOT / "docs" / "interview" / "S10-B-DEMO-SURFACE-GUIDE.md"


def main() -> None:
    html = FRONTEND.read_text(encoding="utf-8")
    guide = GUIDE.read_text(encoding="utf-8")
    passed: list[str] = []

    def check(name: str, condition: bool) -> None:
        assert condition, name
        passed.append(name)

    # The two business roles keep their original first-level menus and routes.
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

    # Only platform management is visually grouped; every legacy management route remains reachable.
    for group in ("能力配置", "版本发布", "质量闭环", "调用与成本"):
        check(f"platform group exists: {group}", f"group: '{group}'" in html)
    for route in (
        "agents", "models", "model-ab", "skills", "mcp", "knowledge",
        "runtime", "badcases", "badcase-detail", "evaluations",
        "cost-governance", "cost-strategy",
    ):
        check(f"platform route remains implemented: {route}", f"case '{route}':" in html)
    check("model A/B remains reachable from advanced management", "进入模型A/B测试" in html)
    check("legacy cost strategy remains reachable from advanced information", "查看高级成本策略" in html)

    # Common AI product terms stay visible with plain-Chinese explanations.
    for phrase in (
        "Agent（AI处理角色）", "Skill（业务规则）", "RAG（知识依据）",
        "MCP Server（外部能力接入服务）", "Tool是其中可调用的具体能力",
        "RuntimeRelease（运行版本）", "Snapshot（会话配置快照）",
        "Draft（待发布配置）", "Published（已发布配置）",
        "Badcase（问题案例）", "Evaluation / Golden Set（评估 / 固定评估集）",
        "Trace（运行调用记录）", "Provider Usage（模型服务真实用量）",
        "ActionGateway（受控业务写入网关）",
    ):
        check(f"term retained: {phrase}", phrase in html)

    check(
        "MCP Tool and ActionGateway boundary remains explicit",
        "MCP/Tool不等于ActionGateway" in html
        and "它不是 MCP/Tool 调用" in html,
    )
    check(
        "Trace is not represented as a provider bill",
        "Trace是系统运行证据，不等于供应商账单" in html,
    )
    check(
        "AI analysis is not represented as human-confirmed root cause",
        "AI分析建议不等于人工确认根因" in html,
    )

    # RuntimeRelease is loaded from the real API; stale fixed labels are prohibited.
    check("RuntimeRelease overview is read dynamically", "loadRuntimeVersionLabel" in html and "/api/runtime/releases/overview" in html)
    check("stale release label removed", "V1.8.1-rc.6" not in html)
    check("v26 is not hard-coded into the frontend", "RuntimeRelease v26" not in html)

    # Technical/test/mutating controls are present but folded by default.
    for summary in (
        "高级信息：查看ID、Snapshot、Span与逐次技术证据",
        "高级管理：模型A/B测试",
        "高级管理：新增文档与检索调试",
        "高级信息：价格、预算与策略说明",
        "查看技术证据",
    ):
        check(f"advanced content folded: {summary}", f"<summary" in html and summary in html)
    check("Trace entry uses action and object wording", "查看Trace证据" in html)
    check("Draft publish action is explicit", "发布当前Draft" in html)

    # The interview guide covers the core default elements but remains explainable.
    rows = [
        line for line in guide.splitlines()
        if line.startswith("|")
        and "页面 | 页面元素" not in line
        and not set(line.replace("|", "").replace("-", "").replace(":", "").strip()) == set()
    ]
    rows = [line for line in rows if not line.startswith("|---")]
    check("guide covers at least twenty core elements", len(rows) >= 20)
    check("guide remains within thirty core elements", len(rows) <= 30)
    check("guide declares complete default-core coverage", "默认可见核心元素覆盖：100%" in guide)

    print({"status": "PASS", "checks": len(passed), "guide_items": len(rows)})


if __name__ == "__main__":
    main()
