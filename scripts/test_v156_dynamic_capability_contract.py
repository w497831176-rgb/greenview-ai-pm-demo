"""Fast no-model checks for V1.5.6 dynamic capability runtime.

Run inside demo-os-api.  It never opens an SSE stream or calls a provider.
"""

from pathlib import Path

from agents.router import _capability_fallback
from app.chat import _unique_rag_results
from app.mcp_policy import allowed_tools_for_agent


def agent(agent_id, name, description, *, enabled=True, skills=None, mcp=None):
    return {
        "agent_id": agent_id,
        "name": name,
        "description": description,
        "enabled": enabled,
        "capability_card": {
            "service_scope": description,
            "skills": skills or [],
            "mcp_servers": mcp or [],
        },
    }


def main():
    root = Path(__file__).resolve().parents[1]
    frontend_source = (root / "frontend" / "index.html").read_text(encoding="utf-8")
    chat_source = (root / "app" / "chat.py").read_text(encoding="utf-8")
    db_source = (root / "db" / "property_db.py").read_text(encoding="utf-8")
    assert "const APP_VERSION = 'V1.5.8'" in frontend_source
    assert "业务回答 Token" in frontend_source
    assert "本轮累计 Token" in frontend_source
    assert "Skill（业务规则）" in frontend_source
    assert "动态能力匹配" in frontend_source
    assert "round_token_count" in chat_source and "round_token_count" in db_source

    maintenance = agent("maintenance", "维修 Agent", "处理报修、漏水和维修工单")
    customer = agent("customer_service", "客服 Agent", "处理一般物业咨询")
    children = agent(
        "children_education",
        "儿童教育 Agent",
        "处理儿童托管、亲子课程和课后活动",
        skills=[{"name": "儿童托管服务", "positive_triggers": ["孩子托管", "亲子课程"]}],
    )
    elderly = agent(
        "elderly_care",
        "老年关怀 Agent",
        "处理长者活动、陪诊和适老服务",
        skills=[{"name": "长者关怀", "positive_triggers": ["老人陪诊", "老年活动"]}],
    )
    disabled = agent("pet_service", "宠物服务 Agent", "宠物寄养", enabled=False)
    registry = [maintenance, customer, children, elderly, disabled]

    target, reason, scores = _capability_fallback("厨房漏水需要报修", registry)
    assert target == "maintenance", (target, reason, scores)

    target, reason, scores = _capability_fallback("孩子放学后想参加亲子课程和托管", registry)
    assert target == "children_education", (target, reason, scores)
    assert reason.startswith("能力匹配路由："), reason
    assert "能力回退" not in reason, reason
    assert all(
        match["term"] != "agent"
        for score in scores
        for match in score["matches"]
    ), (reason, scores)
    assert scores[0]["matches"][0]["source"] != "Agent 名称", (reason, scores)

    # Asking "which Agent" alone is meta-language, not business intent.  It
    # must not randomly route to the first enabled dynamic Agent.
    target, reason, scores = _capability_fallback("请问应该由哪个 Agent 处理？", registry)
    assert target == "customer_service", (target, reason, scores)

    target, reason, scores = _capability_fallback("想咨询老人陪诊和老年活动", registry)
    assert target == "elderly_care", (target, reason, scores)

    dynamic_tools = allowed_tools_for_agent(
        "elderly_care",
        "elderly-service-server",
        bound_server_names={"elderly-service-server"},
        discovered_tool_names={"list_activities", "request_escort"},
    )
    assert dynamic_tools == {"list_activities", "request_escort"}, dynamic_tools
    assert not allowed_tools_for_agent(
        "elderly_care",
        "elderly-service-server",
        bound_server_names=set(),
        discovered_tool_names={"list_activities"},
    )

    deduped = _unique_rag_results([
        {"doc_id": 1, "chunk_index": 0, "content": "first"},
        {"doc_id": 1, "chunk_index": 0, "content": "duplicate"},
        {"doc_id": 2, "chunk_index": 1, "content": "second"},
    ])
    assert len(deduped) == 2, deduped
    print("V1.5.8 dynamic capability and explainability contract checks passed")


if __name__ == "__main__":
    main()
