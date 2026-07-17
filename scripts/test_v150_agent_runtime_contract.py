"""No-model contract test for V1.5 Agent workflow and MCP argument policy."""

from app import work_order_workflow as workflow
from app.chat import _policy_mcp_args


drafts = {}
created = {}


def fake_get(session_id):
    value = drafts.get(session_id)
    return dict(value) if value else None


def fake_save(session_id, **values):
    drafts[session_id] = {"session_id": session_id, **values}
    return fake_get(session_id)


def fake_delete(session_id):
    drafts.pop(session_id, None)


def fake_create(**values):
    created[values["work_order_id"]] = dict(values)
    return {"id": values["work_order_id"], **values}


workflow.get_work_order_draft = fake_get
workflow.save_work_order_draft = fake_save
workflow.delete_work_order_draft = fake_delete
workflow.create_work_order = fake_create

session = "DEMO_TEST_V150_WORKFLOW"
first = workflow.advance_work_order_workflow(session, "厨房漏水，我要报修")
assert first and first["action"] == "draft_updated"
assert first["missing_fields"] == ["紧急程度", "联系电话", "预约上门时间"]
assert not created

second = workflow.advance_work_order_workflow(session, "紧急。18927405209")
assert second and second["action"] == "awaiting_confirmation"
assert not second["missing_fields"]
assert not created

third = workflow.advance_work_order_workflow(session, "确认创建")
assert third and third["action"] == "created"
assert third["work_order_id"] in created
assert session not in drafts

assert _policy_mcp_args("weather-server", "get_current_weather", "查询上海天气") == {"city": "上海"}
assert _policy_mcp_args("weather-server", "get_current_weather", "查询天气") is None
assert _policy_mcp_args("workorder-server", "list_recent_work_orders", "查询我的最近工单") == {"room_id": "3-2-1201", "limit": 5}
assert _policy_mcp_args("workorder-server", "count_work_orders", "待处理工单有多少") == {"status": "pending"}

print("V1.5 agent runtime contract test passed")
