"""No-model acceptance for V1.5.2 Skill governance.

This test is intentionally deterministic: it verifies the same selector used
by chat and the management diagnostic without calling a provider or spending
any owner-chat Token.
"""

from app.skill_runtime import canonical_metadata, evaluate_skill_match, select_skills, skill_contract


maintenance = {
    "id": 1,
    "name": "维修受理 SOP",
    "enabled": True,
    "trigger_condition": "漏水、报修",
    "skill_metadata": {
        "version": "1.2.0",
        "positive_triggers": ["漏水", "报修"],
        "negative_triggers": ["仅咨询价格"],
        "priority": 80,
        "conflict_group": "repair_intake",
    },
}
fallback = {
    "id": 2,
    "name": "通用维修说明",
    "enabled": True,
    "trigger_condition": "报修",
    "skill_metadata": {"version": "1.0.0", "positive_triggers": ["报修"], "priority": 30, "conflict_group": "repair_intake"},
}
unconfigured = {"id": 3, "name": "无触发旧 Skill", "enabled": True, "trigger_condition": "", "skill_metadata": {}}
always_on = {"id": 4, "name": "默认安全提醒", "enabled": True, "trigger_condition": "", "skill_metadata": {"always_on": True, "version": "1.0.0"}}


negative = evaluate_skill_match(maintenance, "我仅咨询价格，不要报修")
assert negative["outcome"] == "negative_excluded"

no_trigger = evaluate_skill_match(unconfigured, "随便问一个问题")
assert no_trigger["matched"] is False
assert "不会隐式注入" in no_trigger["match_reason"]

selected, decisions = select_skills([fallback, maintenance], "厨房漏水，需要报修")
assert [item["skill_id"] for item in selected] == [1]
assert any(item["outcome"] == "conflict_skipped" for item in decisions)

selected, _ = select_skills([always_on, unconfigured], "普通问候")
assert [item["skill_id"] for item in selected] == [4]

metadata = canonical_metadata({"trigger_condition": "报修"}, {"negative_triggers": "仅咨询价格，暂不处理", "priority": "90"})
assert metadata["positive_triggers"] == ["报修"]
assert metadata["negative_triggers"] == ["仅咨询价格", "暂不处理"]
assert metadata["priority"] == 90
assert skill_contract({"skill_metadata": metadata})["contract_version"] == "1.0"

marker_skill = {
    "id": 5, "name": "Marker", "enabled": True,
    "trigger_condition": "DEMO_SKILL_V152",
    "skill_metadata": {"positive_triggers": ["DEMO_SKILL_V152"], "negative_triggers": ["NO_SKILL_V152"]},
}
assert evaluate_skill_match(marker_skill, "DEMO_SKILL_V152")["matched"] is True

print("V1.5.2 Skill governance contract test passed")
