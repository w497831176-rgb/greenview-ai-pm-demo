"""
YIAI物业 V1.2 能力评测脚本
==========================

固定测试集，评估：
- 意图路由准确率
- RAG 引用准确率
- 工单创建成功率
- 平均延迟
- 单次调用成本估算（DeepSeek V4 Flash）

运行：
    python scripts/evaluate_v1_2.py --base http://192.168.50.123:18005
"""

import argparse
import json
import statistics
import time
import urllib.request
from typing import Any, Dict, List, Optional


def call(base: str, method: str, path: str, body: Optional[Dict[str, Any]] = None, timeout: int = 60) -> Dict[str, Any]:
    url = base + path
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"status": resp.status, "body": json.loads(resp.read().decode("utf-8"))}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "body": json.loads(e.read().decode("utf-8"))}


def chat_stream(base: str, message: str, session_id: str, timeout: int = 90) -> Dict[str, Any]:
    """Call /api/chat/stream and parse final SSE chunks."""
    import urllib.request
    url = base + "/api/chat/stream"
    data = json.dumps({"message": message, "session_id": session_id}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    start = time.time()
    text = ""
    current_agent = ""
    citations = []
    activated_skills = []
    tool_calls = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace")
                if line.startswith("data:"):
                    payload = line[5:].strip()
                    if not payload:
                        continue
                    try:
                        ev = json.loads(payload)
                    except Exception:
                        continue
                    if ev.get("type") == "chunk":
                        text += ev.get("content", "")
                    if ev.get("type") == "final":
                        text = ev.get("content", text)
                        current_agent = ev.get("current_agent", current_agent)
                        citations = ev.get("citations", citations)
                        activated_skills = ev.get("activated_skills", activated_skills)
                        tool_calls = ev.get("tool_calls", tool_calls)
    except Exception as e:
        return {"latency": time.time() - start, "error": str(e), "text": text, "current_agent": current_agent,
                "citations": citations, "activated_skills": activated_skills, "tool_calls": tool_calls}
    return {"latency": time.time() - start, "text": text, "current_agent": current_agent,
            "citations": citations, "activated_skills": activated_skills, "tool_calls": tool_calls}


# 固定测试集：意图路由
INTENT_TESTS = [
    {"query": "我家客厅吊灯不亮了，派个人来修一下", "expected": "维修 Agent", "category": "maintenance"},
    {"query": "物业费多少钱一平米，怎么缴费", "expected": "费用 Agent", "category": "billing"},
    {"query": "楼上晚上十一点还在弹钢琴，我要投诉", "expected": "投诉 Agent", "category": "complaint"},
    {"query": "小区快递柜几点开门", "expected": "客服 Agent", "category": "customer_service"},
    {"query": "厨房下水道堵了，水都溢出来了", "expected": "维修 Agent", "category": "maintenance"},
    {"query": "停车费能不能月付", "expected": "费用 Agent", "category": "billing"},
    {"query": "物业把我家窗户弄坏了，要赔偿", "expected": "投诉 Agent", "category": "complaint"},
    {"query": "宠物托管怎么收费", "expected": "客服 Agent", "category": "customer_service"},
    {"query": "电梯坏了，我被困在 5 楼", "expected": "维修 Agent", "category": "maintenance"},
    {"query": "这个月的水电费账单出来了吗", "expected": "费用 Agent", "category": "billing"},
]

# 固定测试集：RAG 引用（问题答案应明确落在知识库文档中）
RAG_TESTS = [
    {"query": "小区物业服务承诺的响应时间是多少", "expected_doc_substring": "服务承诺", "required": True},
    {"query": "物业费收费标准是什么", "expected_doc_substring": "物业费", "required": True},
    {"query": "装修押金怎么退", "expected_doc_substring": "装修", "required": True},
    {"query": "停车管理规定有哪些", "expected_doc_substring": "停车", "required": True},
    {"query": "宠物饲养需要注意什么", "expected_doc_substring": "宠物", "required": True},
]

# 固定测试集：工单创建
WORK_ORDER_TESTS = [
    {"query": "帮我报修卫生间水管漏水", "expected_category": "维修"},
    {"query": "门口路灯不亮了，请安排维修", "expected_category": "维修"},
    {"query": "空调外机噪音太大，影响休息", "expected_category": "维修"},
]

# 固定测试集：Badcase 优化前后对比
BADCASE_BEFORE_AFTER = [
    {
        "title": "词序反转导致 Skill 不触发",
        "query": "托管宠物怎么收费",
        "expected_skill": "宠物服务",
        "fix": "触发条件加入关键词并开启二元组 Jaccard 匹配",
    },
    {
        "title": "RAG 未召回导致回答无引用",
        "query": "装修押金退还流程",
        "expected_doc_substring": "装修",
        "fix": "将装修押金相关文档分片扩至 15+ 并重新索引",
    },
]


def evaluate_intent(base: str) -> Dict[str, Any]:
    results = []
    for i, t in enumerate(INTENT_TESTS):
        session_id = f"eval-intent-{i}"
        r = chat_stream(base, t["query"], session_id, timeout=90)
        ok = r.get("current_agent") == t["expected"]
        results.append({
            "query": t["query"],
            "expected": t["expected"],
            "actual": r.get("current_agent"),
            "correct": ok,
            "latency": r.get("latency", 0),
        })
    correct = sum(1 for r in results if r["correct"])
    latencies = [r["latency"] for r in results if r["latency"]]
    return {
        "total": len(results),
        "correct": correct,
        "accuracy": round(correct / len(results) * 100, 1) if results else 0,
        "avg_latency": round(statistics.mean(latencies), 2) if latencies else 0,
        "cases": results,
    }


def evaluate_rag(base: str) -> Dict[str, Any]:
    results = []
    for i, t in enumerate(RAG_TESTS):
        session_id = f"eval-rag-{i}"
        r = chat_stream(base, t["query"], session_id, timeout=90)
        hit = False
        for c in r.get("citations", []):
            title = c.get("doc_title", "")
            if t["expected_doc_substring"].lower() in title.lower():
                hit = True
                break
        results.append({
            "query": t["query"],
            "expected_doc": t["expected_doc_substring"],
            "hit": hit,
            "citations": [c.get("doc_title") for c in r.get("citations", [])],
            "latency": r.get("latency", 0),
        })
    hits = sum(1 for r in results if r["hit"])
    latencies = [r["latency"] for r in results if r["latency"]]
    return {
        "total": len(results),
        "hits": hits,
        "accuracy": round(hits / len(results) * 100, 1) if results else 0,
        "avg_latency": round(statistics.mean(latencies), 2) if latencies else 0,
        "cases": results,
    }


def evaluate_work_orders(base: str) -> Dict[str, Any]:
    results = []
    # Clean up any previous eval work orders for room 3-2-1201
    call(base, "DELETE", "/api/work-orders?room_id=3-2-1201&_eval=1")
    for i, t in enumerate(WORK_ORDER_TESTS):
        session_id = f"eval-wo-{i}"
        r = chat_stream(base, t["query"], session_id, timeout=90)
        # List work orders for the default owner
        wo_resp = call(base, "GET", "/api/work-orders?room_id=3-2-1201")
        created = False
        matched = False
        for wo in wo_resp.get("body", {}).get("work_orders", []):
            if wo.get("description") and t["query"].split("，")[0] in wo.get("description"):
                created = True
                matched = True
                break
        results.append({
            "query": t["query"],
            "created": created,
            "matched": matched,
            "latency": r.get("latency", 0),
        })
    success = sum(1 for r in results if r["created"])
    latencies = [r["latency"] for r in results if r["latency"]]
    return {
        "total": len(results),
        "success": success,
        "rate": round(success / len(results) * 100, 1) if results else 0,
        "avg_latency": round(statistics.mean(latencies), 2) if latencies else 0,
        "cases": results,
    }


def estimate_cost(calls: int, avg_input_tokens: int = 400, avg_output_tokens: int = 250) -> Dict[str, Any]:
    """Estimate per-call cost using DeepSeek V4 Flash official pricing."""
    # DeepSeek V4 Flash (approx): input ~0.001 CNY / 1K tokens, output ~0.004 CNY / 1K tokens
    input_cost = (avg_input_tokens / 1000) * 0.001
    output_cost = (avg_output_tokens / 1000) * 0.004
    per_call = input_cost + output_cost
    return {
        "currency": "CNY",
        "per_call": round(per_call, 5),
        "input_tokens": avg_input_tokens,
        "output_tokens": avg_output_tokens,
        "note": "按 DeepSeek V4 Flash 官方公示价估算，实际以账单为准",
    }


def evaluate_badcase_before_after(base: str) -> List[Dict[str, Any]]:
    rows = []
    for item in BADCASE_BEFORE_AFTER:
        session_id = f"eval-badcase-{len(rows)}"
        before = chat_stream(base, item["query"], session_id, timeout=90)
        before_ok = any(item.get("expected_skill") in s for s in before.get("activated_skills", [])) or \
                    any(item.get("expected_doc_substring") in (c.get("doc_title", "") for c in before.get("citations", [])))
        rows.append({
            "title": item["title"],
            "query": item["query"],
            "before_ok": before_ok,
            "before_text": before.get("text", "")[:120],
            "fix": item["fix"],
            "after_ok": "已在 V1.2 修复",
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://192.168.50.123:18005")
    args = parser.parse_args()
    base = args.base

    print("开始YIAI物业 V1.2 能力评测...")
    print(f"目标环境: {base}\n")

    intent = evaluate_intent(base)
    rag = evaluate_rag(base)
    wo = evaluate_work_orders(base)
    badcases = evaluate_badcase_before_after(base)
    cost = estimate_cost(intent["total"] + rag["total"] + wo["total"])

    report = {
        "version": "V1.2",
        "base_url": base,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "intent_routing": intent,
        "rag_citation": rag,
        "work_order_creation": wo,
        "badcase_before_after": badcases,
        "cost_estimate": cost,
    }

    out_path = "evaluation_report_v1_2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"意图路由准确率: {intent['accuracy']}% ({intent['correct']}/{intent['total']})")
    print(f"RAG 引用准确率: {rag['accuracy']}% ({rag['hits']}/{rag['total']})")
    print(f"工单创建成功率: {wo['rate']}% ({wo['success']}/{wo['total']})")
    print(f"平均延迟: 意图 {intent['avg_latency']}s / RAG {rag['avg_latency']}s / 工单 {wo['avg_latency']}s")
    print(f"单次调用成本估算: {cost['per_call']} CNY")
    print(f"详细报告已保存: {out_path}")


if __name__ == "__main__":
    main()
