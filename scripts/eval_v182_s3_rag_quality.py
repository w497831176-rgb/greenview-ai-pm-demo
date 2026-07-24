"""Fixed 12-question retrieval evaluation for V1.8.2-S3.

This is deliberately a small Demo scorecard, not a general RAG benchmark.
It calls the existing retrieval debugger and never invokes a chat model.
"""

from __future__ import annotations

import argparse
import json
import urllib.request


ANSWER_CASES = [
    {
        "id": "O1",
        "group": "original",
        "query": "物业紧急维修的响应时效是多少？",
        "expected": [
            {"title": "物业维修服务承诺", "all": ["5 分钟", "30 分钟"]},
            {"title": "常见维修问题 FAQ", "all": ["紧急维修", "30 分钟"]},
        ],
    },
    {
        "id": "O2",
        "group": "original",
        "query": "业主投诉物业维修服务后，多久响应并反馈处理结果？",
        "expected": [
            {
                "title": "物业维修服务承诺",
                "all": ["24 小时内响应", "3 个工作日内反馈"],
            }
        ],
    },
    {
        "id": "O3",
        "group": "original",
        "query": "一般维修报修后多久响应并上门？",
        "expected": [
            {"title": "物业维修服务承诺", "all": ["2 小时内响应", "24 小时内上门"]},
            {"title": "常见维修问题 FAQ", "all": ["一般维修", "24 小时内上门"]},
        ],
    },
    {
        "id": "S1",
        "group": "paraphrase",
        "query": "家里突然爆管，报修后维修师傅最晚多久赶到现场？",
        "expected": [
            {"title": "物业维修服务承诺", "all": ["水管爆裂", "30 分钟内到场"]},
            {"title": "常见维修问题 FAQ", "all": ["紧急维修", "30 分钟内到场"]},
        ],
    },
    {
        "id": "S2",
        "group": "paraphrase",
        "query": "维修结束以后，物业多长时间会回访我？",
        "expected": [
            {"title": "物业维修服务承诺", "all": ["24 小时内", "回访"]}
        ],
    },
    {
        "id": "S3",
        "group": "paraphrase",
        "query": "灯坏了报修，物业多久联系我并安排上门？",
        "expected": [
            {"title": "物业维修服务承诺", "all": ["2 小时内响应", "24 小时内上门"]},
            {"title": "常见维修问题 FAQ", "all": ["一般维修", "24 小时内上门"]},
        ],
    },
    {
        "id": "D1",
        "group": "named_document",
        "query": "请依据《物业维修服务承诺》说明紧急维修的登记和到场时限。",
        "expected": [
            {"title": "物业维修服务承诺", "all": ["5 分钟", "30 分钟"]}
        ],
    },
    {
        "id": "D2",
        "group": "named_document",
        "query": "依据《常见维修问题 FAQ》，报修后多久能上门？",
        "expected": [
            {"title": "常见维修问题 FAQ", "all": ["紧急维修", "一般维修", "上门"]}
        ],
    },
]

NO_ANSWER_CASES = [
    ("N1", "小区是否提供无人机上门维修？"),
    ("N2", "维修期间物业会免费安排48小时酒店住宿吗？"),
    ("N3", "物业维修必须使用德国进口配件并提供终身保修吗？"),
    ("N4", "维修人员上门时，物业是否必须赠送三次免费保洁？"),
]


def _compact(value: str) -> str:
    return "".join((value or "").split())


def _matches(result, alternatives):
    title = str(result.get("doc_title") or result.get("title") or "")
    content = _compact(str(result.get("content") or ""))
    for expected in alternatives:
        if title != expected["title"]:
            continue
        if all(_compact(phrase) in content for phrase in expected["all"]):
            return True
    return False


def _hit_at(results, alternatives, k):
    return any(_matches(item, alternatives) for item in (results or [])[:k])


def _debug_http(base_url: str, query: str):
    body = json.dumps(
        {
            "query": query,
            "top_k": 5,
            "keyword_weight": 0.3,
            "semantic_weight": 0.7,
            "rrf_k": 60,
            "enable_rerank": False,
            "score_threshold": 0.0,
            "context_threshold": 0.2,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/api/retrieval/debug",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def _debug_direct(query: str):
    import rag_retrieval

    return rag_retrieval.debug_search(
        query=query,
        top_k=5,
        keyword_weight=0.3,
        semantic_weight=0.7,
        rrf_k=60,
        enable_rerank=False,
        score_threshold=0.0,
        context_threshold=0.2,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Import the local retrieval module instead of calling the HTTP API.",
    )
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    debug = _debug_direct if args.direct else lambda query: _debug_http(base_url, query)

    rows = []
    for case in ANSWER_CASES:
        data = debug(case["query"])
        row = {
            "id": case["id"],
            "group": case["group"],
            "query": case["query"],
            "keyword_hit_5": _hit_at(data.get("keyword_results"), case["expected"], 5),
            "vector_hit_5": _hit_at(data.get("semantic_results"), case["expected"], 5),
            "hybrid_hit_5": _hit_at(data.get("results"), case["expected"], 5),
            "hybrid_hit_3": _hit_at(data.get("results"), case["expected"], 3),
            "top_results": [
                {
                    "title": item.get("doc_title") or item.get("title"),
                    "chunk_index": item.get("chunk_index"),
                    "score": item.get("score"),
                    "content": str(item.get("content") or "")[:160],
                }
                for item in (data.get("results") or [])[:5]
            ],
        }
        rows.append(row)
        print(
            f"{row['id']} keyword={row['keyword_hit_5']} "
            f"vector={row['vector_hit_5']} hybrid5={row['hybrid_hit_5']} "
            f"hybrid3={row['hybrid_hit_3']}"
        )

    no_answer_rows = []
    for case_id, query in NO_ANSWER_CASES:
        data = debug(query)
        evidence = data.get("results") or []
        row = {
            "id": case_id,
            "query": query,
            "rejected": not evidence,
            "evidence_count": len(evidence),
            "top_results": [
                {
                    "title": item.get("doc_title") or item.get("title"),
                    "chunk_index": item.get("chunk_index"),
                    "score": item.get("score"),
                    "content": str(item.get("content") or "")[:160],
                }
                for item in evidence[:5]
            ],
        }
        no_answer_rows.append(row)
        print(f"{case_id} rejected={row['rejected']} evidence={len(evidence)}")

    metrics = {
        "keyword_hit_5": sum(row["keyword_hit_5"] for row in rows),
        "vector_hit_5": sum(row["vector_hit_5"] for row in rows),
        "hybrid_hit_5": sum(row["hybrid_hit_5"] for row in rows),
        "hybrid_hit_3": sum(row["hybrid_hit_3"] for row in rows),
        "answer_cases": len(rows),
        "no_answer_rejected": sum(row["rejected"] for row in no_answer_rows),
        "no_answer_cases": len(no_answer_rows),
    }
    report = {
        "suite": "V1.8.2-S3 fixed 12-question retrieval evaluation",
        "base_url": base_url,
        "metrics": metrics,
        "answer_cases": rows,
        "no_answer_cases": no_answer_rows,
        "note": "Citation correctness is verified separately with real SSE final events.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    passed = (
        metrics["hybrid_hit_5"] >= 7
        and metrics["hybrid_hit_3"] >= 6
        and metrics["no_answer_rejected"] == 4
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
