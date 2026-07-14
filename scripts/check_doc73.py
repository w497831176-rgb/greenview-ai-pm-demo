"""
Inspect and optionally disable the non-business test document used for badcase tests.

The document titled "测试 badcase" is a regression test fixture. It must remain in
DB for badcase workflow demos, but it should not be indexed for business RAG
evidence. This script checks its is_indexed flag and disables it if needed.

Usage:
    python scripts/check_doc73.py [--base-url BASE_URL] [--disable]
"""

import argparse
import json
import os
import sys

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("TEST_BASE_URL", "https://maiyouxiong.duckdns.org:18004"))
    parser.add_argument("--disable", action="store_true", help="Set is_indexed=0 if it is currently indexed")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    # List docs and find the test badcase doc by title (we only read, not hardcode id).
    resp = requests.get(f"{base}/api/knowledge/docs", timeout=30)
    resp.raise_for_status()
    docs = resp.json().get("knowledge_docs", [])

    target = None
    for d in docs:
        if d.get("title") == "测试 badcase":
            target = d
            break

    if not target:
        print("Test badcase doc not found in knowledge base.")
        sys.exit(0)

    print(json.dumps(target, ensure_ascii=False, indent=2))

    if args.disable and target.get("is_indexed"):
        doc_id = target["id"]
        patch_resp = requests.patch(
            f"{base}/api/knowledge/docs/{doc_id}/indexed",
            json={"is_indexed": False},
            timeout=30,
        )
        patch_resp.raise_for_status()
        print(f"Disabled indexing for doc {doc_id}.")
        updated = patch_resp.json().get("knowledge_doc", {})
        print(json.dumps(updated, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
