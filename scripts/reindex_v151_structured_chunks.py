"""Reindex active docs only when structural chunking changes evidence units."""

import argparse
import json

from db.property_db import list_knowledge_docs
import rag_chunking
import rag_indexer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write replacement vectors")
    args = parser.parse_args()
    candidates = []
    for doc in list_knowledge_docs():
        if not doc.get("is_indexed"):
            continue
        proposed = rag_chunking.split_text(
            doc.get("content") or "",
            strategy=doc.get("split_strategy") or "auto",
            chunk_size=doc.get("chunk_size") or 512,
            chunk_overlap=doc.get("chunk_overlap") or 64,
        )
        # Upgrade only one-block documents. Existing long documents already have
        # a stable chunk layout; changing them en masse would alter recall risk.
        if int(doc.get("chunk_count") or 0) <= 1 and len(proposed) > int(doc.get("chunk_count") or 0):
            candidates.append({
                "id": doc["id"],
                "title": doc.get("title"),
                "before": doc.get("chunk_count"),
                "after": len(proposed),
            })
    applied = []
    if args.apply:
        for candidate in candidates:
            if rag_indexer.reindex_document(candidate["id"]):
                applied.append(candidate["id"])
    print(json.dumps({
        "candidates": candidates,
        "applied": applied,
        "mode": "apply" if args.apply else "dry_run",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
