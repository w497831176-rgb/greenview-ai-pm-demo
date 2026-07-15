"""
Idempotent runtime knowledge base brand migration.

Replaces "绿景智服" with "YIAI物业" in all knowledge document titles and
content, then reindexes affected documents so that vector chunks and search
results reflect the new brand. Running the script twice produces zero changes
the second time.
"""

import sys
from pathlib import Path

# Allow imports when running from repo root or scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import property_db as db
import rag_indexer

OLD_BRAND = "绿景智服"
NEW_BRAND = "YIAI物业"


def migrate() -> dict:
    docs = db.list_knowledge_docs()
    changed = 0
    reindexed = 0
    failed = 0
    for doc in docs:
        doc_id = doc["id"]
        title = doc.get("title") or ""
        content = doc.get("content") or ""
        new_title = title
        new_content = content
        replaced = False
        if OLD_BRAND in title or OLD_BRAND in content:
            new_title = title.replace(OLD_BRAND, NEW_BRAND)
            new_content = content.replace(OLD_BRAND, NEW_BRAND)
            replaced = True
        try:
            if replaced:
                db.update_knowledge_doc(
                    doc_id=doc_id,
                    title=new_title,
                    content=new_content,
                    category=doc.get("category"),
                )
                changed += 1
                print(f"[REPLACED] doc_id={doc_id} title={new_title[:40]}...")
            # Always reindex so vector chunks reflect the current DB content.
            if not rag_indexer.reindex_document(doc_id):
                raise RuntimeError("reindex failed")
            reindexed += 1
        except Exception as exc:
            failed += 1
            print(f"[FAILED] doc_id={doc_id}: {exc}")
    # Ensure retrieval settings allow enough evidence chunks for composite queries.
    settings = db.get_retrieval_settings("default")
    if settings and settings.get("top_k", 5) < 5:
        db.update_retrieval_settings("default", top_k=5)
        print("[SETTINGS] updated default top_k to 5")

    return {"total": len(docs), "changed": changed, "reindexed": reindexed, "failed": failed}


if __name__ == "__main__":
    result = migrate()
    print("\nMigration result:", result)
    if result["failed"] > 0:
        sys.exit(1)
