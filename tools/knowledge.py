"""
Knowledge Tools
===============

Agno Toolkit wrapper around the property knowledge base.
"""

from typing import Optional

from agno.tools import Toolkit

from db.property_db import search_knowledge as db_search_knowledge


class KnowledgeTools(Toolkit):
    """Search the property knowledge base."""

    def __init__(self):
        super().__init__(name="knowledge_tools")

    def search_knowledge(self, query: str, top_k: int = 3) -> str:
        """Search property knowledge docs and return a summarized reference string."""
        results = db_search_knowledge(query, top_k=top_k)
        if not results:
            return "未在知识库中找到相关内容。"

        lines = []
        for idx, row in enumerate(results, 1):
            title = row.get("title", "未命名文档")
            content = row.get("content", "")[:400]
            lines.append(f"[{idx}]《{title}》：{content}")
        return "\n\n".join(lines)
