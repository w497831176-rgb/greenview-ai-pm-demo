"""
Bing Search Tools
=================

Optional web search toolkit. This demo keeps it as a no-op fallback so that
agents can start without a Bing API key.
"""

from agno.tools import Toolkit


class BingSearchTools(Toolkit):
    """No-op Bing search fallback for offline/demo deployments."""

    def __init__(self):
        super().__init__(name="bing_search_tools")

    def search_web(self, query: str) -> str:
        """Return a placeholder message indicating web search is unavailable."""
        return "当前环境未配置 Bing 搜索，建议基于知识库内容回答。"
