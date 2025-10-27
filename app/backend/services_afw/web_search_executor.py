"""
Search functionality for Microsoft Agent Framework.

This module provides search executors and tools for web search using Bing API,
migrated from Semantic Kernel's search_plugin.py.
"""

import os
import json
import logging
import base64
from typing import List, Dict, Any, Optional, Annotated, AsyncGenerator
import httpx
import asyncio
from scrapy import Selector

from agent_framework import (
    Executor,
    WorkflowContext,
    handler,
    ai_function,
)
from typing_extensions import Never
from i18n.locale_msg import LOCALE_MESSAGES
from utils.yield_message import (
    send_step_with_code,
    send_step_with_input,
    send_step_with_code_and_input,
)

logger = logging.getLogger(__name__)


# AI Function for web search (can be used as a tool by ChatAgents)
@ai_function(description="Search the web for information using Bing Search API")
def search_web(
    query: Annotated[str, "The search query"],
    locale: Annotated[str, "Locale for search results (e.g., ko-KR, en-US)"] = "ko-KR",
    max_results: Annotated[int, "Maximum number of results to return"] = 5,
) -> str:
    """
    Search the web for information using Bing Search API.

    Args:
        query: The search query
        locale: Locale for search results
        max_results: Maximum number of results to return

    Returns:
        JSON string containing search results
    """
    # This is a synchronous wrapper for the async function
    # In practice, this would need to be called in an async context
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    searcher = WebSearchHelper()
    return loop.run_until_complete(searcher.search_bing_api(query, locale, max_results))


class WebSearchHelper:
    """
    Helper class for web search operations.

    This class provides the core search functionality that can be used
    by both AI functions and executors.
    """

    def __init__(
        self,
        bing_api_key: str = None,
        bing_endpoint: str = None,
        bing_custom_config_id: str = None,
    ):
        """
        Initialize the WebSearchHelper with Bing Search API credentials.

        Args:
            bing_api_key: Bing Search API key
            bing_endpoint: Bing Search API endpoint
            bing_custom_config_id: Bing custom config ID for custom search
        """
        self.bing_api_key = bing_api_key or os.getenv("BING_API_KEY")
        self.bing_endpoint = bing_endpoint or os.getenv(
            "BING_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search"
        )
        self.bing_custom_config_id = bing_custom_config_id or os.getenv(
            "BING_CUSTOM_CONFIG_ID"
        )

        # Common HTTP client configuration
        self.client_config = {
            "timeout": 30.0,
            "headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            },
            "follow_redirects": True,
        }

        logger.info("WebSearchHelper initialized with:")
        logger.info(f"  - Bing API Key: {'SET' if self.bing_api_key else 'NOT SET'}")
        logger.info(f"  - Bing Endpoint: {self.bing_endpoint}")

    async def search_bing_api(
        self,
        query: str,
        locale: str = "ko-KR",
        max_results: int = 5,
        max_context_length: int = 3000,
    ) -> str:
        """
        Perform a web search using Bing API.

        Args:
            query: The search query
            locale: Locale for search results
            max_results: Maximum number of results to return
            max_context_length: Maximum length of content to extract

        Returns:
            JSON string containing search results and content
        """
        try:
            logger.info(f"Executing search for query: {query}")

            # Execute Bing API search
            results = await self._search_bing_api(query, locale, max_results)

            if not results:
                return json.dumps(
                    {
                        "query": query,
                        "results": [],
                        "total_results": 0,
                        "error": "No search results found",
                    }
                )

            # Enrich results with content
            enriched_results = await self._enrich_results_with_content(
                results, max_results, max_context_length
            )

            response_data = {
                "query": query,
                "results": enriched_results,
                "total_results": len(enriched_results),
            }

            logger.info(
                f"Search completed successfully. Found {len(enriched_results)} results."
            )
            return json.dumps(response_data, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return json.dumps(
                {"query": query, "results": [], "total_results": 0, "error": str(e)}
            )

    async def _search_bing_api(
        self, query: str, locale: str = "ko-KR", max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Bing API search implementation.

        Args:
            query: Search query
            locale: Locale for search
            max_results: Maximum number of results

        Returns:
            List of search results
        """
        logger.info(f"Starting Bing search for query: '{query}', locale: {locale}")

        if not self.bing_api_key:
            logger.error("Bing API key is not configured")
            return []

        # Determine endpoint based on custom config
        if self.bing_custom_config_id:
            endpoint = "https://api.bing.microsoft.com/v7.0/custom/search"
        else:
            endpoint = "https://api.bing.microsoft.com/v7.0/search"

        headers = {
            "Ocp-Apim-Subscription-Key": self.bing_api_key,
            **self.client_config["headers"],
        }

        params = {
            "q": f"{query} -filetype:pdf",
            "count": max_results,
            "offset": 0,
            "mkt": locale,
            "safesearch": "Moderate",
            "responseFilter": "Webpages",
        }

        if self.bing_custom_config_id:
            params["customconfig"] = self.bing_custom_config_id

        try:
            async with httpx.AsyncClient(**self.client_config) as client:
                response = await client.get(endpoint, headers=headers, params=params)

                if response.status_code == 200:
                    data = response.json()
                    web_pages = data.get("webPages", {}).get("value", [])

                    results = []
                    for page in web_pages:
                        results.append(
                            {
                                "title": page.get("name", ""),
                                "url": page.get("url", ""),
                                "snippet": page.get("snippet", ""),
                            }
                        )

                    logger.info(f"Bing API returned {len(results)} results")
                    return results
                else:
                    logger.error(
                        f"Bing API error: {response.status_code} - {response.text}"
                    )
                    return []

        except Exception as e:
            logger.error(f"Bing API request failed: {e}")
            return []

    async def _enrich_results_with_content(
        self, results: List[Dict[str, Any]], max_results: int, max_context_length: int
    ) -> List[Dict[str, Any]]:
        """
        Enrich search results with content extracted from URLs (parallel processing).
        """

        async def enrich(result, rank):
            try:
                content = await self._extract_content_from_url(
                    result.get("url", ""), max_context_length
                )
                return {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("snippet", ""),
                    "content": content or result.get("snippet", ""),
                    "rank": rank,
                }
            except Exception as e:
                logger.warning(
                    f"Failed to extract content from {result.get('url', '')}: {e}"
                )
                return {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("snippet", ""),
                    "content": result.get("snippet", ""),
                    "rank": rank,
                }

        tasks = [
            enrich(result, i + 1) for i, result in enumerate(results[:max_results])
        ]
        enriched_results = await asyncio.gather(*tasks)
        return enriched_results

    async def _extract_content_from_url(
        self, url: str, max_context_length: int
    ) -> Optional[str]:
        """
        Extract content from a URL.

        Args:
            url: URL to extract content from
            max_context_length: Maximum length of content

        Returns:
            Extracted text content or None if failed
        """
        if not url:
            return None

        try:
            # Remove 'timeout' from self.client_config to avoid duplicate
            client_config = {
                k: v for k, v in self.client_config.items() if k != "timeout"
            }
            async with httpx.AsyncClient(timeout=15.0, **client_config) as client:
                response = await client.get(url)

                if response.status_code != 200:
                    logger.warning(
                        f"Failed to fetch content from {url}: {response.status_code}"
                    )
                    return None

                html_content = response.text

                # Use scrapy selector for HTML parsing
                selector = Selector(text=html_content)

                # Remove unwanted elements
                for unwanted in selector.css(
                    "script, style, nav, footer, header, aside"
                ).getall():
                    html_content = html_content.replace(unwanted, "")

                # Extract text content with priority selectors
                selector = Selector(text=html_content)

                # Try to extract main content with priority order
                main_content = None
                main_selectors = [
                    "main ::text",
                    "article ::text",
                    ".content ::text",
                    "#content ::text",
                    ".post ::text",
                    "#post ::text",
                    "body ::text",  # Final fallback
                ]

                for main_selector in main_selectors:
                    content_elements = selector.css(main_selector).getall()
                    if content_elements:
                        main_content = content_elements
                        break

                if not main_content:
                    return None

                # Clean and join text
                text = " ".join(text.strip() for text in main_content if text.strip())

                # Limit content length
                if len(text) > max_context_length:
                    text = text[:max_context_length] + "..."

                return text if text else None

        except Exception as e:
            logger.warning(f"Content extraction failed for {url}: {e}")
            return None


class WebSearchExecutor(Executor):
    """
    Executor that performs web search using Bing API.

    This executor receives search queries (either single query or list of queries)
    and performs web search, forwarding the results to downstream executors.
    """

    def __init__(
        self,
        id: str,
        bing_api_key: str = None,
        bing_endpoint: str = None,
        bing_custom_config_id: str = None,
    ):
        super().__init__(id=id)
        self._search_helper = WebSearchHelper(
            bing_api_key=bing_api_key,
            bing_endpoint=bing_endpoint,
            bing_custom_config_id=bing_custom_config_id,
        )

    @handler
    async def search_single(
        self,
        search_data: Dict[str, Any],
        ctx: WorkflowContext[Dict[str, Any], str],  # Added str for yield_output
    ) -> None:
        """
        Perform web search for queries in search_data.

        Args:
            search_data: Dictionary containing search queries and parameters
            ctx: Workflow context for sending search results
        """
        try:
            logger.info("WebSearchExecutor: Starting web search")

            # Get metadata for verbose and locale
            metadata = search_data.get("metadata", {})
            locale = metadata.get("locale", "ko-KR")
            verbose = metadata.get("verbose", False)
            LOCALE_MSG = LOCALE_MESSAGES.get(locale, LOCALE_MESSAGES["ko-KR"])

            # ✅ Yield starting message
            await ctx.yield_output(f"data: ### {LOCALE_MSG['searching']}\n\n")

            sub_topics = search_data.get("sub_topics", [])
            max_results = search_data.get("max_results", 3)

            if not sub_topics:
                # No search needed
                await ctx.send_message(
                    {
                        **search_data,
                        "sub_topic_web_contexts": {},  # Changed from web_search_results
                    }
                )
                return

            # Search by sub-topic
            sub_topic_results = {}

            for sub_topic_data in sub_topics:
                sub_topic_name = sub_topic_data.get("sub_topic", "research")
                queries = sub_topic_data.get("queries", [])

                sub_topic_search_results = []

                for query in queries:
                    logger.info(
                        f"WebSearchExecutor: Searching for '{query}' in sub-topic '{sub_topic_name}'"
                    )

                    search_result_json = await self._search_helper.search_bing_api(
                        query=query,
                        locale=locale,
                        max_results=max_results,
                        max_context_length=3000,
                    )

                    search_result = json.loads(search_result_json)

                    if search_result.get("results"):
                        sub_topic_search_results.append(search_result)

                if sub_topic_search_results:
                    sub_topic_results[sub_topic_name] = sub_topic_search_results

            logger.info(
                f"WebSearchExecutor: Completed search for {len(sub_topic_results)} sub-topics"
            )

            if verbose and sub_topic_results:
                results_str = json.dumps(
                    sub_topic_results, ensure_ascii=False, indent=2
                )
                truncated = (
                    results_str[:200] + "... [truncated for display]"
                    if len(results_str) > 200
                    else results_str
                )
                await ctx.yield_output(
                    f"data: {send_step_with_code(LOCALE_MSG['search_done'], truncated)}\n\n"
                )
            else:
                await ctx.yield_output(f"data: ### {LOCALE_MSG['search_done']}\n\n")

            # Add search results to search_data (using SK-compatible key name)
            await ctx.send_message(
                {
                    **search_data,
                    "sub_topic_web_contexts": sub_topic_results,  # Changed from web_search_results
                }
            )

        except Exception as e:
            logger.error(f"WebSearchExecutor: Error during search: {e}")
            await ctx.send_message(
                {
                    **search_data,
                    "sub_topic_web_contexts": {},  # Changed from web_search_results
                    "search_error": str(e),
                }
            )

    @handler
    async def search(
        self,
        search_data: Dict[str, Any],
        ctx: WorkflowContext[Dict[str, Any], str],
    ) -> None:
        """Web search handler."""
        try:
            logger.info("WebSearchExecutor: Starting web search")

            # Get metadata for verbose and locale
            metadata = search_data.get("metadata", {})
            locale = metadata.get("locale", "ko-KR")
            verbose = metadata.get("verbose", False)
            LOCALE_MSG = LOCALE_MESSAGES.get(locale, LOCALE_MESSAGES["ko-KR"])

            # ✅ Yield starting message
            await ctx.yield_output(f"data: ### {LOCALE_MSG['searching']}\n\n")

            sub_topics = search_data.get("sub_topics", [])
            max_results = search_data.get("max_results", 3)

            if not sub_topics:
                # No search needed
                await ctx.send_message(
                    {
                        **search_data,
                        "sub_topic_web_contexts": {},  # Changed from web_search_results
                    }
                )
                return

            # Search by sub-topic
            sub_topic_results = {}

            for sub_topic_data in sub_topics:
                sub_topic_name = sub_topic_data.get("sub_topic", "research")
                queries = sub_topic_data.get("queries", [])

                sub_topic_search_results = []

                for query in queries:
                    logger.info(
                        f"WebSearchExecutor: Searching for '{query}' in sub-topic '{sub_topic_name}'"
                    )

                    search_result_json = await self._search_helper.search_bing_api(
                        query=query,
                        locale=locale,
                        max_results=max_results,
                        max_context_length=3000,
                    )

                    search_result = json.loads(search_result_json)

                    if search_result.get("results"):
                        sub_topic_search_results.append(search_result)

                if sub_topic_search_results:
                    sub_topic_results[sub_topic_name] = sub_topic_search_results

            logger.info(
                f"WebSearchExecutor: Completed search for {len(sub_topic_results)} sub-topics"
            )

            if verbose and sub_topic_results:
                results_str = json.dumps(
                    sub_topic_results, ensure_ascii=False, indent=2
                )
                truncated = (
                    results_str[:200] + "... [truncated for display]"
                    if len(results_str) > 200
                    else results_str
                )
                await ctx.yield_output(
                    f"data: {send_step_with_code(LOCALE_MSG['search_done'], truncated)}\n\n"
                )
            else:
                await ctx.yield_output(f"data: ### {LOCALE_MSG['search_done']}\n\n")

            # Add search results to search_data (using SK-compatible key name)
            await ctx.send_message(
                {
                    **search_data,
                    "sub_topic_web_contexts": sub_topic_results,  # Changed from web_search_results
                }
            )

        except Exception as e:
            error_str = str(e)
            logger.error(f"[WebSearchExecutor] Search failed: {error_str}")

            # ✅ 동일한 형식으로 에러 전달
            await ctx.send_message(
                {
                    **search_data,
                    "sub_topic_web_contexts": {},
                    "executor_error": {
                        "executor": "web_search",
                        "error_type": "api_failure",
                        "error_message": error_str,
                        "is_fatal": False,  # 또는 False (치명적이지 않을 수도)
                    },
                }
            )
