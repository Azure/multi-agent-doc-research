"""
Search plugin for semantic kernel that provides web search functionality.
Implements search functionality using httpx and scrapy selectors.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import httpx
from scrapy import Selector
from semantic_kernel.functions import kernel_function
import asyncio

logger = logging.getLogger(__name__)


class SearchPlugin:
    """
    Search plugin for semantic kernel that provides web search functionality.
    Uses httpx for HTTP requests and scrapy selectors for HTML parsing.
    """

    def __init__(
        self,
        bing_api_key: str = None,
        bing_endpoint: str = None,
        fallback_crawler=None,
        bing_custom_config_id: str = None,
    ):
        """
        Initialize the SearchPlugin with Bing Search API credentials.

        Args:
            bing_api_key: Bing Search API key
            bing_endpoint: Bing Search API endpoint
            fallback_crawler: Fallback search crawler to use if Bing API fails
            bing_custom_config_id: Bing custom config ID for custom search
        """
        self.bing_api_key = bing_api_key or os.getenv("BING_API_KEY")
        self.bing_endpoint = bing_endpoint or os.getenv(
            "BING_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search"
        )
        self.bing_custom_config_id = bing_custom_config_id or os.getenv(
            "BING_CUSTOM_CONFIG_ID"
        )
        self.fallback_crawler = fallback_crawler

        # Common HTTP client configuration
        self.client_config = {
            "timeout": 30.0,
            "headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            },
            "follow_redirects": True,
        }

        logger.info("SearchPlugin initialized with:")
        logger.info(f"  - Bing API Key: {'SET' if self.bing_api_key else 'NOT SET'}")
        logger.info(f"  - Bing Endpoint: {self.bing_endpoint}")
        logger.info(
            f"  - Bing Custom Config: {'SET' if self.bing_custom_config_id else 'NOT SET'}"
        )
        logger.info(
            f"  - Fallback Crawler: {'SET' if self.fallback_crawler else 'NOT SET'}"
        )

    async def cleanup(self):
        """Cleanup resources - no persistent sessions to clean"""
        pass

    @kernel_function(
        name="search_single_query", description="Search the web for the given query"
    )
    async def search_single_query(
        self,
        query: str,
        locale: str = "ko-KR",
        max_results: int = 5,
        max_context_length: int = 3000,
    ) -> str:
        """
        Perform a single web search query using semantic kernel.

        Args:
            query: The search query string
            locale: Locale for search (default: ko-KR)
            max_results: Maximum number of results to return

        Returns:
            JSON string containing search results and content
        """
        try:
            logger.info(f"Executing search for query: {query}")

            # Execute unified search
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

            # Extract content from top results
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
        logger.info(
            f"Starting Bing search for query: '{query}', locale: {locale}, max_results: {max_results}"
        )

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
