"""
YouTube MCP Executor using Microsoft Agent Framework.

This executor performs YouTube video search using MCP (Model Context Protocol) server.
Migrated from services_sk/youtube_mcp_plugin.py.
"""

import os
import json
import base64
import logging
from typing import Dict, Any, Optional, List

from agent_framework import Executor, WorkflowContext, handler
from i18n.locale_msg import LOCALE_MESSAGES
from utils.yield_message import (
    send_step_with_code,
    send_step_with_input,
    send_step_with_code_and_input,
)


logger = logging.getLogger(__name__)

# MCP client library imports
try:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning(
        "MCP library not installed. YouTube MCP functionality will be disabled."
    )

# YouTube MCP server configuration
YOUTUBE_MCP_SERVER_COMMAND = "youtube-data-mcp-server"
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


class YouTubeMCPExecutor(Executor):
    """
    Executor for YouTube video search using MCP server.

    This executor:
    - Connects to YouTube MCP server
    - Searches for videos using YouTube Data API
    - Returns formatted results with video metadata
    """

    def __init__(self, id: str, api_key: Optional[str] = None, max_results: int = 10):
        """
        Initialize the YouTube MCP executor.

        Args:
            id: Executor ID
            api_key: YouTube API key
            max_results: Maximum number of results to return
        """
        super().__init__(id=id)

        self.api_key = api_key or YOUTUBE_API_KEY
        self.max_results = max_results
        self.mcp_client = None
        self.mcp_session = None

        # Check MCP availability
        if not MCP_AVAILABLE:
            logger.warning(
                "MCP library not available. YouTube search will be disabled."
            )
        elif not self.api_key:
            logger.warning("YOUTUBE_API_KEY not set. YouTube search will be disabled.")
        else:
            logger.info("YouTubeMCPExecutor initialized successfully")

    async def _ensure_mcp_connection(self):
        """Ensure MCP server connection is established."""
        if self.mcp_session is None and MCP_AVAILABLE and self.api_key:
            try:
                # Clean up any existing connection
                await self.cleanup()

                # Create server parameters
                server_params = StdioServerParameters(
                    command=YOUTUBE_MCP_SERVER_COMMAND,
                    args=["--api-key", self.api_key] if self.api_key else [],
                    env={"YOUTUBE_API_KEY": self.api_key} if self.api_key else {},
                )

                # Create new connection
                self.mcp_client = stdio_client(server_params)
                read, write = await self.mcp_client.__aenter__()
                self.mcp_session = ClientSession(read, write)
                await self.mcp_session.__aenter__()
                await self.mcp_session.initialize()

                # Log available tools
                tools = await self.mcp_session.list_tools()
                tool_names = [tool.name for tool in tools.tools] if tools.tools else []
                logger.info(
                    f"YouTube MCP server connected. Available tools: {tool_names}"
                )

            except Exception as e:
                logger.error(f"MCP server connection failed: {e}")
                await self.cleanup()

        return self.mcp_session is not None

    @handler
    async def search(
        self,
        search_data: Dict[str, Any],
        ctx: WorkflowContext[Dict[str, Any], str],
    ) -> None:
        """YouTube search handler."""
        try:
            # Get metadata for verbose and locale
            metadata = search_data.get("metadata", {})
            locale = metadata.get("locale", "ko-KR")
            verbose = metadata.get("verbose", False)
            LOCALE_MSG = LOCALE_MESSAGES.get(locale, LOCALE_MESSAGES["ko-KR"])

            sub_topics = search_data.get("sub_topics", [])
            max_results = search_data.get("max_results", self.max_results)

            # ✅ Yield starting message
            await ctx.yield_output(f"data: ### {LOCALE_MSG['searching_YouTube']}\n\n")

            if not sub_topics:
                # No YouTube search needed
                await ctx.send_message(
                    {**search_data, "sub_topic_youtube_contexts": {}}
                )
                return

            logger.info(
                f"[YouTubeMCPExecutor] Searching YouTube for {len(sub_topics)} sub-topics"
            )

            if not MCP_AVAILABLE:
                error_msg = "MCP library not available"
                logger.error(f"[YouTubeMCPExecutor] {error_msg}")
                await ctx.send_message(
                    {
                        **search_data,
                        "sub_topic_youtube_contexts": {},
                        "youtube_error": error_msg,
                    }
                )
                return

            if not self.api_key:
                error_msg = "YouTube API key not configured"
                logger.error(f"[YouTubeMCPExecutor] {error_msg}")
                await ctx.send_message(
                    {
                        **search_data,
                        "sub_topic_youtube_contexts": {},
                        "youtube_error": error_msg,
                    }
                )
                return

            # Search by sub-topic
            sub_topic_results = {}

            for sub_topic_data in sub_topics:
                sub_topic_name = sub_topic_data.get("sub_topic", "research")
                queries = sub_topic_data.get("queries", [])

                sub_topic_videos = []

                for query in queries:
                    logger.info(
                        f"[YouTubeMCPExecutor] Searching YouTube for '{query}' in sub-topic '{sub_topic_name}'"
                    )

                    # Search YouTube videos
                    result = await self._search_youtube_videos(query, max_results)

                    if result.get("status") == "success" and result.get("videos"):
                        sub_topic_videos.extend(result["videos"])

                if sub_topic_videos:
                    # Store results keyed by sub_topic name
                    sub_topic_results[sub_topic_name] = {
                        "status": "success",
                        "videos": sub_topic_videos,
                        "total_results": len(sub_topic_videos),
                    }

            logger.info(
                f"[YouTubeMCPExecutor] Completed YouTube search for {len(sub_topic_results)} sub-topics"
            )

            # ✅ Yield completion message (SK compatible format with results)
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
                    f"data: {send_step_with_code(LOCALE_MSG['YouTube_done'], truncated)}\n\n"
                )
            else:
                await ctx.yield_output(f"data: ### {LOCALE_MSG['YouTube_done']}\n\n")

            # Send results to next executor (using SK-compatible key name)
            await ctx.send_message(
                {**search_data, "sub_topic_youtube_contexts": sub_topic_results}
            )

        except Exception as e:
            error_str = str(e)
            logger.error(f"[YouTubeExecutor] Search failed: {error_str}")

            await ctx.send_message(
                {
                    **search_data,
                    "sub_topic_youtube_contexts": {},
                    "executor_error": {
                        "executor": "youtube_search",
                        "error_type": "mcp_connection_failure",
                        "error_message": error_str,
                        "is_fatal": False,  # YouTube는 선택적이므로 치명적이지 않음
                    },
                }
            )

    async def _search_youtube_videos(
        self, query: str, max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Search YouTube videos using MCP searchVideos tool.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            Dictionary with search results
        """
        # Use a fresh connection for each search to avoid connection issues
        local_client = None
        local_session = None

        try:
            # Create temporary MCP connection
            server_params = StdioServerParameters(
                command=YOUTUBE_MCP_SERVER_COMMAND,
                args=["--api-key", self.api_key] if self.api_key else [],
                env={"YOUTUBE_API_KEY": self.api_key} if self.api_key else {},
            )

            local_client = stdio_client(server_params)
            read, write = await local_client.__aenter__()
            local_session = ClientSession(read, write)
            await local_session.__aenter__()
            await local_session.initialize()

            # Call searchVideos tool
            search_result = await local_session.call_tool(
                "searchVideos", {"query": query, "maxResults": max_results}
            )

            if getattr(search_result, "is_err", False):
                logger.error(f"YouTube search error: {search_result.content}")
                return {
                    "status": "error",
                    "message": f"Search failed: {search_result.content}",
                    "videos": [],
                    "search_query": query,
                }

            # Parse results
            videos_data = []
            if hasattr(search_result, "content") and search_result.content:
                try:
                    # Extract content from MCP response
                    for content_item in search_result.content:
                        if hasattr(content_item, "text"):
                            result_text = content_item.text

                            # Parse JSON from result
                            try:
                                result_json = json.loads(result_text)

                                # Handle different response formats
                                if isinstance(result_json, dict):
                                    if "items" in result_json:
                                        items = result_json["items"]
                                    elif "videos" in result_json:
                                        items = result_json["videos"]
                                    else:
                                        items = [result_json]
                                elif isinstance(result_json, list):
                                    items = result_json
                                else:
                                    items = []

                                # Process items
                                for item in items:
                                    video_id = item.get("id", {})
                                    if isinstance(video_id, dict):
                                        video_id = video_id.get("videoId", "")

                                    snippet = item.get("snippet", {})

                                    video_data = {
                                        "videoId": video_id,
                                        "title": snippet.get("title", "N/A"),
                                        "description": snippet.get(
                                            "description", "N/A"
                                        ),
                                        "channelTitle": snippet.get(
                                            "channelTitle", "N/A"
                                        ),
                                        "publishedAt": snippet.get(
                                            "publishedAt", "N/A"
                                        ),
                                        "thumbnails": snippet.get("thumbnails", {}),
                                    }

                                    videos_data.append(video_data)

                            except json.JSONDecodeError:
                                # If not JSON, treat as plain text result
                                logger.warning(
                                    f"Could not parse JSON from result: {result_text[:100]}"
                                )

                except Exception as e:
                    logger.error(f"Error parsing search results: {e}")

            return {
                "status": "success",
                "videos": videos_data,
                "search_query": query,
                "total_results": len(videos_data),
            }

        except Exception as e:
            logger.error(f"YouTube search error: {e}")
            return {
                "status": "error",
                "message": f"Search error: {str(e)}",
                "videos": [],
                "search_query": query,
            }
        finally:
            # Clean up temporary connection
            try:
                if local_session:
                    await local_session.__aexit__(None, None, None)
                if local_client:
                    await local_client.__aexit__(None, None, None)
            except Exception as cleanup_error:
                logger.warning(f"Cleanup error: {cleanup_error}")

    def format_results(self, videos: List[Dict[str, Any]]) -> str:
        """
        Format search results as human-readable text.

        Args:
            videos: List of video data dictionaries

        Returns:
            Formatted string
        """
        if not videos:
            return "❌ No search results found."

        formatted_results = []
        for i, video in enumerate(videos, 1):
            result_text = f"""
{i}. **{video.get("title", "N/A")}**
   📺 Channel: {video.get("channelTitle", "N/A")}
   📅 Published: {video.get("publishedAt", "N/A")}
   🔗 Link: https://www.youtube.com/watch?v={video.get("videoId", "")}
   📝 Description: {video.get("description", "N/A")[:150]}...
            """.strip()
            formatted_results.append(result_text)

        return "\n\n".join(formatted_results)

    def create_video_context(self, videos: List[Dict]) -> str:
        """
        Create context string from video information.

        Args:
            videos: List of video data dictionaries

        Returns:
            Context string
        """
        context_parts = []

        for i, video in enumerate(videos[:5], 1):  # Top 5 only
            video_context = f"""
Video {i}:
- Title: {video.get("title", "N/A")}
- Channel: {video.get("channelTitle", "N/A")}
- Description: {video.get("description", "N/A")[:200]}...
- URL: https://www.youtube.com/watch?v={video.get("videoId", "")}
- Published: {video.get("publishedAt", "N/A")}
            """.strip()
            context_parts.append(video_context)

        return "\n\n".join(context_parts)

    async def cleanup(self):
        """Clean up MCP connection."""
        try:
            # Clean up session first
            if self.mcp_session:
                try:
                    await self.mcp_session.__aexit__(None, None, None)
                except Exception as e:
                    logger.warning(f"Session cleanup error: {e}")

            # Clean up client
            if self.mcp_client:
                try:
                    await self.mcp_client.__aexit__(None, None, None)
                except Exception as e:
                    logger.warning(f"Client cleanup error: {e}")

        except Exception as e:
            logger.warning(f"MCP connection cleanup error: {e}")
        finally:
            # Force reset connection objects
            self.mcp_session = None
            self.mcp_client = None
