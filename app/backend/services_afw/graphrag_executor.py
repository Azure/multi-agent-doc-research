"""
GraphRAG Executor (Agent Framework)

Agent Framework 기반 GraphRAG 지식 그래프 검색 Executor
MCP 서버를 통해 local/global 검색 수행
"""

import os
import json
import logging
from typing import Dict, Any, Optional

from agent_framework import Executor, WorkflowContext, handler
from config.config import Settings
from services_sk.graphrag_mcp_plugin import GraphRAGMCPPlugin

logger = logging.getLogger(__name__)


class GraphRAGExecutor(Executor):
    """
    Executor for GraphRAG knowledge graph search using MCP server.
    
    This executor handles:
    - Indexing markdown documents into knowledge graph (operation="index")
    - Local search - entity-focused, detailed (operation="local_search")
    - Global search - community-focused, summaries (operation="global_search")
    
    Operation is determined by the "operation" field in input data.
    """
    
    def __init__(self, id: str, graphrag_enabled: Optional[bool] = None):
        """
        Initialize GraphRAG Executor.
        
        Args:
            id: Executor ID
            graphrag_enabled: Enable GraphRAG functionality (from env if not provided)
        """
        super().__init__(id=id)
        
        self.settings = Settings()
        self.graphrag_plugin = GraphRAGMCPPlugin()
        self.graphrag_enabled = (
            graphrag_enabled 
            if graphrag_enabled is not None 
            else os.getenv("GRAPHRAG_ENABLED", "false").lower() == "true"
        )
        
        logger.info(f"GraphRAGExecutor initialized (enabled: {self.graphrag_enabled})")
    
    @handler
    async def process_graphrag(
        self,
        data: Dict[str, Any],
        ctx: WorkflowContext[Dict[str, Any], str],
    ) -> Dict[str, Any]:
        """
        Unified handler for all GraphRAG operations.
        
        Args:
            data: {
                "operation": "index" | "local_search" | "global_search",
                # For indexing:
                "markdown_files": ["file1.md", "file2.md"],
                "force_reindex": bool,
                # For searching:
                "query": str,
                "top_k": int,
                # Common:
                "metadata": {"locale": "ko-KR", "verbose": bool}
            }
            ctx: Workflow context
            
        Returns:
            Dict with graphrag_result
        """
        operation = data.get("operation", "local_search")
        
        if operation == "index":
            return await self._index_documents(data, ctx)
        elif operation == "local_search":
            return await self._local_search(data, ctx)
        elif operation == "global_search":
            return await self._global_search(data, ctx)
        else:
            logger.error(f"Unknown operation: {operation}")
            return {
                **data,
                "graphrag_result": {
                    "status": "error",
                    "message": f"Unknown operation: {operation}"
                }
            }
    
    async def _index_documents(
        self,
        data: Dict[str, Any],
        ctx: WorkflowContext[Dict[str, Any], str],
    ) -> Dict[str, Any]:
        """
        Index markdown documents to build knowledge graph.
        
        Args:
            data: {
                "markdown_files": ["file1.md", "file2.md"],
                "force_reindex": bool,
                "metadata": {"locale": "ko-KR", "verbose": bool}
            }
            ctx: Workflow context
            
        Returns:
            Dict with graphrag_result
        """
        try:
            
            # Check if GraphRAG is enabled
            if not self.graphrag_enabled:
                logger.warning("GraphRAG is disabled")
                return {
                    **data,
                    "graphrag_result": {
                        "status": "disabled",
                        "message": "GraphRAG is disabled"
                    }
                }
            
            # Get parameters
            markdown_files = data.get("markdown_files", [])
            force_reindex = data.get("force_reindex", False)
            
            if not markdown_files:
                logger.warning("No markdown files provided for indexing")
                return {
                    **data,
                    "graphrag_result": {
                        "status": "error",
                        "message": "No markdown files provided"
                    }
                }
            
            logger.info(f"[GraphRAGExecutor] Indexing {len(markdown_files)} markdown files")
            
            # Call GraphRAG MCP plugin
            result_json = await self.graphrag_plugin.index_documents(
                markdown_files=json.dumps(markdown_files),
                force_reindex=force_reindex
            )
            
            result = json.loads(result_json)
            
            # Log completion message
            if result.get("status") == "success":
                success_msg = f"✅ Indexing completed: {result.get('files_indexed', 0)} files"
                logger.info(f"[GraphRAGExecutor] {success_msg}")
            else:
                error_msg = f"⚠️ Indexing failed: {result.get('message', 'Unknown error')}"
                logger.error(f"[GraphRAGExecutor] {error_msg}")
            
            # Return result
            return {
                **data,
                "graphrag_result": result
            }
        
        except Exception as e:
            logger.error(f"[GraphRAGExecutor] Indexing error: {e}", exc_info=True)
            return {
                **data,
                "graphrag_result": {
                    "status": "error",
                    "message": str(e)
                }
            }
    
    async def _local_search(
        self,
        data: Dict[str, Any],
        ctx: WorkflowContext[Dict[str, Any], str],
    ) -> Dict[str, Any]:
        """
        Perform local search (entity-focused, detailed information).
        
        Args:
            data: {
                "query": str,
                "top_k": int,
                "metadata": {"locale": "ko-KR", "verbose": bool}
            }
            ctx: Workflow context
            
        Returns:
            Dict with graphrag_local_result
        """
        try:
            
            # Check if GraphRAG is enabled
            if not self.graphrag_enabled:
                logger.warning("GraphRAG is disabled")
                result_data = {
                    **data,
                    "graphrag_local_result": {
                        "status": "disabled",
                        "message": "GraphRAG is disabled"
                    }
                }
                await ctx.send_message(result_data)
                return result_data
            
            # Get parameters
            query = data.get("query", "")
            top_k = data.get("top_k", 10)
            
            if not query:
                logger.warning("No query provided for local search")
                result_data = {
                    **data,
                    "graphrag_local_result": {
                        "status": "error",
                        "message": "No query provided"
                    }
                }
                await ctx.send_message(result_data)
                return result_data
            
            logger.info(f"[GraphRAGExecutor] Local search: {query}")
            
            # Perform local search via MCP
            result_json = await self.graphrag_plugin.local_search(
                query=query,
                top_k=top_k
            )
            
            result = json.loads(result_json)
            
            # Log completion message with detailed results
            if result.get("status") == "success":
                context_data = result.get("context_data", {})
                response_text = result.get("response") or ""
                # Log context data structure (GraphRAG 2.7 uses context_chunks and context_records)
                chunks_len = len(context_data.get("context_chunks", "")) if isinstance(context_data.get("context_chunks"), str) else 0
                
                # Extract structured records for detailed logging
                # TODO send_message with context_records
                context_records = context_data.get("context_records", {})
                entities = context_records.get("entities", [])
                relationships = context_records.get("relationships", [])
                reports = context_records.get("reports", [])
                sources = context_records.get("sources", [])
                
                logger.info("[GraphRAGExecutor] ✅ Local search completed:")
                logger.info(f"  - Context chunks: {chunks_len} chars")
                
                # Log structured records details
                if context_records:
                    for record_type, records in context_records.items():
                        if isinstance(records, list):
                            logger.info(f"  - {record_type}: {len(records)} items")
                        else:
                            logger.info(f"  - {record_type}: {type(records)}")
                
                # Show token usage
                logger.info(f"  - Prompt tokens: {context_data.get('prompt_tokens', 0)}")
                logger.info(f"  - Output tokens: {context_data.get('output_tokens', 0)}")
                logger.info(f"  - LLM calls: {context_data.get('llm_calls', 0)}")
                
                if response_text:
                    logger.info(f"  - Response length: {len(response_text)} chars")
                    logger.info(f"  - Response preview: {response_text[:200]}...")
                else:
                    logger.info("  - Response: None (context only mode)")
                    if chunks_len > 0:
                        logger.info(f"  - Context available for LLM generation: {chunks_len} chars")
            else:
                error_msg = f"⚠️ Search failed: {result.get('message', 'Unknown error')}"
                logger.error(f"[GraphRAGExecutor] {error_msg}")
            
            # Send result and return
            result_data = {
                **data,
                "graphrag_local_result": result
            }
            await ctx.send_message(result_data)
            return result_data
        
        except Exception as e:
            logger.error(f"[GraphRAGExecutor] Local search error: {e}", exc_info=True)
            result_data = {
                **data,
                "graphrag_local_result": {
                    "status": "error",
                    "message": str(e)
                }
            }
            await ctx.send_message(result_data)
            return result_data
    
    async def _global_search(
        self,
        data: Dict[str, Any],
        ctx: WorkflowContext[Dict[str, Any], str],
    ) -> None:
        """
        Perform global search (community-focused, thematic summaries).
        
        Args:
            data: {
                "query": str,
                "metadata": {"locale": "ko-KR", "verbose": bool}
            }
            ctx: Workflow context
        """
        try:
            # Get metadata
            
            # Check if GraphRAG is enabled
            if not self.graphrag_enabled:
                logger.warning("GraphRAG is disabled")
                await ctx.send_message({
                    **data,
                    "graphrag_global_result": {
                        "status": "disabled",
                        "message": "GraphRAG is disabled"
                    }
                })
                return
            
            # Get parameters
            query = data.get("query", "")
            
            if not query:
                logger.warning("No query provided for global search")
                await ctx.send_message({
                    **data,
                    "graphrag_global_result": {
                        "status": "error",
                        "message": "No query provided"
                    }
                })
                return
            
            logger.info(f"[GraphRAGExecutor] Global search: {query}")
            
            # Perform global search via MCP
            result_json = await self.graphrag_plugin.global_search(query=query)
            
            result = json.loads(result_json)
            # Log completion message with detailed results
            if result.get("status") == "success":
                context_data = result.get("context_data", {})
                response_text = result.get("response") or ""
                communities_count = len(context_data.get("communities", []))
                
                logger.info("[GraphRAGExecutor] ✅ Global search completed:")
                logger.info(f"  - Communities: {communities_count}")
                
                # Show context data summary
                if context_data:
                    # TODO send_message with reports
                    reports = context_data["reports"]
                    reports_str = str(reports)
                    context_preview = reports_str[:300] + "..." if len(reports_str) > 300 else reports_str
                    logger.info(f"  - Context data: {context_preview}")
                
                if response_text:
                    logger.info(f"  - Response length: {len(response_text)} chars")
                    logger.info(f"  - Response preview: {response_text[:200]}...")
                else:
                    logger.info("  - Response: None (context only mode)")
            else:
                error_msg = f"⚠️ Search failed: {result.get('message', 'Unknown error')}"
                logger.error(f"[GraphRAGExecutor] {error_msg}")
            
            # Send result and return
            result_data = {
                **data,
                "graphrag_global_result": result
            }
            await ctx.send_message(result_data)
            return result_data
        
        except Exception as e:
            logger.error(f"[GraphRAGExecutor] Global search error: {e}", exc_info=True)
            result_data = {
                **data,
                "graphrag_global_result": {
                    "status": "error",
                    "message": str(e)
                }
            }
            await ctx.send_message(result_data)
            return result_data
    
    async def cleanup(self):
        """Cleanup MCP connection resources."""
        try:
            await self.graphrag_plugin.cleanup()
            logger.info("GraphRAG executor cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
