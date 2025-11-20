"""
GraphRAG MCP Plugin

MCP 클라이언트로 GraphRAG MCP 서버와 통신하여 지식 그래프 검색 수행
"""

import os
import logging
import json
import asyncio
from pathlib import Path
from typing import Annotated, Optional
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from dotenv import load_dotenv
from config.config import Settings

# MCP client imports
try:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logging.warning("MCP library not installed. GraphRAG features disabled.")

load_dotenv(override=True)

logger = logging.getLogger(__name__)


class GraphRAGMCPPlugin:
    """GraphRAG MCP Client Plugin for Semantic Kernel"""
    
    def __init__(self):
        self.mcp_client = None
        self.mcp_session = None
        self.settings = Settings()
        self.graphrag_enabled = os.getenv("GRAPHRAG_ENABLED", "false").lower() == "true"
        self._connection_lock = asyncio.Lock()  # 동시성 제어
        self._is_connecting = False  # 연결 중 플래그
        
        if not MCP_AVAILABLE:
            logger.warning("MCP library not available. GraphRAG disabled.")
            self.graphrag_enabled = False
            return
        
        if not self.graphrag_enabled:
            logger.info("GraphRAG is disabled (GRAPHRAG_ENABLED=false)")
            return
        
        logger.info("GraphRAG MCP Plugin initialized")
    
    async def _ensure_mcp_connection(self) -> bool:
        """
        Ensure MCP server connection with proper concurrency control.
        
        Returns:
            bool: True if connection is available, False otherwise
        """
        # 이미 연결되어 있으면 재사용
        if self.mcp_session is not None:
            try:
                # 연결 상태 확인 (ping)
                await self.mcp_session.list_tools()
                return True
            except Exception as e:
                logger.warning(f"Existing MCP connection lost: {e}")
                await self.cleanup()
        
        if not MCP_AVAILABLE or not self.graphrag_enabled:
            return False
        
        # 동시에 여러 연결 시도 방지 (락 사용)
        async with self._connection_lock:
            # 락을 획득한 후 다시 확인 (다른 요청이 이미 연결했을 수 있음)
            if self.mcp_session is not None:
                return True
            
            if self._is_connecting:
                logger.info("MCP connection already in progress, waiting...")
                # 다른 스레드가 연결 중이면 대기
                await asyncio.sleep(1)
                return self.mcp_session is not None
            
            try:
                self._is_connecting = True
                logger.info("Establishing new MCP connection...")
                
                # GraphRAG 전용 가상환경의 Python 사용
                graphrag_root = Path(os.getenv("GRAPHRAG_ROOT", "/afh/code/multi-agent-doc-research/graphrag"))
                graphrag_venv_python = graphrag_root / ".venv" / "bin" / "python"
                graphrag_server_script = graphrag_root / "server.py"
                
                # 가상환경 존재 확인
                if not graphrag_venv_python.exists():
                    raise RuntimeError(
                        f"GraphRAG venv not found at {graphrag_venv_python}. "
                        f"Please run: cd {graphrag_root} && python -m venv .venv && .venv/bin/pip install -e ."
                    )
                
                server_params = StdioServerParameters(
                    command=str(graphrag_venv_python),  # ✅ GraphRAG 전용 Python 사용
                    args=[str(graphrag_server_script)],
                    env={
                        "GRAPHRAG_ROOT": str(graphrag_root),
                        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
                        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
                        "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION"),
                        "AZURE_OPENAI_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
                    }
                )
                
                self.mcp_client = stdio_client(server_params)
                read, write = await self.mcp_client.__aenter__()
                self.mcp_session = ClientSession(read, write)
                await self.mcp_session.__aenter__()
                await self.mcp_session.initialize()
                
                tools = await self.mcp_session.list_tools()
                tool_names = [tool.name for tool in tools.tools] if tools.tools else []
                logger.info(f"✅ GraphRAG MCP connected. Tools: {tool_names}")
                
                return True
            
            except Exception as e:
                logger.error(f"❌ GraphRAG MCP connection failed: {e}", exc_info=True)
                await self.cleanup()
                return False
            
            finally:
                self._is_connecting = False
    
    async def cleanup(self):
        """
        Cleanup MCP connection and subprocess.
        Ensures proper resource cleanup to prevent zombie processes.
        """
        logger.info("🧹 Cleaning up GraphRAG MCP connection...")
        
        if self.mcp_session:
            try:
                await self.mcp_session.__aexit__(None, None, None)
                logger.debug("MCP session closed")
            except (RuntimeError, Exception) as e:
                # Ignore task/cancel scope errors during cleanup
                logger.debug(f"MCP session cleanup error (ignored): {e}")
            finally:
                self.mcp_session = None
        
        if self.mcp_client:
            try:
                await self.mcp_client.__aexit__(None, None, None)
                logger.debug("MCP client closed")
            except (RuntimeError, Exception) as e:
                # Ignore task/cancel scope errors during cleanup
                logger.debug(f"MCP client cleanup error (ignored): {e}")
            finally:
                self.mcp_client = None
        
        logger.info("✅ GraphRAG MCP cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup on object deletion"""
        if self.mcp_session is not None or self.mcp_client is not None:
            logger.warning("⚠️ GraphRAGMCPPlugin deleted without explicit cleanup")
    
    @kernel_function(
        description="Index markdown documents in GraphRAG knowledge graph",
        name="index_documents"
    )
    async def index_documents(
        self,
        markdown_files: Annotated[str, "JSON string of markdown file paths"],
        force_reindex: Annotated[bool, "Force re-indexing"] = False
    ) -> str:
        """Index documents via GraphRAG MCP server"""
        try:
            if not self.graphrag_enabled:
                return json.dumps({
                    "status": "disabled",
                    "message": "GraphRAG is disabled. Set GRAPHRAG_ENABLED=true"
                })
            
            if not await self._ensure_mcp_connection():
                return json.dumps({
                    "status": "error",
                    "message": "GraphRAG MCP server not available"
                })
            
            files_list = json.loads(markdown_files) if isinstance(markdown_files, str) else markdown_files
            
            result = await self.mcp_session.call_tool(
                "index_documents",
                arguments={
                    "markdown_files": files_list,
                    "force_reindex": force_reindex
                }
            )
            
            response_text = result.content[0].text if result.content else "{}"
            logger.info(f"GraphRAG indexing result: {response_text}")
            
            return response_text
        
        except Exception as e:
            logger.error(f"GraphRAG indexing error: {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": str(e)
            })
    
    @kernel_function(
        description="Search GraphRAG knowledge graph with local search (entity-focused)",
        name="local_search"
    )
    async def local_search(
        self,
        query: Annotated[str, "Search query"],
        top_k: Annotated[int, "Number of results"] = 10
    ) -> str:
        """Local search via GraphRAG MCP server"""
        try:
            if not self.graphrag_enabled:
                return json.dumps({
                    "status": "disabled",
                    "message": "GraphRAG is disabled"
                })
            
            if not await self._ensure_mcp_connection():
                return json.dumps({
                    "status": "error",
                    "message": "GraphRAG MCP server not available"
                })
            
            result = await self.mcp_session.call_tool(
                "local_search",
                arguments={
                    "query": query,
                    "top_k": top_k
                }
            )

            response_text = result.content[0].text if result.content else "{}"
            logger.info(f"GraphRAG local search completed: {len(response_text)} chars")
            
            return response_text
        
        except Exception as e:
            logger.error(f"GraphRAG local search error: {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": str(e)
            })
    
    @kernel_function(
        description="Search GraphRAG knowledge graph with global search (community-focused)",
        name="global_search"
    )
    async def global_search(
        self,
        query: Annotated[str, "Search query"]
    ) -> str:
        """Global search via GraphRAG MCP server"""
        try:
            if not self.graphrag_enabled:
                return json.dumps({
                    "status": "disabled",
                    "message": "GraphRAG is disabled"
                })
            
            if not await self._ensure_mcp_connection():
                return json.dumps({
                    "status": "error",
                    "message": "GraphRAG MCP server not available"
                })
            
            result = await self.mcp_session.call_tool(
                "global_search",
                arguments={"query": query}
            )

            response_text = result.content[0].text if result.content else "{}"
            logger.info(f"GraphRAG global search completed: {len(response_text)} chars")
            
            return response_text
        
        except Exception as e:
            logger.error(f"GraphRAG global search error: {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": str(e)
            })
