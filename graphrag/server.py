"""
GraphRAG MCP Server

Independent MCP server providing GraphRAG functionality:
- Document indexing (markdown → knowledge graph) - uses CLI
- Local search (entity-focused queries) - uses SDK
- Global search (community-focused summaries) - uses SDK

Uses GraphRAG SDK for search operations for better control
"""

import asyncio
import logging
import os
import json
from pathlib import Path
from typing import Any, Dict, List

# MCP server imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Load environment variables
from dotenv import load_dotenv

# GraphRAG SDK imports for search
import pandas as pd
from graphrag.config.load_config import load_config
from graphrag.config.models.graph_rag_config import GraphRagConfig

# Import local query helper
import sys
sys.path.insert(0, str(Path(__file__).parent))
from query_helper import local_search, global_search

# Load .env from parent directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GraphRAGMCPServer:
    """GraphRAG MCP Server implementation"""
    
    def __init__(self):
        self.server = Server("graphrag-mcp-server")
        
        # Load GRAPHRAG_ROOT from environment
        graphrag_root_str = os.getenv("GRAPHRAG_ROOT")
        if not graphrag_root_str:
            raise ValueError("GRAPHRAG_ROOT environment variable not set")
        
        self.graphrag_root = Path(graphrag_root_str).resolve() 
        self.input_dir = self.graphrag_root / "input"
        self.output_dir = self.graphrag_root / "output"
        
        # Determine Python executable from virtual environment (for CLI indexing)
        self.venv_path = self.graphrag_root / ".venv"
        if self.venv_path.exists():
            self.python_executable = str(self.venv_path / "bin" / "python")
        else:
            self.python_executable = "python"
        
        logger.info(f"Using Python: {self.python_executable}")
        
        # Load GraphRAG configuration for SDK operations
        try:
            # GraphRAG 2.7: Use load_config() instead of from_root_dir()
            self.config = load_config(root_dir=self.graphrag_root)
            logger.info(f"✅ Loaded GraphRAG config from {self.graphrag_root}")
        except Exception as e:
            logger.error(f"❌ Could not load GraphRAG config: {e}", exc_info=True)
            self.config = None
        
        # Azure OpenAI settings
        self.openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        self.openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        
        # Create directories
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify settings.yaml exists
        settings_path = self.graphrag_root / "settings.yaml"
        if not settings_path.exists():
            raise FileNotFoundError(
                f"settings.yaml not found at {settings_path}. "
                "Please create it manually before running the server."
            )
        
        # Register handlers
        self._register_handlers()
        
        logger.info(f"GraphRAG MCP Server initialized. Root: {self.graphrag_root}")
        logger.info(f"Settings loaded from: {settings_path}")
    
    def _load_parquet_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load parquet files from output directory.
        
        GraphRAG 2.7+ stores parquet files directly in output/ directory:
        - entities.parquet
        - communities.parquet
        - community_reports.parquet
        - text_units.parquet
        - relationships.parquet
        
        Returns:
            Dict of dataframes with keys: entities, communities, community_reports, 
            text_units, relationships
        """
        logger.info(f"Loading parquet data from {self.output_dir}")
        
        data = {}
        required_files = ["entities", "communities", "community_reports"]
        optional_files = ["text_units", "relationships", "documents"]
        
        # Load required files
        for filename in required_files:
            parquet_file = self.output_dir / f"{filename}.parquet"
            if not parquet_file.exists():
                raise FileNotFoundError(
                    f"Required file not found: {parquet_file}. "
                    "Please run indexing first."
                )
            
            data[filename] = pd.read_parquet(parquet_file)
            logger.info(f"✅ Loaded {filename}: {len(data[filename])} rows")
        
        # Load optional files
        for filename in optional_files:
            parquet_file = self.output_dir / f"{filename}.parquet"
            
            if parquet_file.exists():
                data[filename] = pd.read_parquet(parquet_file)
                logger.info(f"✅ Loaded {filename}: {len(data[filename])} rows")
            else:
                data[filename] = None
                logger.info(f"⚠️  Optional file not found: {filename} (skipping)")
        
        return data
    
    def _register_handlers(self):
        """Register MCP tool handlers"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available GraphRAG tools"""
            return [
                Tool(
                    name="index_documents",
                    description="Index markdown documents to build knowledge graph",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "markdown_files": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of markdown file paths to index"
                            },
                            "force_reindex": {
                                "type": "boolean",
                                "description": "Force re-indexing even if files exist",
                                "default": False
                            }
                        },
                        "required": ["markdown_files"]
                    }
                ),
                Tool(
                    name="local_search",
                    description="Search knowledge graph for entity-focused, detailed information",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for local search"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of top results",
                                "default": 10
                            },
                            "generate_answer": {
                                "type": "boolean",
                                "description": "Generate LLM answer (slow) or return only context data (fast)",
                                "default": False
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="global_search",
                    description="Search knowledge graph for broad, community-focused summaries",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for global search"
                            },
                            "generate_answer": {
                                "type": "boolean",
                                "description": "Generate LLM answer (slow) or return only context data (fast)",
                                "default": False
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls"""
            try:
                if name == "index_documents":
                    result = await self._index_documents(
                        arguments.get("markdown_files", []),
                        arguments.get("force_reindex", False)
                    )
                elif name == "local_search":
                    result = await self._local_search_handler(
                        arguments.get("query"),
                        arguments.get("top_k", 10),
                        arguments.get("generate_answer", False)
                    )
                elif name == "global_search":
                    result = await self._global_search_handler(
                        arguments.get("query"),
                        arguments.get("generate_answer", False)
                    )
                else:
                    result = {"error": f"Unknown tool: {name}"}
                
                return [TextContent(
                    type="text",
                    text=json.dumps(result, ensure_ascii=False, indent=2)
                )]
            
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}", exc_info=True)
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)}, ensure_ascii=False)
                )]
    
    async def _index_documents(
        self, 
        markdown_files: List[str], 
        force_reindex: bool = False
    ) -> Dict[str, Any]:
        """Index markdown documents using GraphRAG CLI"""
        try:
            # Copy files to input directory
            copied_files = []
            for file_path in markdown_files:
                src_path = Path(file_path)
                if not src_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                dest_path = self.input_dir / src_path.name
                if dest_path.exists() and not force_reindex:
                    logger.info(f"File already indexed: {src_path.name}")
                    continue
                
                # Copy file
                dest_path.write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")
                copied_files.append(str(dest_path))
                logger.info(f"Copied {src_path.name} to GraphRAG input")
            
            if not copied_files and not force_reindex:
                return {
                    "status": "skipped",
                    "message": "All files already indexed",
                    "files_indexed": 0
                }
            
            # Run GraphRAG indexer using CLI
            logger.info("Starting GraphRAG indexing pipeline...")
            
            indexing_cmd = [
                self.python_executable, "-m", "graphrag", "index",
                "--root", str(self.graphrag_root),
                "--verbose"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *indexing_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.graphrag_root)
            )
            
            stdout, stderr = await process.communicate()
            
            # Check if indexing actually failed (not just RuntimeWarnings)
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                # Filter out LiteLLM RuntimeWarnings - they don't indicate actual failure
                if "RuntimeWarning" not in error_msg or "Error" in error_msg or "Exception" in error_msg:
                    logger.error(f"GraphRAG indexing failed: {error_msg}")
                    return {
                        "status": "error",
                        "message": f"Indexing failed: {error_msg}",
                        "files_indexed": 0
                    }
                else:
                    # Just a warning, check if output was actually generated
                    logger.warning(f"GraphRAG indexing completed with warnings: {error_msg}")
            
            # Verify indexing success by checking output directory
            output_files = list((self.graphrag_root / "output").glob("*.parquet"))
            if not output_files:
                logger.error("GraphRAG indexing produced no output files")
                return {
                    "status": "error",
                    "message": "Indexing produced no output files",
                    "files_indexed": 0
                }
            
            logger.info(f"GraphRAG indexing completed successfully. Output files: {len(output_files)}")
            
            return {
                "status": "success",
                "message": "Indexing completed",
                "files_indexed": len(copied_files) if copied_files else len(list(self.input_dir.glob("*.md"))),
                "files": copied_files if copied_files else [str(f) for f in self.input_dir.glob("*.md")],
                "output_files": [f.name for f in output_files]
            }
        
        except Exception as e:
            logger.error(f"Indexing error: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e),
                "files_indexed": 0
            }
    
    async def _local_search_handler(self, query: str, top_k: int = 10, generate_answer: bool = False) -> Dict[str, Any]:
        """
        Handle local search using GraphRAG SDK.
        
        Args:
            query: Search query
            top_k: Number of top results (not used in SDK but kept for compatibility)
            generate_answer: If True, generate LLM answer; if False, return only context
            
        Returns:
            Dict with search results
        """
        try:
            logger.info(f"Executing local search: {query} (generate_answer={generate_answer})")
            
            if not self.config:
                return {
                    "status": "error",
                    "message": "GraphRAG configuration not loaded"
                }
            
            # Load parquet data
            data = self._load_parquet_data()
            
            # Perform local search using helper
            result, context_data = await local_search(
                config=self.config,
                entities=data["entities"],
                communities=data["communities"],
                community_reports=data["community_reports"],
                text_units=data["text_units"],
                relationships=data["relationships"],
                covariates=data.get("covariates"),
                community_level=2,  # Default community level
                response_type="multiple paragraphs",
                query=query,
                generate_answer=generate_answer,  # ⭐ Pass parameter
            )
            
            response_len = len(str(result)) if result else 0
            logger.info(f"Local search completed: {response_len} chars (answer={'generated' if result else 'skipped'})")
            
            return {
                "status": "success",
                "query": query,
                "search_type": "local",
                "generate_answer": generate_answer,
                "response": str(result) if result else None,
                "context_data": self._serialize_context_data(context_data),
            }
        
        except FileNotFoundError as e:
            logger.error(f"Local search data error: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
        except Exception as e:
            logger.error(f"Local search error: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
    
    async def _global_search_handler(self, query: str, generate_answer: bool = False) -> Dict[str, Any]:
        """
        Handle global search using GraphRAG SDK.
        
        Args:
            query: Search query
            generate_answer: If True, generate LLM answer; if False, return only context
            
        Returns:
            Dict with search results
        """
        try:
            logger.info(f"Executing global search: {query} (generate_answer={generate_answer})")
            
            if not self.config:
                return {
                    "status": "error",
                    "message": "GraphRAG configuration not loaded"
                }
            
            # Load parquet data
            data = self._load_parquet_data()
            
            # Perform global search using helper
            result, context_data = await global_search(
                config=self.config,
                entities=data["entities"],
                communities=data["communities"],
                community_reports=data["community_reports"],
                community_level=2,  # Default community level
                dynamic_community_selection=False,
                response_type="multiple paragraphs",
                query=query,
                generate_answer=generate_answer,  # ⭐ Pass parameter
            )
            
            response_len = len(str(result)) if result else 0
            logger.info(f"Global search completed: {response_len} chars (answer={'generated' if result else 'skipped'})")
            
            return {
                "status": "success",
                "query": query,
                "search_type": "global",
                "generate_answer": generate_answer,
                "response": str(result) if result else None,
                "context_data": self._serialize_context_data(context_data),
            }
        
        except FileNotFoundError as e:
            logger.error(f"Global search data error: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
        except Exception as e:
            logger.error(f"Global search error: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
    
    def _serialize_context_data(self, context_data: Any) -> Dict[str, Any]:
        """
        Serialize context data for JSON response.
        
        Args:
            context_data: Context data from search (could be DataFrame, dict, or objects)
            
        Returns:
            Serializable dict
        """
        if context_data is None:
            return {}
        
        if isinstance(context_data, dict):
            serialized = {}
            for key, value in context_data.items():
                if isinstance(value, pd.DataFrame):
                    # DataFrame -> list of dicts
                    serialized[key] = value.to_dict(orient="records")
                elif isinstance(value, list):
                    # Handle list of various types
                    if len(value) > 0:
                        if isinstance(value[0], pd.DataFrame):
                            serialized[key] = [df.to_dict(orient="records") for df in value]
                        elif hasattr(value[0], '__dict__'):
                            # Objects (like Community) -> convert to dict
                            serialized[key] = [
                                {k: str(v) for k, v in obj.__dict__.items()} 
                                for obj in value
                            ]
                        else:
                            serialized[key] = [str(item) for item in value]
                    else:
                        serialized[key] = []
                elif hasattr(value, '__dict__'):
                    # Single object -> dict
                    serialized[key] = {k: str(v) for k, v in value.__dict__.items()}
                else:
                    # Primitive types or already serializable
                    try:
                        import json
                        json.dumps(value)  # Test if serializable
                        serialized[key] = value
                    except (TypeError, ValueError):
                        serialized[key] = str(value)
            return serialized
        elif isinstance(context_data, pd.DataFrame):
            return {"data": context_data.to_dict(orient="records")}
        else:
            return {"data": str(context_data)}
    
    async def run(self):
        """Run the MCP server"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point"""
    server = GraphRAGMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
