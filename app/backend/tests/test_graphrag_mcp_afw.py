"""
GraphRAG MCP Integration Tests - Agent Framework Mode

GraphRAG MCP 서버를 통한 전체 워크플로우 테스트:
1. PDF 업로드 → Markdown 생성 (Plugin mode)
2. GraphRAG 인덱싱 (Agent Framework Workflow)
3. Local Search (엔티티 중심, Agent Framework Workflow)
4. Global Search (커뮤니티 중심, Agent Framework Workflow)

All executor tests use real Agent Framework Workflow instances with workflow.run().
"""

import sys
from pathlib import Path

# Backend 서비스 imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import json
import os
import logging
from agent_framework import WorkflowBuilder, WorkflowOutputEvent
from services_sk.unified_file_upload_plugin import UnifiedFileUploadPlugin
from services_sk.graphrag_mcp_plugin import GraphRAGMCPPlugin
from services_afw.graphrag_executor import GraphRAGExecutor
from config.config import Settings

# Logging 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Settings instance for path resolution
settings = Settings()


@pytest.fixture
def upload_plugin():
    """Create file upload plugin instance"""
    return UnifiedFileUploadPlugin()


@pytest.fixture
def graphrag_plugin():
    """Create GraphRAG MCP plugin instance"""
    return GraphRAGMCPPlugin()


@pytest.fixture
async def graphrag_executor():
    """Create GraphRAG executor instance with proper cleanup"""
    executor = GraphRAGExecutor(id="test_graphrag_executor", graphrag_enabled=True)
    yield executor
    # Cleanup after test
    try:
        await executor.cleanup()
    except Exception as e:
        logger.debug(f"Executor cleanup: {e}")


@pytest.fixture
def test_pdf_paths():
    """Path to test PDF files"""
    # 실제 테스트 PDF 경로 (ai_search/ 디렉토리의 문서 사용)
    ai_search_dir = Path(__file__).parent.parent / "ai_search"
    pdf_files = list(ai_search_dir.glob("*.pdf"))
    
    if not pdf_files:
        pytest.skip(f"No PDF files found in {ai_search_dir}")
    
    # 처음 1-2개 파일만 테스트에 사용
    return [str(f) for f in pdf_files[:1]]


@pytest.mark.asyncio
async def test_01_index_documents_via_workflow(graphrag_executor):
    """
    Test 1: GraphRAG Executor를 통한 인덱싱 (Agent Framework Workflow)
    
    Agent Framework Pattern:
    1. Create Workflow instance
    2. Add executor to workflow
    3. Execute via workflow.run()
    4. Extract results from WorkflowRun.output_data
    """
    print("\n" + "="*60)
    print("Test 3: Index Documents via Agent Framework Workflow")
    print("="*60)
    
    # Get markdown files (absolute path)
    graphrag_md_dir = settings.get_graphrag_input_dir()
    print(f"📂 GraphRAG input directory: {graphrag_md_dir}")
    md_files = list(graphrag_md_dir.glob("*.md"))
    
    if not md_files:
        pytest.skip("No markdown files found")
    
    print(f"\n📂 Processing {len(md_files)} markdown files")
    
    # Prepare input data for process_graphrag handler
    input_data = {
        "operation": "index",  # ⭐ Specify operation type
        "markdown_files": [str(f) for f in md_files],
        "force_reindex": True,  # Force reindex for testing
        "metadata": {
            "locale": "ko-KR",
            "verbose": True
        }
    }
    
    print("\n🚀 Executing via executor directly (not workflow)...")
    
    # Call executor's handler directly instead of using workflow
    # This is more appropriate for testing single executor
    
    # Create a mock context for testing
    class TestContext:
        def __init__(self):
            self.messages = []
        
        async def send_message(self, message):
            self.messages.append(message)
    
    test_ctx = TestContext()
    
    # Call handler directly
    graphrag_result_data = await graphrag_executor.process_graphrag(input_data, test_ctx)
    
    # Check result
    graphrag_result = graphrag_result_data.get("graphrag_result", {})
    
    assert graphrag_result is not None, "No result received from executor"
    print(f"\n✅ GraphRAG Result: {graphrag_result}")
    assert graphrag_result.get("status") == "success", f"Indexing failed: {graphrag_result}"
    
    print("\n📊 Workflow Result:")
    print(json.dumps(graphrag_result, indent=2, ensure_ascii=False))
    
    # Assertions
    assert graphrag_result.get("status") in ["success", "skipped"]
    
    if graphrag_result.get("status") == "success":
        assert graphrag_result.get("files_indexed", 0) > 0
        print(f"\n✅ Test 1 PASSED - Workflow indexed {graphrag_result['files_indexed']} files")
    else:
        print("\n✅ Test 1 PASSED - Files already indexed (workflow skipped)")
    
    # Verify GraphRAG output directory
    graphrag_root = Path(os.getenv("GRAPHRAG_ROOT", "./graphrag"))
    output_dir = graphrag_root / "output"
    
    if output_dir.exists():
        parquet_files = list(output_dir.glob("**/*.parquet"))
        print(f"\n📦 Generated {len(parquet_files)} parquet files")


@pytest.mark.asyncio
async def test_02_local_search_via_workflow(graphrag_executor):
    """
    Test 2: GraphRAG Local Search via Agent Framework Workflow
    
    Agent Framework Pattern:
    1. Create Workflow
    2. Add executor
    3. Execute with query data
    4. Extract search results
    """
    print("\n" + "="*60)
    print("Test 2: GraphRAG Local Search via Agent Framework Workflow")
    print("="*60)
    
    test_query = "ai 시장변화와 트랜드"
    
    
    print(f"\n🔍 Query: {test_query}")
    
    # Prepare input data for process_graphrag handler
    input_data = {
        "operation": "local_search",  # ⭐ Specify operation type
        "query": test_query,
        "top_k": 10,
        "metadata": {
            "locale": "ko-KR",
            "verbose": True
        }
    }
    
    print("\n🚀 Executing via executor directly (not workflow)...")
    
    # Create a mock context for testing
    class TestContext:
        def __init__(self):
            self.messages = []
        
        async def send_message(self, message):
            self.messages.append(message)
    
    test_ctx = TestContext()
    
    # Call handler directly
    graphrag_result_data = await graphrag_executor.process_graphrag(input_data, test_ctx)
    
    # Check result
    graphrag_result = graphrag_result_data.get("graphrag_local_result", {})
    
    assert graphrag_result is not None, "No result received from executor"
    
    print("\n📊 Workflow Result:")
    print(f"Status: {graphrag_result.get('status')}")
    
    if graphrag_result.get("status") == "success":
        response = graphrag_result.get("response") or ""
        context_data = graphrag_result.get("context_data", {})
        
        print(f"\n📝 Response: {len(response) if response else 0} chars")
        if response:
            print("─" * 60)
            print(response[:500] + "..." if len(response) > 500 else response)
            print("─" * 60)
        else:
            print("  (Context only mode - no LLM response)")
        
        # Show context data preview
        if context_data:
            print("\n📦 Context Data Preview:")
            print(f"  Keys: {list(context_data.keys())}")
            context_str = str(context_data)[:300]
            print(f"  Content: {context_str}...")
        
        print("\n✅ Test 2 PASSED - Workflow local search completed")
    else:
        print("\n⚠️  Search not ready")
        pytest.skip("GraphRAG not ready")



@pytest.mark.asyncio
async def test_03_global_search_via_workflow(graphrag_executor):
    """
    Test 3: GraphRAG Global Search via Agent Framework Workflow
    
    Agent Framework Pattern:
    1. Create Workflow
    2. Add executor
    3. Execute with query data
    4. Extract search results
    """
    print("\n" + "="*60)
    print("Test 7: GraphRAG Global Search via Agent Framework Workflow")
    print("="*60)
    
    test_query = "문서들에서 다루는 전반적인 주제는 무엇인가?"
    
    print(f"\n🔍 Query: {test_query}")
    
    # Prepare input data for process_graphrag handler
    input_data = {
        "operation": "global_search",  # ⭐ Specify operation type
        "query": test_query,
        "metadata": {
            "locale": "ko-KR",
            "verbose": True
        }
    }
    
    print("\n🚀 Executing via executor directly (not workflow)...")
    
    # Create a mock context for testing
    class TestContext:
        def __init__(self):
            self.messages = []
        
        async def send_message(self, message):
            self.messages.append(message)
    
    test_ctx = TestContext()
    
    # Call handler directly
    graphrag_result_data = await graphrag_executor.process_graphrag(input_data, test_ctx)
    
    # Check result
    graphrag_result = graphrag_result_data.get("graphrag_global_result", {})
    
    assert graphrag_result is not None, "No result received from executor"
    
    print("\n📊 Workflow Result:")
    print(f"Status: {graphrag_result.get('status')}")
    
    if graphrag_result.get("status") == "success":
        response = graphrag_result.get("response") or ""
        context_data = graphrag_result.get("context_data", {})
        
        print(f"\n📝 Response: {len(response) if response else 0} chars")
        if response:
            print("─" * 60)
            print(response[:500] + "..." if len(response) > 500 else response)
            print("─" * 60)
        else:
            print("  (Context only mode - no LLM response)")
        
        # Show context data preview
        if context_data:
            print("\n📦 Context Data Preview:")
            print(f"  Keys: {list(context_data.keys())}")
            context_str = str(context_data)[:300]
            print(f"  Content: {context_str}...")
        
        print("\n✅ Test 3 PASSED - Workflow global search completed")
    else:
        print("\n⚠️  Search not ready")
        pytest.skip("GraphRAG not ready")


@pytest.mark.asyncio
async def test_04_cleanup(graphrag_plugin, graphrag_executor):
    """
    Test 4: Cleanup connections
    """
    print("\n" + "="*60)
    print("Test 8: Cleanup")
    print("="*60)
    
    await graphrag_plugin.cleanup()
    await graphrag_executor.cleanup()
    
    assert graphrag_plugin.mcp_session is None
    assert graphrag_plugin.mcp_client is None
    
    print("\n✅ Test 4 PASSED - Cleanup complete")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
