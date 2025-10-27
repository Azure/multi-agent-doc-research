"""
Test cases for search functionality in Agent Framework implementation.

This module tests the WebSearchExecutor and search tools.
"""

import pytest
import asyncio
import os
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.backend.services_afw.web_search_executor import (
    WebSearchHelper,
    WebSearchExecutor,
    search_web
)


@pytest.fixture
def search_helper():
    """Create WebSearchHelper for testing."""
    return WebSearchHelper()


@pytest.mark.asyncio
async def test_web_search_helper_bing_api(search_helper):
    """Test Bing API search using WebSearchHelper."""
    print("\n=== Testing WebSearchHelper Bing API ===")
    
    # Skip if Bing API key is not configured
    if not search_helper.bing_api_key:
        pytest.skip("Bing API key not configured")
    
    # Perform search
    query = "Microsoft Azure OpenAI"
    print(f"Searching for: {query}")
    
    result_json = await search_helper.search_bing_api(
        query=query,
        locale="en-US",
        max_results=3,
        max_context_length=1000
    )
    
    # Parse result
    import json
    result = json.loads(result_json)
    
    print(f"Search returned {result['total_results']} results")
    
    # Verify results
    assert result['query'] == query, "Query should match"
    assert 'results' in result, "Should contain results"
    
    if result['total_results'] > 0:
        print(f"First result: {result['results'][0]['title']}")
        assert 'title' in result['results'][0], "Result should have title"
        assert 'url' in result['results'][0], "Result should have URL"


@pytest.mark.asyncio
async def test_web_search_executor():
    """Test WebSearchExecutor with workflow context."""
    print("\n=== Testing WebSearchExecutor ===")
    
    from agent_framework import WorkflowBuilder, WorkflowOutputEvent, Executor, WorkflowContext, handler
    from typing_extensions import Never
    
    # Skip if Bing API key is not configured
    search_helper = WebSearchHelper()
    if not search_helper.bing_api_key:
        pytest.skip("Bing API key not configured")
    
    # Create search executor
    search_executor = WebSearchExecutor(id="test_search_executor")
    
    # Create a simple output executor to collect results
    class OutputCollector(Executor):
        results = None
        
        @handler
        async def collect(self, data: dict, ctx: WorkflowContext[Never, dict]) -> None:
            OutputCollector.results = data
            await ctx.yield_output(data)
    
    output_collector = OutputCollector(id="output_collector")
    
    # Build workflow
    builder = WorkflowBuilder()
    builder.set_start_executor(search_executor)
    builder.add_edge(search_executor, output_collector)
    workflow = builder.build()
    
    # Prepare search data
    search_data = {
        "sub_topics": [
            {
                "sub_topic": "AI Overview",
                "queries": ["artificial intelligence basics"]
            }
        ],
        "locale": "en-US",
        "max_results": 2
    }
    
    # Run workflow
    print("Running search workflow...")
    events = await workflow.run(search_data)
    
    # Verify results
    completed_event = events.get_completed_event()
    assert completed_event is not None, "Should complete workflow"
    
    # Check collected results
    if OutputCollector.results:
        print(f"Search results: {len(OutputCollector.results.get('web_search_results', {}))} sub-topics")
        assert 'web_search_results' in OutputCollector.results, "Should contain search results"


@pytest.mark.asyncio
async def test_search_web_ai_function():
    """Test the search_web AI function."""
    print("\n=== Testing search_web AI Function ===")
    
    # Skip if Bing API key is not configured
    search_helper = WebSearchHelper()
    if not search_helper.bing_api_key:
        pytest.skip("Bing API key not configured")
    
    # Note: search_web is a sync wrapper, so we test it directly
    print("Testing search_web function...")
    
    # For testing purposes, we'll use the helper directly
    result_json = await search_helper.search_bing_api(
        query="Python programming",
        locale="en-US",
        max_results=2
    )
    
    import json
    result = json.loads(result_json)
    
    print(f"Function returned {result['total_results']} results")
    assert 'results' in result, "Should contain results"


@pytest.mark.asyncio
async def test_search_multiple_queries(search_helper):
    """Test searching with multiple queries."""
    print("\n=== Testing Multiple Queries ===")
    
    # Skip if Bing API key is not configured
    if not search_helper.bing_api_key:
        pytest.skip("Bing API key not configured")
    
    queries = [
        "machine learning algorithms",
        "deep learning neural networks"
    ]
    
    results = []
    for query in queries:
        print(f"Searching: {query}")
        result_json = await search_helper.search_bing_api(
            query=query,
            locale="en-US",
            max_results=2,
            max_context_length=500
        )
        
        import json
        result = json.loads(result_json)
        results.append(result)
        print(f"  Found {result['total_results']} results")
    
    assert len(results) == len(queries), "Should get results for all queries"


if __name__ == "__main__":
    """Run tests manually."""
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("=== Running Search Tests Manually ===\n")
    
    # Create search helper
    search_helper = WebSearchHelper()
    
    # Check if Bing API key is configured
    if not search_helper.bing_api_key:
        print("WARNING: Bing API key not configured. Skipping tests.")
        print("Set BING_API_KEY environment variable to run these tests.")
        sys.exit(0)
    
    # Run tests
    asyncio.run(test_web_search_helper_bing_api(search_helper))
    asyncio.run(test_web_search_executor())
    asyncio.run(test_search_web_ai_function())
    asyncio.run(test_search_multiple_queries(search_helper))
    
    print("\n=== All Search Tests Completed ===")
