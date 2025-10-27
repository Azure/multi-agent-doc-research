"""
Test cases for Agent Framework (AFW) implementation.

This module tests the PlanSearchOrchestrator and related executors.
"""

import pytest
import asyncio
import os
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from config.config import Settings
from model.models import ChatMessage
from app.backend.services_afw.doc_research_orchestrator_afw import (
    PlanSearchOrchestrator,
    IntentAnalyzerExecutor,
    TaskPlannerExecutor,
    ResponseGeneratorExecutor
)
from agent_framework.azure import AzureOpenAIChatClient


@pytest.fixture
def settings():
    """Create settings for testing."""
    return Settings()


@pytest.fixture
def chat_client(settings):
    """Create Azure OpenAI chat client for testing."""
    return AzureOpenAIChatClient(
        model_id=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
        endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION
    )


@pytest.mark.asyncio
async def test_intent_analyzer_executor(settings, chat_client):
    """Test the IntentAnalyzerExecutor with intent_analyze_prompt.yaml."""
    print("\n=== Testing IntentAnalyzerExecutor ===")
    
    # Create executor
    executor = IntentAnalyzerExecutor(
        id="test_intent_analyzer",
        chat_client=chat_client,
        settings=settings
    )
    
    # Test messages
    messages = [
        {"role": "user", "content": "What is the latest news about artificial intelligence?"}
    ]
    
    # Create a simple workflow to test
    from agent_framework import WorkflowBuilder
    
    class TestReceiver:
        """Simple receiver to capture results."""
        received_data = None
        
        @staticmethod
        async def receive(data):
            TestReceiver.received_data = data
    
    # Build workflow with metadata
    builder = WorkflowBuilder()
    builder.set_start_executor(executor)
    workflow = builder.build()
    
    # Run workflow with locale metadata
    print("Running intent analysis with locale=ko-KR...")
    metadata = {"locale": "ko-KR"}
    events = await workflow.run(messages, metadata=metadata)
    
    # Get completed event
    completed_event = events.get_completed_event()
    print(f"Intent analysis completed: {completed_event is not None}")
    
    assert completed_event is not None, "Should receive intent analysis results"
    
    # Verify intent_analyze_prompt.yaml fields are present
    if hasattr(completed_event, 'data') and isinstance(completed_event.data, dict):
        intent_data = completed_event.data
        print(f"Intent detected: {intent_data.get('user_intent')}")
        print(f"Enriched query: {intent_data.get('enriched_query')}")
        print(f"Confidence: {intent_data.get('confidence')}")
        print(f"Keywords: {intent_data.get('keywords')}")
        
        # Check for fields defined in intent_analyze_prompt.yaml
        assert "user_intent" in intent_data
        assert "enriched_query" in intent_data
        assert "confidence" in intent_data
        assert "keywords" in intent_data
        assert "target_info" in intent_data
        
        # Validate intent value
        valid_intents = ["research", "general_query", "small_talk", "tool_calling"]
        assert intent_data["user_intent"] in valid_intents


@pytest.mark.asyncio
async def test_plan_search_executor_afw_basic(settings):
    """Test basic functionality of PlanSearchOrchestrator."""
    print("\n=== Testing PlanSearchOrchestrator Basic ===")
    
    # Create orchestrator
    executor = PlanSearchOrchestrator(settings)
    
    # Test messages
    messages = [
        ChatMessage(role="user", content="What is machine learning?")
    ]
    
    # Generate response (non-streaming)
    print("Generating response...")
    response_generator = executor.generate_response(
        messages=messages,
        max_tokens=500,
        temperature=0.7,
        research=False,  # Disable research for simple test
        planning=True,
        stream=False,
        locale="en-US",
        include_web_search=False,  # Disable web search for simple test
        include_ytb_search=False,
        include_mcp_server=False,
        include_ai_search=False,
        verbose=False
    )
    
    # Get response
    response = await response_generator.__anext__()
    
    print(f"Response received: {response[:100]}...")
    assert response is not None and len(response) > 0, "Should receive a response"


@pytest.mark.asyncio
async def test_plan_search_executor_afw_streaming(settings):
    """Test streaming functionality of PlanSearchOrchestrator."""
    print("\n=== Testing PlanSearchOrchestrator Streaming ===")
    
    # Create orchestrator
    executor = PlanSearchOrchestrator(settings)
    
    # Test messages
    messages = [
        ChatMessage(role="user", content="Tell me about Python programming language.")
    ]
    
    # Generate response (streaming)
    print("Generating streaming response...")
    response_generator = executor.generate_response(
        messages=messages,
        max_tokens=500,
        temperature=0.7,
        research=False,
        planning=False,  # Disable planning for simpler test
        stream=True,
        locale="en-US",
        include_web_search=False,
        include_ytb_search=False,
        include_mcp_server=False,
        include_ai_search=False,
        verbose=False
    )
    
    # Collect streaming chunks
    chunks = []
    async for chunk in response_generator:
        chunks.append(chunk)
        print(".", end="", flush=True)
    
    print(f"\nReceived {len(chunks)} chunks")
    assert len(chunks) > 0, "Should receive streaming chunks"


@pytest.mark.asyncio
async def test_plan_search_executor_afw_with_planning(settings):
    """Test PlanSearchOrchestrator with planning enabled."""
    print("\n=== Testing PlanSearchOrchestrator with Planning ===")
    
    # Create orchestrator
    executor = PlanSearchOrchestrator(settings)
    
    # Test messages
    messages = [
        ChatMessage(role="user", content="What are the benefits of cloud computing?")
    ]
    
    # Generate response with planning
    print("Generating response with planning...")
    response_generator = executor.generate_response(
        messages=messages,
        max_tokens=800,
        temperature=0.7,
        research=False,
        planning=True,  # Enable planning
        stream=False,
        locale="en-US",
        include_web_search=False,
        include_ytb_search=False,
        include_mcp_server=False,
        include_ai_search=False,
        verbose=True
    )
    
    # Get response
    response = await response_generator.__anext__()
    
    print(f"Response with planning received: {response[:100]}...")
    assert response is not None and len(response) > 0, "Should receive a response"


@pytest.mark.asyncio
async def test_plan_search_executor_afw_comparison_with_sk(settings):
    """Compare AFW and SK implementations."""
    print("\n=== Comparing AFW vs SK ===")
    
    from app.backend.services_sk.doc_research_orchestrator_sk import PlanSearchOrchestratorSK
    
    # Test messages
    messages = [
        ChatMessage(role="user", content="What is deep learning?")
    ]
    
    # Test AFW
    print("Testing AFW implementation...")
    afw_executor = PlanSearchOrchestrator(settings)
    afw_response_gen = afw_executor.generate_response(
        messages=messages,
        max_tokens=500,
        temperature=0.7,
        research=False,
        planning=False,
        stream=False,
        locale="en-US",
        include_web_search=False,
        include_ytb_search=False,
        include_mcp_server=False,
        include_ai_search=False,
        verbose=False
    )
    afw_response = await afw_response_gen.__anext__()
    print(f"AFW Response: {afw_response[:100]}...")
    
    # Test SK
    print("\nTesting SK implementation...")
    sk_executor = PlanSearchOrchestratorSK(settings)
    sk_response_gen = sk_executor.generate_response(
        messages=messages,
        max_tokens=500,
        temperature=0.7,
        research=False,
        planning=False,
        stream=False,
        locale="en-US",
        include_web_search=False,
        include_ytb_search=False,
        include_mcp_server=False,
        include_ai_search=False,
        verbose=False
    )
    sk_response = await sk_response_gen.__anext__()
    print(f"SK Response: {sk_response[:100]}...")
    
    # Both should produce responses
    assert afw_response is not None and len(afw_response) > 0, "AFW should produce response"
    assert sk_response is not None and len(sk_response) > 0, "SK should produce response"
    
    print("\n✓ Both implementations produced responses")


if __name__ == "__main__":
    """Run tests manually."""
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create settings
    settings = Settings()
    
    # Create chat client
    chat_client = AzureOpenAIChatClient(
        model_id=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
        endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION
    )
    
    print("=== Running AFW Tests Manually ===\n")
    
    # Run tests
    asyncio.run(test_intent_analyzer_executor(settings, chat_client))
    asyncio.run(test_plan_search_executor_afw_basic(settings))
    asyncio.run(test_plan_search_executor_afw_streaming(settings))
    asyncio.run(test_plan_search_executor_afw_with_planning(settings))
    asyncio.run(test_plan_search_executor_afw_comparison_with_sk(settings))
    
    print("\n=== All Tests Completed ===")
