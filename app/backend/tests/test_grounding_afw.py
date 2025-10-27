"""
Tests for Grounding Executor using Agent Framework.

This module tests the GroundingExecutor for Bing Grounding API integration.
"""

import sys
import os
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from services_afw.grounding_executor import GroundingExecutor
from config.config import Settings


class TestGroundingExecutor:
    """Test suite for GroundingExecutor"""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock Settings"""
        settings = Mock(spec=Settings)
        settings.TIME_ZONE = "Asia/Seoul"
        settings.AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4"
        settings.AZURE_OPENAI_ENDPOINT = "https://test.openai.azure.com"
        settings.AZURE_OPENAI_API_KEY = "test_key"
        settings.AZURE_OPENAI_API_VERSION = "2024-08-01-preview"
        return settings
    
    @pytest.fixture
    def grounding_executor(self, mock_settings):
        """Create GroundingExecutor instance for testing"""
        with patch('services_afw.grounding_executor.AgentsClient'):
            with patch('services_afw.grounding_executor.DefaultAzureCredential'):
                executor = GroundingExecutor(
                    id="test_grounding",
                    settings=mock_settings,
                    project_endpoint="https://test.ai.azure.com",
                    connection_id="test_connection",
                    agent_model_deployment_name="gpt-4",
                    max_results=5,
                    market="ko-KR"
                )
                return executor
    
    @pytest.fixture
    def mock_context(self):
        """Create a mock workflow context"""
        context = Mock()
        context.emit = AsyncMock()
        return context
    
    @pytest.mark.asyncio
    async def test_grounding_search_single_query(self, grounding_executor, mock_context):
        """Test grounding search with single query"""
        # Mock the execution
        grounding_executor._execute_grounding_search = AsyncMock(
            return_value="Test answer with references"
        )
        
        search_data = {
            "search_queries": "What is Azure AI?",
            "max_tokens": 1024,
            "temperature": 0.7,
            "locale": "ko-KR"
        }
        
        await grounding_executor.grounding_search(search_data, mock_context)
        
        # Verify emit was called
        assert mock_context.emit.called
        emit_call = mock_context.emit.call_args[0][0]
        assert emit_call["result"] == "Test answer with references"
        assert emit_call["locale"] == "ko-KR"
    
    @pytest.mark.asyncio
    async def test_grounding_search_multiple_queries(self, grounding_executor, mock_context):
        """Test grounding search with multiple queries"""
        grounding_executor._execute_grounding_search = AsyncMock(
            return_value="Combined answers with references"
        )
        
        search_data = {
            "search_queries": '["What is Azure?", "What is AI?"]',
            "max_tokens": 1024,
            "temperature": 0.7,
            "locale": "en-US"
        }
        
        await grounding_executor.grounding_search(search_data, mock_context)
        
        # Verify emit was called
        assert mock_context.emit.called
        emit_call = mock_context.emit.call_args[0][0]
        assert "result" in emit_call
        assert emit_call["locale"] == "en-US"
    
    @pytest.mark.asyncio
    async def test_grounding_search_error_handling(self, grounding_executor, mock_context):
        """Test error handling in grounding search"""
        grounding_executor._execute_grounding_search = AsyncMock(
            side_effect=Exception("Grounding failed")
        )
        
        search_data = {
            "search_queries": "test query"
        }
        
        await grounding_executor.grounding_search(search_data, mock_context)
        
        # Verify error was emitted
        assert mock_context.emit.called
        emit_call = mock_context.emit.call_args[0][0]
        assert "error" in emit_call
        assert emit_call["result"] == ""
    
    @pytest.mark.asyncio
    async def test_execute_grounding_search_with_thread(self, grounding_executor):
        """Test grounding search execution with thread management"""
        # Mock agents client
        mock_thread = Mock()
        mock_thread.id = "test_thread_id"
        
        mock_message = Mock()
        mock_message.id = "test_message_id"
        
        mock_run = Mock()
        mock_run.id = "test_run_id"
        mock_run.status = "completed"
        
        grounding_executor.agents_client.threads.create = Mock(return_value=mock_thread)
        grounding_executor.agents_client.messages.create = Mock(return_value=mock_message)
        grounding_executor.agents_client.runs.create = Mock(return_value=mock_run)
        grounding_executor.agents_client.runs.get = Mock(return_value=mock_run)
        
        # Mock messages response
        mock_messages = Mock()
        mock_content = Mock()
        mock_content.text = Mock()
        mock_content.text.value = "Test response"
        mock_assistant_msg = Mock()
        mock_assistant_msg.role = "assistant"
        mock_assistant_msg.content = [mock_content]
        mock_messages.data = [mock_assistant_msg]
        grounding_executor.agents_client.messages.list = Mock(return_value=mock_messages)
        
        result = await grounding_executor._execute_grounding_search(
            search_queries="test query",
            max_tokens=1024,
            temperature=0.7,
            locale="ko-KR"
        )
        
        assert "Test response" in result
        assert grounding_executor.agents_client.threads.create.called
        assert grounding_executor.agents_client.messages.create.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
