"""
Tests for Multi-Agent Research Executor using Agent Framework.

This module tests the MultiAgentResearchExecutor for parallel writer/reviewer workflows.
"""

import sys
import os
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from services_afw.multi_agent_executor import MultiAgentResearchExecutor, SubTopicTask
from config.config import Settings
from agent_framework.azure import AzureOpenAIChatClient


class TestMultiAgentResearchExecutor:
    """Test suite for MultiAgentResearchExecutor"""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock Settings"""
        settings = Mock(spec=Settings)
        settings.TIME_ZONE = "Asia/Seoul"
        settings.AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4"
        settings.AZURE_OPENAI_ENDPOINT = "https://test.openai.azure.com"
        settings.AZURE_OPENAI_API_KEY = "test_key"
        settings.AZURE_OPENAI_API_VERSION = "2024-08-01-preview"
        settings.MAX_TOKENS = 8000
        settings.DEFAULT_TEMPERATURE = 0.7
        return settings
    
    @pytest.fixture
    def mock_chat_client(self):
        """Create mock AzureOpenAIChatClient"""
        return Mock(spec=AzureOpenAIChatClient)
    
    @pytest.fixture
    def multi_agent_executor(self, mock_chat_client, mock_settings):
        """Create MultiAgentResearchExecutor instance for testing"""
        executor = MultiAgentResearchExecutor(
            id="test_multi_agent",
            chat_client=mock_chat_client,
            settings=mock_settings,
            context_max_chars=20000,
            writer_parallel_limit=4
        )
        return executor
    
    @pytest.fixture
    def mock_context(self):
        """Create a mock workflow context"""
        context = Mock()
        context.emit = AsyncMock()
        return context
    
    @pytest.mark.asyncio
    async def test_run_multi_agent_single_topic(self, multi_agent_executor, mock_context):
        """Test multi-agent research with single sub-topic"""
        # Mock the workflow execution
        multi_agent_executor._run_research_workflow = AsyncMock(
            return_value={
                "status": "success",
                "question": "Test question",
                "all_ready_to_publish": True,
                "results": [
                    {
                        "sub_topic": "Test Topic",
                        "draft": {"answer": "Test answer"},
                        "review": {"ready_to_publish": True, "status": "success"}
                    }
                ]
            }
        )
        
        research_data = {
            "question": "Test question",
            "sub_topics": ["Test Topic"],
            "sub_topic_contexts": '{"Test Topic": "Test context"}',
            "locale": "ko-KR",
            "max_tokens": 8000,
            "parallel": True
        }
        
        await multi_agent_executor.run_multi_agent(research_data, mock_context)
        
        # Verify emit was called
        assert mock_context.emit.called
        emit_call = mock_context.emit.call_args[0][0]
        assert emit_call["status"] == "success"
        assert emit_call["all_ready_to_publish"] == True
    
    @pytest.mark.asyncio
    async def test_run_multi_agent_multiple_topics(self, multi_agent_executor, mock_context):
        """Test multi-agent research with multiple sub-topics"""
        multi_agent_executor._run_research_workflow = AsyncMock(
            return_value={
                "status": "success",
                "question": "Test question",
                "all_ready_to_publish": True,
                "results": [
                    {"sub_topic": "Topic 1", "draft": {}, "review": {"ready_to_publish": True, "status": "success"}},
                    {"sub_topic": "Topic 2", "draft": {}, "review": {"ready_to_publish": True, "status": "success"}}
                ],
                "total_sub_topics": 2
            }
        )
        
        research_data = {
            "question": "Test question",
            "sub_topics": '["Topic 1", "Topic 2"]',
            "locale": "en-US",
            "max_tokens": 8000,
            "parallel": True,
            "parallel_limit": 2
        }
        
        await multi_agent_executor.run_multi_agent(research_data, mock_context)
        
        # Verify emit was called
        assert mock_context.emit.called
        emit_call = mock_context.emit.call_args[0][0]
        assert emit_call["total_sub_topics"] == 2
    
    @pytest.mark.asyncio
    async def test_run_multi_agent_error_handling(self, multi_agent_executor, mock_context):
        """Test error handling in multi-agent research"""
        multi_agent_executor._run_research_workflow = AsyncMock(
            side_effect=Exception("Research failed")
        )
        
        research_data = {
            "question": "Test question",
            "sub_topics": ["Test Topic"]
        }
        
        await multi_agent_executor.run_multi_agent(research_data, mock_context)
        
        # Verify error was emitted
        assert mock_context.emit.called
        emit_call = mock_context.emit.call_args[0][0]
        assert emit_call["status"] == "error"
        assert "Research failed" in emit_call["message"]
    
    @pytest.mark.asyncio
    async def test_invoke_writer(self, multi_agent_executor):
        """Test writer invocation"""
        # Mock ChatAgent response
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = '{"sub_topic": "Test", "answer": "Test answer", "key_points": [], "sources": []}'
        mock_response.contents = [mock_content]
        
        with patch('services_afw.multi_agent_executor.ChatAgent') as MockChatAgent:
            mock_agent = AsyncMock()
            mock_agent.get_response = AsyncMock(return_value=mock_response)
            MockChatAgent.return_value = mock_agent
            
            task = SubTopicTask(
                sub_topic="Test Topic",
                question="Test question",
                contexts="Test context",
                max_tokens=8000
            )
            
            result = await multi_agent_executor._invoke_writer(
                task=task,
                locale="ko-KR",
                current_date="2024-01-01"
            )
            
            assert result["status"] == "success"
            assert result["sub_topic"] == "Test Topic"
            assert "draft" in result
    
    @pytest.mark.asyncio
    async def test_invoke_reviewer(self, multi_agent_executor):
        """Test reviewer invocation"""
        # Mock ChatAgent response
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = '{"ready_to_publish": true, "feedback": "Good", "improvements": [], "final_answer": "Final answer"}'
        mock_response.contents = [mock_content]
        
        with patch('services_afw.multi_agent_executor.ChatAgent') as MockChatAgent:
            mock_agent = AsyncMock()
            mock_agent.get_response = AsyncMock(return_value=mock_response)
            MockChatAgent.return_value = mock_agent
            
            task = SubTopicTask(
                sub_topic="Test Topic",
                question="Test question",
                contexts="Test context",
                max_tokens=8000
            )
            
            writer_parsed = {
                "sub_topic": "Test Topic",
                "answer": "Test answer",
                "key_points": [],
                "sources": []
            }
            
            result = await multi_agent_executor._invoke_reviewer(
                task=task,
                writer_parsed=writer_parsed,
                locale="ko-KR",
                current_date="2024-01-01"
            )
            
            assert result["status"] == "success"
            assert result["ready_to_publish"] == True
    
    def test_normalize_tasks(self, multi_agent_executor):
        """Test task normalization"""
        # Test with list of strings
        tasks = multi_agent_executor._normalize_tasks(
            question="Test question",
            sub_topics=["Topic 1", "Topic 2"],
            sub_topic_contexts='{"Topic 1": "Context 1", "Topic 2": "Context 2"}',
            contexts="General context",
            max_tokens=8000
        )
        
        assert len(tasks) == 2
        assert tasks[0].sub_topic == "Topic 1"
        assert tasks[0].contexts == "Context 1"
        assert tasks[1].sub_topic == "Topic 2"
        assert tasks[1].contexts == "Context 2"
    
    def test_trim_context(self, multi_agent_executor):
        """Test context trimming"""
        long_context = "x" * 25000
        trimmed = multi_agent_executor._trim_context(long_context)
        
        assert len(trimmed) <= multi_agent_executor.context_max_chars + 20
        assert "[truncated]" in trimmed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
