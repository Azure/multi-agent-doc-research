"""
Tests for YouTube MCP Executor using Agent Framework.

This module tests the YouTubeMCPExecutor for video search via MCP.
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

from services_afw.youtube_executor import YouTubeMCPExecutor


class TestYouTubeMCPExecutor:
    """Test suite for YouTubeMCPExecutor"""
    
    @pytest.fixture
    def youtube_executor(self):
        """Create YouTubeMCPExecutor instance for testing"""
        executor = YouTubeMCPExecutor(
            id="test_youtube",
            api_key="test_api_key",
            max_results=10
        )
        return executor
    
    @pytest.fixture
    def mock_context(self):
        """Create a mock workflow context"""
        context = Mock()
        context.emit = AsyncMock()
        return context
    
    @pytest.mark.asyncio
    async def test_search_youtube_success(self, youtube_executor, mock_context):
        """Test successful YouTube search"""
        # Mock the search function
        mock_result = {
            "status": "success",
            "videos": [
                {
                    "videoId": "test123",
                    "title": "Test Video",
                    "description": "Test description",
                    "channelTitle": "Test Channel",
                    "publishedAt": "2024-01-01"
                }
            ],
            "search_query": "test query",
            "total_results": 1
        }
        
        youtube_executor._search_youtube_videos = AsyncMock(return_value=mock_result)
        
        search_data = {
            "query": "test query",
            "max_results": 10
        }
        
        await youtube_executor.search_youtube(search_data, mock_context)
        
        # Verify emit was called
        assert mock_context.emit.called
        emit_call = mock_context.emit.call_args[0][0]
        assert emit_call["status"] == "success"
        assert len(emit_call["videos"]) == 1
        assert emit_call["videos"][0]["title"] == "Test Video"
    
    @pytest.mark.asyncio
    async def test_search_youtube_no_mcp(self, mock_context):
        """Test YouTube search when MCP is not available"""
        with patch('services_afw.youtube_executor.MCP_AVAILABLE', False):
            executor = YouTubeMCPExecutor(
                id="test_youtube",
                api_key="test_key",
                max_results=10
            )
            
            search_data = {
                "query": "test query"
            }
            
            await executor.search_youtube(search_data, mock_context)
            
            # Verify error was emitted
            assert mock_context.emit.called
            emit_call = mock_context.emit.call_args[0][0]
            assert emit_call["status"] == "error"
            assert "MCP library not available" in emit_call["message"]
    
    @pytest.mark.asyncio
    async def test_search_youtube_no_api_key(self, mock_context):
        """Test YouTube search without API key"""
        executor = YouTubeMCPExecutor(
            id="test_youtube",
            api_key=None,
            max_results=10
        )
        
        search_data = {
            "query": "test query"
        }
        
        await executor.search_youtube(search_data, mock_context)
        
        # Verify error was emitted
        assert mock_context.emit.called
        emit_call = mock_context.emit.call_args[0][0]
        assert emit_call["status"] == "error"
        assert "API key" in emit_call["message"]
    
    @pytest.mark.asyncio
    async def test_search_youtube_error_handling(self, youtube_executor, mock_context):
        """Test error handling in YouTube search"""
        youtube_executor._search_youtube_videos = AsyncMock(
            side_effect=Exception("Search failed")
        )
        
        search_data = {
            "query": "test query"
        }
        
        await youtube_executor.search_youtube(search_data, mock_context)
        
        # Verify error was emitted
        assert mock_context.emit.called
        emit_call = mock_context.emit.call_args[0][0]
        assert emit_call["status"] == "error"
    
    def test_format_results(self, youtube_executor):
        """Test result formatting"""
        videos = [
            {
                "videoId": "test123",
                "title": "Test Video",
                "description": "Test description",
                "channelTitle": "Test Channel",
                "publishedAt": "2024-01-01"
            }
        ]
        
        formatted = youtube_executor.format_results(videos)
        
        assert "Test Video" in formatted
        assert "Test Channel" in formatted
        assert "https://www.youtube.com/watch?v=test123" in formatted
    
    def test_format_results_empty(self, youtube_executor):
        """Test formatting empty results"""
        formatted = youtube_executor.format_results([])
        assert "No search results" in formatted
    
    def test_create_video_context(self, youtube_executor):
        """Test video context creation"""
        videos = [
            {
                "videoId": "test123",
                "title": "Test Video",
                "description": "Test description",
                "channelTitle": "Test Channel",
                "publishedAt": "2024-01-01"
            }
        ]
        
        context = youtube_executor.create_video_context(videos)
        
        assert "Test Video" in context
        assert "Test Channel" in context
        assert "https://www.youtube.com/watch?v=test123" in context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
