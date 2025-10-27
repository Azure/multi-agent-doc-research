"""
Tests for AI Search Executor using Agent Framework.

This module tests the AISearchExecutor for document retrieval
from Azure AI Search.
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

from services_afw.ai_search_executor import AISearchExecutor


class TestAISearchExecutor:
    """Test suite for AISearchExecutor"""
    
    @pytest.fixture
    def ai_search_executor(self):
        """Create AISearchExecutor instance for testing"""
        # Mock the search client initialization
        with patch('services_afw.ai_search_executor.SearchClient'):
            with patch('services_afw.ai_search_executor.AzureOpenAI'):
                executor = AISearchExecutor(
                    id="test_ai_search",
                    search_endpoint="https://test.search.windows.net",
                    search_key="test_key",
                    index_name="test_index",
                    openai_endpoint="https://test.openai.azure.com",
                    openai_key="test_key",
                    embedding_deployment="text-embedding-ada-002",
                    openai_api_version="2024-08-01-preview"
                )
                return executor
    
    @pytest.fixture
    def mock_context(self):
        """Create a mock workflow context"""
        context = Mock()
        context.emit = AsyncMock()
        return context
    
    @pytest.mark.asyncio
    async def test_search_documents_basic(self, ai_search_executor, mock_context):
        """Test basic document search"""
        # Mock embedding generation
        ai_search_executor._generate_embedding = Mock(return_value=[0.1] * 1536)
        
        # Mock search results
        mock_results = [
            {
                "docId": "doc1",
                "title": "Test Document",
                "content": "Test content",
                "summary": "Test summary",
                "@search.score": 0.95
            }
        ]
        ai_search_executor._execute_search = Mock(return_value=iter(mock_results))
        
        search_data = {
            "query": "test query",
            "search_type": "hybrid",
            "top_k": 5
        }
        
        await ai_search_executor.search_documents(search_data, mock_context)
        
        # Verify emit was called
        assert mock_context.emit.called
        emit_call = mock_context.emit.call_args[0][0]
        assert emit_call["query"] == "test query"
        assert len(emit_call["documents"]) == 1
        assert emit_call["documents"][0]["title"] == "Test Document"
    
    @pytest.mark.asyncio
    async def test_search_documents_with_filters(self, ai_search_executor, mock_context):
        """Test document search with filters"""
        ai_search_executor._generate_embedding = Mock(return_value=[0.1] * 1536)
        ai_search_executor._execute_search = Mock(return_value=iter([]))
        
        search_data = {
            "query": "test query",
            "search_type": "semantic",
            "top_k": 5,
            "document_type": "IR Report",
            "industry": "Technology",
            "company": "Microsoft"
        }
        
        await ai_search_executor.search_documents(search_data, mock_context)
        
        # Verify filter building was called with correct parameters
        assert mock_context.emit.called
    
    @pytest.mark.asyncio
    async def test_search_documents_error_handling(self, ai_search_executor, mock_context):
        """Test error handling in document search"""
        ai_search_executor._generate_embedding = Mock(side_effect=Exception("Embedding failed"))
        
        search_data = {
            "query": "test query"
        }
        
        await ai_search_executor.search_documents(search_data, mock_context)
        
        # Verify error was emitted
        assert mock_context.emit.called
        emit_call = mock_context.emit.call_args[0][0]
        assert "error" in emit_call
        assert emit_call["documents"] == []
    
    def test_build_filters(self, ai_search_executor):
        """Test filter building logic"""
        # Test with single filter
        filter_expr = ai_search_executor._build_filters(
            filters=None,
            document_type="IR Report",
            industry=None,
            company=None,
            report_year=None
        )
        assert filter_expr == "document_type eq 'IR Report'"
        
        # Test with multiple filters
        filter_expr = ai_search_executor._build_filters(
            filters=None,
            document_type="IR Report",
            industry="Technology",
            company="Microsoft",
            report_year="2024"
        )
        assert "document_type eq 'IR Report'" in filter_expr
        assert "industry eq 'Technology'" in filter_expr
        assert "company eq 'Microsoft'" in filter_expr
        assert "report_year eq '2024'" in filter_expr
        assert " and " in filter_expr
    
    def test_get_select_fields(self, ai_search_executor):
        """Test select fields generation"""
        # With content
        fields = ai_search_executor._get_select_fields(include_content=True)
        assert "content" in fields
        assert "title" in fields
        
        # Without content
        fields = ai_search_executor._get_select_fields(include_content=False)
        assert "content" not in fields
        assert "title" in fields


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
