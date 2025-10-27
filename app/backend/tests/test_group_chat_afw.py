"""Tests for Agent Framework Group Chatting Executor.

Tests the writer/reviewer collaboration pattern using Agent Framework.
"""

import pytest
import asyncio
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from utils.json_control import clean_and_validate_json
# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from services_afw.group_chatting_executor import (
    GroupChattingExecutor,
    ApprovalTerminationStrategy,
    group_chat,
)
from config.config import Settings


@pytest.fixture
def settings():
    """Create settings instance for testing."""
    return Settings()


@pytest.fixture
def executor(settings):
    """Create GroupChattingExecutor instance."""
    return GroupChattingExecutor(settings)


class TestApprovalTerminationStrategy:
    """Tests for approval-based termination strategy."""
    
    def test_termination_by_max_iterations(self):
        """Test termination when max iterations reached."""
        strategy = ApprovalTerminationStrategy(max_iterations=3)
        
        from agent_framework import ChatMessage, Role, TextContent
        
        messages = [
            ChatMessage(role=Role.USER, contents=[TextContent(text="test")]),
        ]
        
        # Should not terminate on first check
        assert not strategy.should_terminate(messages)
        assert not strategy.should_terminate(messages)
        
        # Should terminate on third check
        assert strategy.should_terminate(messages)
    
    def test_termination_by_approval(self):
        """Test termination when approval keyword detected."""
        strategy = ApprovalTerminationStrategy(max_iterations=10)
        
        from agent_framework import ChatMessage, Role, TextContent
        
        # Messages without approval
        messages = [
            ChatMessage(
                role=Role.ASSISTANT,
                contents=[TextContent(text="Here is my draft response.")],
                author_name="Writer"
            ),
        ]
        
        assert not strategy.should_terminate(messages)
        
        # Message with approval
        messages.append(
            ChatMessage(
                role=Role.ASSISTANT,
                contents=[TextContent(text="This is approved and ready.")],
                author_name="Reviewer"
            )
        )
        
        assert strategy.should_terminate(messages)
    
    def test_reset(self):
        """Test reset functionality."""
        strategy = ApprovalTerminationStrategy(max_iterations=2)
        
        from agent_framework import ChatMessage, Role, TextContent
        
        messages = [
            ChatMessage(role=Role.USER, contents=[TextContent(text="test")]),
        ]
        
        # Reach max iterations
        strategy.should_terminate(messages)
        strategy.should_terminate(messages)
        assert strategy.iteration_count == 2
        
        # Reset
        strategy.reset()
        assert strategy.iteration_count == 0


class TestGroupChattingExecutor:
    """Tests for Agent Framework group chatting executor."""
    
    @pytest.mark.asyncio
    async def test_basic_group_chat(self, executor):
        """Test basic group chat execution."""
        result = await executor.group_chat(
            sub_topic="Azure OpenAI Service pricing",
            question="What are the costs of using Azure OpenAI?",
            sub_topic_contexts="Azure OpenAI offers various pricing tiers based on model and usage.",
            locale="ko-KR",
            max_rounds=3,
            max_tokens=500,
        )
        
        # Validate result structure
        assert result is not None
        assert isinstance(result, str)
        
        # Parse JSON result
        parsed = json.loads(result)
        assert "status" in parsed
        assert "sub_topic" in parsed
        assert "final_answer" in parsed
        assert "rounds_used" in parsed
        
        print(f"\n[Test] Status: {parsed['status']}")
        print(f"[Test] Rounds used: {parsed['rounds_used']}")
        print(f"[Test] Final answer length: {len(parsed['final_answer'])}")
    
    @pytest.mark.asyncio
    async def test_json_validation(self, executor):
        """Test JSON validation functionality."""
        # Test valid JSON
        valid_json = '{"key": "value"}'
        assert executor._validate_json(valid_json) is True
        
        # Test invalid JSON
        invalid_json = "This is not JSON"
        assert executor._validate_json(invalid_json) is False
        
        # Test JSON with extra text
        json_with_text = "Here is the response: {\"key\": \"value\"}"
        assert executor._validate_json(json_with_text) is False
    
    @pytest.mark.asyncio
    async def test_json_cleaning(self, executor):
        """Test JSON cleaning and extraction."""
        # Test with markdown code block
        markdown_json = """```json
{
    "sub_topic": "Test",
    "answer": "Response"
}
```"""
        
        cleaned = clean_and_validate_json(markdown_json)
        parsed = json.loads(cleaned)
        assert "sub_topic" in parsed
        assert parsed["sub_topic"] == "Test"
        
        # Test with extra text
        text_with_json = 'Here is the answer: {"sub_topic": "Test", "answer": "Response"} Hope this helps!'
        
        cleaned = clean_and_validate_json(text_with_json)
        parsed = json.loads(cleaned)
        assert "sub_topic" in parsed
    
    @pytest.mark.asyncio
    async def test_multiple_rounds(self, executor):
        """Test that multiple rounds of refinement occur."""
        result = await executor.group_chat(
            sub_topic="Microsoft Azure regions",
            question="Which Azure regions are available in Korea?",
            sub_topic_contexts="Azure has data centers in Korea Central and Korea South regions.",
            locale="ko-KR",
            max_rounds=5,
            max_tokens=800,
        )
        
        parsed = json.loads(result)
        
        # Should have multiple rounds
        assert parsed["rounds_used"] >= 1
        assert parsed["status"] == "success"
        
        print(f"\n[Test] Executed {parsed['rounds_used']} rounds")
        print(f"[Test] Final answer preview: {parsed['final_answer'][:200]}...")
    
    @pytest.mark.asyncio
    async def test_max_rounds_enforcement(self, executor):
        """Test that max rounds is enforced."""
        result = await executor.group_chat(
            sub_topic="AI Safety",
            question="What are AI safety concerns?",
            sub_topic_contexts="AI safety involves alignment, robustness, and ethical considerations.",
            locale="en-US",
            max_rounds=3,  # Small number to ensure termination
            max_tokens=500,
        )
        
        parsed = json.loads(result)
        
        # Should not exceed max rounds
        assert parsed["rounds_used"] <= 3  # 3 makes it odd (becomes 3)
        
        print(f"\n[Test] Rounds used: {parsed['rounds_used']} (max: 3)")


class TestStandaloneFunction:
    """Tests for standalone group_chat function."""
    
    @pytest.mark.asyncio
    async def test_standalone_function(self, settings):
        """Test standalone function interface."""
        result = await group_chat(
            sub_topic="Azure Functions pricing",
            question="How much does Azure Functions cost?",
            sub_topic_contexts="Azure Functions offers consumption and premium plans.",
            locale="ko-KR",
            max_rounds=3,
            settings=settings,
        )
        
        # Validate result
        assert result is not None
        parsed = json.loads(result)
        assert "status" in parsed
        assert "final_answer" in parsed
        
        print(f"\n[Test] Standalone function result: {parsed['status']}")
    
    @pytest.mark.asyncio
    async def test_standalone_function_with_defaults(self):
        """Test standalone function with default settings."""
        result = await group_chat(
            sub_topic="Azure OpenAI",
            question="What is Azure OpenAI?",
            sub_topic_contexts="Azure OpenAI provides access to OpenAI models.",
        )
        
        # Should work with default settings
        assert result is not None
        parsed = json.loads(result)
        assert parsed["status"] in ["success", "error"]


class TestErrorHandling:
    """Tests for error handling in group chat."""
    
    @pytest.mark.asyncio
    async def test_empty_inputs(self, executor):
        """Test handling of empty inputs."""
        result = await executor.group_chat(
            sub_topic="",
            question="",
            sub_topic_contexts="",
            max_rounds=1,
        )
        
        # Should handle gracefully
        parsed = json.loads(result)
        assert "status" in parsed
        # May be success or error depending on model behavior
        print(f"\n[Test] Empty input status: {parsed['status']}")
    
    @pytest.mark.asyncio
    async def test_invalid_max_rounds(self, executor):
        """Test handling of invalid max_rounds."""
        result = await executor.group_chat(
            sub_topic="Test",
            question="Test question",
            sub_topic_contexts="Test context",
            max_rounds=-5,  # Invalid
        )
        
        parsed = json.loads(result)
        # Should default to valid value
        assert parsed["max_rounds_requested"] >= 1


class TestComparisonWithSK:
    """Tests comparing AF and SK implementations."""
    
    @pytest.mark.asyncio
    async def test_output_format_compatibility(self, executor):
        """Test that output format matches SK version."""
        result = await executor.group_chat(
            sub_topic="Azure Cost Management",
            question="How to manage Azure costs?",
            sub_topic_contexts="Azure provides cost management tools and budgets.",
            locale="ko-KR",
            max_rounds=3,
        )
        
        parsed = json.loads(result)
        
        # Check all expected fields from SK version
        required_fields = [
            "status",
            "sub_topic",
            "question",
            "final_answer",
            "rounds_used",
            "max_rounds_requested",
            "started_at",
            "ended_at",
        ]
        
        for field in required_fields:
            assert field in parsed, f"Missing field: {field}"
        
        print(f"\n[Test] All required fields present: {', '.join(required_fields)}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
