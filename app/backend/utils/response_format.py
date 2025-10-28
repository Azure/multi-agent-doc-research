"""
Standard response format definitions for multi-agent research executors.

This module defines the standard return format that all executors and plugins
should use to ensure consistency and avoid JSON parsing errors.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
import json


@dataclass
class SubTopicResult:
    """
    Standard format for individual sub-topic results.
    
    All executors/plugins should return results in this format to ensure
    consistent handling in orchestrators.
    """
    sub_topic: str
    status: str  # "success" | "error"
    answer_markdown: str  # Actual markdown answer (unescaped)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    reviewer_score: Optional[str] = "N/A"
    ready_to_publish: bool = False
    rounds_used: int = 0
    error: Optional[str] = None
    
    # Additional optional fields
    orchestration_rounds: Optional[int] = None
    writer_rounds: Optional[int] = None
    question: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict (JSON serializable)"""
        return asdict(self)


@dataclass
class MultiAgentResult:
    """
    Standard format for multi-agent execution results.
    
    This is the top-level result format that orchestrators expect.
    """
    status: str  # "success" | "error" | "partial_success"
    question: str
    sub_topic_results: List[Dict[str, Any]]  # List of SubTopicResult dicts
    all_ready_to_publish: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict"""
        return {
            "status": self.status,
            "question": self.question,
            "sub_topic_results": self.sub_topic_results,
            "all_ready_to_publish": self.all_ready_to_publish,
            "error": self.error,
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


def normalize_result(result: Any) -> Dict[str, Any]:
    """
    Normalize various result formats to standard Dict format.
    
    Handles:
    - Already a dict
    - JSON string
    - SubTopicResult dataclass
    
    Returns:
        Standardized dict with 'status', 'sub_topic', 'answer_markdown', etc.
    """
    if isinstance(result, dict):
        # Already a dict, ensure standard keys
        return {
            "status": result.get("status", "success"),
            "sub_topic": result.get("sub_topic", ""),
            "answer_markdown": result.get("answer_markdown", result.get("final_answer", "")),
            "citations": result.get("citations", []),
            "reviewer_score": result.get("reviewer_score", "N/A"),
            "ready_to_publish": result.get("ready_to_publish", False),
            "rounds_used": result.get("rounds_used", result.get("orchestration_rounds", 0)),
            "error": result.get("error"),
        }
    elif isinstance(result, str):
        # JSON string, parse it
        try:
            parsed = json.loads(result)
            return normalize_result(parsed)  # Recursive call with dict
        except json.JSONDecodeError:
            # Not JSON, treat as plain markdown
            return {
                "status": "success",
                "sub_topic": "Unknown",
                "answer_markdown": result,
                "citations": [],
                "reviewer_score": "N/A",
                "ready_to_publish": False,
                "rounds_used": 0,
                "error": None,
            }
    elif hasattr(result, 'to_dict'):
        # Dataclass with to_dict method
        return normalize_result(result.to_dict())
    else:
        # Unknown format, wrap it
        return {
            "status": "error",
            "sub_topic": "Unknown",
            "answer_markdown": str(result),
            "citations": [],
            "reviewer_score": "N/A",
            "ready_to_publish": False,
            "rounds_used": 0,
            "error": "Unknown result format",
        }
