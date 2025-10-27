"""
Microsoft Agent Framework based services for document inquiry chatbot.

This package contains the Agent Framework implementation of the chatbot services,
migrated from Semantic Kernel (services_sk).

Main Components:
- PlanSearchOrchestrator: Main orchestrator using AFW workflows
- IntentAnalyzerExecutor: Analyzes user intent
- SearchPlannerExecutor: Creates search plans
- ResponseGeneratorExecutor: Generates final responses
- WebSearchExecutor: Performs web search
- WebSearchHelper: Core search functionality
- GroupChattingExecutor: Writer/reviewer collaboration
- AISearchExecutor: AI Search for uploaded documents
- GroundingExecutor: Bing Grounding API integration
- YouTubeMCPExecutor: YouTube search via MCP
- MultiAgentResearchExecutor: Multi-agent writer/reviewer research

Usage:
    from services_afw import PlanSearchOrchestrator
    from config.config import Settings
    
    settings = Settings()
    orchestrator = PlanSearchOrchestrator(settings)
"""

from services_afw.doc_research_orchestrator_afw import PlanSearchOrchestratorAFW
from services_afw.group_chatting_executor import GroupChattingExecutor
from services_afw.ai_search_executor import AISearchExecutor
from services_afw.grounding_executor import GroundingExecutor
from services_afw.youtube_executor import YouTubeMCPExecutor
from services_afw.web_search_executor import WebSearchExecutor

__all__ = [
    "PlanSearchOrchestratorAFW",
    "GroupChattingExecutor",
    "AISearchExecutor",
    "GroundingExecutor",
    "YouTubeMCPExecutor",
    "WebSearchExecutor",
]

__all__ = [
    "PlanSearchOrchestratorAFW",
    "GroupChattingExecutor",
    "AISearchExecutor",
    "GroundingExecutor",
    "YouTubeMCPExecutor",
    "WebSearchExecutor",
]
