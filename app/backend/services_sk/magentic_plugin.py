"""Magentic Plugin using Semantic Kernel 1.37.0 orchestration pattern.

Implements a research workflow using Magentic Orchestration with:
- MagenticOrchestration with StandardMagenticManager
- InProcessRuntime
- Sequential agent collaboration (lead_researcher -> credibility_critic -> citation_agent -> report_writer)
- o3-mini model for manager reasoning
- Real-time progress streaming via callbacks

Compatible with existing caller signature:
  question, contexts, locale, max_tokens, current_date

Returns a JSON string with final report or yields progress updates.
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional, AsyncGenerator, Callable, Any
import asyncio

from semantic_kernel.functions import kernel_function
from semantic_kernel.agents import (
    ChatCompletionAgent,
    MagenticOrchestration,
    StandardMagenticManager,
)
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.contents import ChatMessageContent
from langchain.prompts import load_prompt
from config.config import Settings
from utils.json_control import clean_and_validate_json

logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load prompts
LEAD_RESEARCHER_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "magentic_lead_researcher_prompt.yaml"),
    encoding="utf-8",
)
CREDIBILITY_CRITIC_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "magentic_credibility_critic_prompt.yaml"),
    encoding="utf-8",
)
CITATION_AGENT_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "magentic_citation_agent_prompt.yaml"),
    encoding="utf-8",
)
REPORT_WRITER_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "magentic_report_writer_prompt.yaml"),
    encoding="utf-8",
)
MANAGER_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "magentic_manager_prompt.yaml"),
    encoding="utf-8",
)
FINAL_ANSWER_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "magentic_final_answer_prompt.yaml"),
    encoding="utf-8",
)


class MagenticPlugin:
    def __init__(self, settings: Settings):
        self.settings = settings
        # Chat completion service for agents
        self.chat_service = AzureChatCompletion(
            deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            api_key=settings.AZURE_OPENAI_API_KEY,
            endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )
        
        # Manager service with o3-mini reasoning model
        self.manager_service = AzureChatCompletion(
            deployment_name=settings.AZURE_OPENAI_REASONING_DEPLOYMENT_NAME or settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            api_key=settings.AZURE_OPENAI_API_KEY,
            endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )
        
        # Progress tracking
        self.progress_queue: Optional[asyncio.Queue] = None
        
        # Singleton agents - created once and reused
        self._agents_created = False
        self._lead_researcher: Optional[ChatCompletionAgent] = None
        self._credibility_critic: Optional[ChatCompletionAgent] = None
        self._citation_agent: Optional[ChatCompletionAgent] = None
        self._report_writer: Optional[ChatCompletionAgent] = None
        self._orchestration: Optional[MagenticOrchestration] = None
        self._runtime: Optional[InProcessRuntime] = None

    def _create_agents_once(
        self,
        question: str,
        contexts: str,
        locale: str,
        current_date: str,
    ) -> None:
        """
        Create agents only once (singleton pattern).
        Reuses agents across multiple sub-topic invocations.
        """
        if self._agents_created:
            logger.info("♻️ Reusing existing agents (singleton pattern)")
            return
        
        logger.info("� Creating research agents (first time only)...")
        
        # Create agents with generic instructions (context will be passed per invocation)
        self._lead_researcher = ChatCompletionAgent(
            service=self.chat_service,
            name="LeadResearcher",
            description="Advanced lead researcher that coordinates multiple internal research agents for comprehensive research tasks using ONLY internal document repositories.",
            instructions=LEAD_RESEARCHER_PROMPT.format(
                question="{question}",  # Placeholder, will be set per invocation
                context="{context}",
                locale=locale,
                current_date=current_date,
            ),
        )

        self._credibility_critic = ChatCompletionAgent(
            service=self.chat_service,
            name="CredibilityCritic",
            description="Analyzes credibility and coverage of internal search results.",
            instructions=CREDIBILITY_CRITIC_PROMPT.format(
                research_analysis="{research_analysis}",
                context="{context}",
                locale=locale,
                current_date=current_date,
            ),
        )

        self._citation_agent = ChatCompletionAgent(
            service=self.chat_service,
            name="CitationAgent",
            description="Processes research documents to identify citation locations.",
            instructions=CITATION_AGENT_PROMPT.format(
                research_content="{research_content}",
                context="{context}",
                locale=locale,
                current_date=current_date,
            ),
        )

        self._report_writer = ChatCompletionAgent(
            service=self.chat_service,
            name="ReportWriter",
            description="Creates structured markdown reports with citations.",
            instructions=REPORT_WRITER_PROMPT.format(
                cited_content="{cited_content}",
                credibility_assessment="{credibility_assessment}",
                locale=locale,
                current_date=current_date,
            ),
        )

        # Create reasoning settings for o3-mini model
        reasoning_high_settings = AzureChatPromptExecutionSettings(
            reasoning_effort="high"
        )

        # Create orchestration with StandardMagenticManager
        self._orchestration = MagenticOrchestration(
            members=[
                self._lead_researcher,
                self._credibility_critic,
                self._citation_agent,
                self._report_writer
            ],
            manager=StandardMagenticManager(
                chat_completion_service=self.manager_service,
                system_prompt=MANAGER_PROMPT.format(locale=locale),
                final_answer_prompt=FINAL_ANSWER_PROMPT.format(locale=locale),
                prompt_execution_settings=reasoning_high_settings,
            ),
            agent_response_callback=lambda msg: asyncio.create_task(
                self.progress_queue.put({"type": "agent_activity", "message": str(msg)})
            ) if self.progress_queue else None
        )

        # Create runtime
        self._runtime = InProcessRuntime()
        self._runtime.start()

        self._agents_created = True
        logger.info("✅ Agents and orchestration created successfully (singleton)")

    @kernel_function(
        description="Execute Magentic multi-agent research workflow with streaming.",
        name="magentic_flow_stream",
    )
    async def magentic_flow_stream(
        self,
        question: str,
        contexts: str,
        locale: str = "ko-KR",
        max_tokens: int = 8000,
        current_date: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Orchestrate a sequential research workflow using Magentic pattern with streaming.
        Yields progress updates as agents complete their work.
        Uses singleton pattern - agents are created once and reused.
        """
        if current_date is None:
            current_date = datetime.now().strftime("%Y-%m-%d")

        # Initialize progress queue
        self.progress_queue = asyncio.Queue()

        logger.info(
            f"🔬 Starting Magentic research flow (streaming) for question: {question[:100]}..."
        )

        try:
            # Create agents once (singleton pattern)
            if not self._agents_created:
                yield "data: ### 👥 Initializing research agents...\n\n"
                self._create_agents_once(question, contexts, locale, current_date)
                yield "data: ### ✅ Agents ready: LeadResearcher, CredibilityCritic, CitationAgent, ReportWriter\n\n"
            else:
                yield "data: ### ♻️ Reusing existing agents (singleton pattern)\n\n"

            yield "data: ### 🎯 Starting o3-mini orchestration...\n\n"

            # Prepare task
            task = f"Research Question: {question}\n\nContext: {contexts}"
            
            # Start orchestration in background
            orchestration_task = asyncio.create_task(
                self._orchestration.invoke(task=task, runtime=self._runtime)
            )

            # Monitor progress while orchestration runs
            last_agent = None
            while not orchestration_task.done():
                try:
                    # Check for progress updates with timeout
                    progress = await asyncio.wait_for(
                        self.progress_queue.get(), 
                        timeout=10.0
                    )
                    
                    # Yield progress to caller
                    if progress.get("type") == "agent_activity":
                        msg = progress.get("message", "")
                        # Extract agent name from message if possible
                        if "agent_name" in msg.lower():
                            try:
                                agent_info = json.loads(msg)
                                agent_name = agent_info.get("agent_name", "Unknown")
                                if agent_name != last_agent:
                                    yield f"data: ### 🤖 Agent Active: {agent_name}\n\n"
                                    last_agent = agent_name
                            except Exception:
                                pass
                    elif progress.get("agent"):
                        agent_name = progress.get("agent")
                        content_len = progress.get("content_length", 0)
                        if agent_name != last_agent:
                            yield f"data: ### 📝 {agent_name} is working... ({content_len} chars)\n\n"
                            last_agent = agent_name
                        
                except asyncio.TimeoutError:
                    # No progress update, continue waiting
                    continue
                except Exception as e:
                    logger.error(f"Error processing progress: {str(e)}")

            # Get final result
            result_proxy = await orchestration_task
            result = await result_proxy.get()

            # Extract final report
            if hasattr(result, 'content'):
                final_report = str(result.content)
            elif hasattr(result, 'value'):
                final_report = str(result.value)
            else:
                final_report = str(result)

            logger.info(f"✅ Magentic orchestration completed: {len(final_report)} chars")
            
            # Yield final report (without execution time here - orchestrator handles it)
            yield final_report

        except Exception as e:
            logger.error(f"❌ Error in magentic_flow_stream: {str(e)}", exc_info=True)
            yield f"data: ### ❌ Error: {str(e)}\n\n"
        finally:
            # Clean up progress queue
            self.progress_queue = None
    
    async def cleanup(self):
        """Clean up resources when orchestrator is done."""
        if self._runtime:
            await self._runtime.stop_when_idle()
            self._runtime = None
        self._agents_created = False
        logger.info("🧹 Magentic plugin cleanup completed")
