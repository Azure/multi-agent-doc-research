"""Magentic Plugin using Semantic Kernel orchestration pattern.

Creates fresh agents for each invocation to ensure clean state isolation.
Uses MagenticOrchestration with StandardMagenticManager and o3-mini reasoning model.
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional, AsyncGenerator
import asyncio

from semantic_kernel.functions import kernel_function
from semantic_kernel.agents import (
    ChatCompletionAgent,
    MagenticOrchestration,
    StandardMagenticManager,
)
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from langchain.prompts import load_prompt
from config.config import Settings
from utils.json_control import clean_and_validate_json, clean_markdown_escapes

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
        
        self.progress_queue: Optional[asyncio.Queue] = None

    def _create_agents(
        self,
        question: str,
        contexts: str,
        locale: str,
        current_date: str,
    ) -> tuple[MagenticOrchestration, InProcessRuntime]:
        """Create fresh agents and orchestration for each invocation."""
        
        lead_researcher = ChatCompletionAgent(
            service=self.chat_service,
            name="LeadResearcher",
            description="Advanced lead researcher that coordinates multiple internal research agents for comprehensive research tasks using ONLY internal document repositories.",
            instructions=LEAD_RESEARCHER_PROMPT.format(
                question=question,
                context=contexts,
                locale=locale,
                current_date=current_date,
            ),
        )

        credibility_critic = ChatCompletionAgent(
            service=self.chat_service,
            name="CredibilityCritic",
            description="Analyzes credibility and coverage of internal search results.",
            instructions=CREDIBILITY_CRITIC_PROMPT.format(
                research_analysis="{research_analysis}",
                context=contexts,
                locale=locale,
                current_date=current_date,
            ),
        )

        citation_agent = ChatCompletionAgent(
            service=self.chat_service,
            name="CitationAgent",
            description="Processes research documents to identify citation locations.",
            instructions=CITATION_AGENT_PROMPT.format(
                research_content="{research_content}",
                context=contexts,
                locale=locale,
                current_date=current_date,
            ),
        )

        report_writer = ChatCompletionAgent(
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

        reasoning_level_settings = AzureChatPromptExecutionSettings(
            #TODO adjust settings as needed
            #TODO move as configurable settings
            reasoning_effort="medium"
        )

        orchestration = MagenticOrchestration(
            members=[
                lead_researcher,
                credibility_critic,
                citation_agent,
                report_writer
            ],
            manager=StandardMagenticManager(
                chat_completion_service=self.manager_service,
                system_prompt=MANAGER_PROMPT.format(locale=locale),
                final_answer_prompt=FINAL_ANSWER_PROMPT.format(locale=locale),
                prompt_execution_settings=reasoning_level_settings,
            ),
            agent_response_callback=lambda msg: asyncio.create_task(
                self.progress_queue.put({"type": "agent_activity", "message": str(msg)})
            ) if self.progress_queue else None
        )

        runtime = InProcessRuntime()
        runtime.start()

        return orchestration, runtime

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
        if current_date is None:
            current_date = datetime.now().strftime("%Y-%m-%d")

        self.progress_queue = asyncio.Queue()
        runtime = None  # ✅ Initialize outside try block

        logger.info(f"Starting Magentic research flow for question: {question[:100]}...")

        try:
            # # Extract sub-topic from question if present
            # sub_topic_title = None
            # if question.startswith("Sub-topic:"):
            #     lines = question.split("\n")
            #     if len(lines) > 0:
            #         sub_topic_line = lines[0].replace("Sub-topic:", "").strip()
            #         if sub_topic_line:
            #             sub_topic_title = sub_topic_line
    
            # # Yield sub-topic header if found
            # if sub_topic_title:
            #     yield f"\n## 🎯 {sub_topic_title}\n\n"
            
            yield "data: ### 👥 Initializing research agents...\n\n"

            orchestration, runtime = self._create_agents(question, contexts, locale, current_date)

            yield "data: ### ✅ Agents ready: LeadResearcher, CredibilityCritic, CitationAgent, ReportWriter\n\n"
            yield "data: ### 🎯 Starting reasoning orchestration...\n\n"

            # ✅ TTFT 마커 전송 (첫 의미있는 출력)
            yield "data: __TTFT_MARKER__\n\n"

            task = f"Research Question: {question}\n\nContext: {contexts}"

            # ✅ 진행 상황 모니터링 제거하고 직접 대기
            result_proxy = await orchestration.invoke(task=task, runtime=runtime)
            result = await result_proxy.get()

            if hasattr(result, 'content'):
                final_report = str(result.content)
            elif hasattr(result, 'value'):
                final_report = str(result.value)
            else:
                final_report = str(result)

            if final_report.strip().startswith('{'):
                parsed = clean_and_validate_json(final_report, return_dict=True)
                final_report = (
                    parsed.get('revised_answer_markdown', '') or
                    parsed.get('draft_answer_markdown', '') or
                    parsed.get('final_answer', '') or
                    parsed.get('answer', '') or
                    str(parsed)
                )

            # ✅ 항상 마지막에 clean (JSON이든 아니든)
            final_report = clean_markdown_escapes(final_report)
           
            logger.info(f"Magentic orchestration completed: {len(final_report)} chars")
            
            yield final_report

        except Exception as e:
            logger.error(f"Error in magentic_flow_stream: {str(e)}", exc_info=True)
            yield f"data: ### ❌ Error: {str(e)}\n\n"
        finally:
            self.progress_queue = None
            if runtime:  # ✅ Now safely defined
                try:
                    await runtime.stop_when_idle()
                    logger.info("Runtime cleanup completed")
                except Exception as cleanup_error:
                    logger.error(f"Error during runtime cleanup: {cleanup_error}")

