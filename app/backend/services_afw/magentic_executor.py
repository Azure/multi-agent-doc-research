"""Magentic Executor using Microsoft Agent Framework.

Implements multi-agent research using the Magentic orchestration pattern:
- Intelligent orchestrator coordinates specialized agents
- ResearcherAgent for information synthesis
- WriterAgent for content generation
- Dynamic planning and adaptive execution
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
import asyncio
import sys
import io
from contextlib import redirect_stdout

from agent_framework import (
    ChatAgent,
    ChatMessage,
    TextContent,
    Role,
    Executor,
    WorkflowContext,
    handler,
    MagenticBuilder,
    MagenticCallbackEvent,
    MagenticOrchestratorMessageEvent,
    MagenticAgentDeltaEvent,
    MagenticAgentMessageEvent,
    MagenticFinalResultEvent,
    MagenticCallbackMode,
    WorkflowOutputEvent,
    HostedCodeInterpreterTool,
)
from agent_framework.azure import AzureOpenAIChatClient

from langchain.prompts import load_prompt
from config.config import Settings
from i18n.locale_msg import LOCALE_MESSAGES
from utils.yield_message import send_step_with_code
from utils.json_control import clean_and_validate_json, clean_duplicate_table_content

logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load prompt templates
RESEARCH_ANALYST_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "research_analyst_prompt.yaml"),
    encoding="utf-8",
)
RESEARCH_WRITER_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "research_writer_prompt.yaml"),
    encoding="utf-8",
)
RESEARCH_REVIEWER_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "research_reviewer_prompt.yaml"),
    encoding="utf-8",
)


class MagenticExecutor(Executor):
    """
    Agent Framework Executor for Magentic orchestration pattern.

    Uses intelligent orchestration with a manager agent that coordinates
    three specialized agents to process research tasks:
    - ResearchAnalyst: Analyzes and synthesizes information from contexts
    - ResearchWriter: Creates comprehensive, well-structured content
    - ResearchReviewer: Validates quality, accuracy, and citation integrity

    This executor processes multiple sub-topics using Magentic pattern,
    providing intelligent task decomposition and adaptive execution with
    built-in quality assurance through the reviewer agent.
    """

    def __init__(
        self,
        id: str,
        chat_client: AzureOpenAIChatClient,
        settings: Settings,
        context_max_chars: int = 400000,  # SK와 동일: 40만자
        max_document_length: int = 10000,  # 문서당 1만자
        writer_parallel_limit: int = 4,
    ):
        super().__init__(id=id)
        self.chat_client = chat_client
        self.settings = settings
        self.context_max_chars = context_max_chars
        self.max_document_length = max_document_length

        logger.info(
            f"MagenticExecutor initialized with context_max_chars={context_max_chars}, max_document_length={max_document_length}"
        )

    @handler
    async def run_magentic_research(
        self, research_data: Dict[str, Any], ctx: WorkflowContext[Dict[str, Any], str]
    ) -> None:
        """Run Magentic orchestration for each sub-topic with streaming progress."""
        try:
            executor_error = research_data.get("executor_error")
            if executor_error and executor_error.get("is_fatal"):
                logger.error(
                    f"[MagenticExecutor] Fatal error detected from upstream executor: {executor_error}"
                )
                
                await ctx.yield_output(research_data)  
                return
            
            # Extract parameters
            question = (
                research_data.get("question")
                or research_data.get("enriched_query")
                or research_data.get("original_query")
                or ""
            )
            sub_topics = research_data.get("sub_topics", [])

            # Get metadata
            metadata = research_data.get("metadata", {})
            locale = metadata.get("locale", "ko-KR")
            verbose = metadata.get("verbose", False)
            max_tokens = metadata.get("max_tokens", 8000)

            logger.info(
                f"[MagenticExecutor] Starting Magentic research for {len(sub_topics)} sub-topics"
            )
            logger.info(f"[MagenticExecutor] Question: {question[:100]}...")

            # Get locale messages
            LOCALE_MSG = LOCALE_MESSAGES.get(locale, LOCALE_MESSAGES["ko-KR"])

            # If no sub_topics, create default
            if not sub_topics:
                logger.info(
                    "[MagenticExecutor] No sub_topics provided, creating default"
                )
                sub_topics = [{"sub_topic": "research report", "queries": [question]}]

            # ✅ Context 검증: AI Search가 필수인데 없으면 에러
            has_ai_search_context = bool(research_data.get("sub_topic_ai_search_contexts"))
            has_web_context = bool(research_data.get("sub_topic_web_contexts"))
            has_youtube_context = bool(research_data.get("sub_topic_youtube_contexts"))
            
            # ✅ 최소 1개의 context source가 있어야 함
            if not (has_ai_search_context or has_web_context or has_youtube_context):
                error_msg = "No context available for research (AI Search, Web Search, and YouTube all failed or returned empty)"
                logger.error(f"[MagenticExecutor] {error_msg}")
                
                await ctx.send_message(
                    {
                        **research_data,
                        "executor_error": {
                            "executor": "magentic_research",
                            "error_type": "no_context_available",
                            "error_message": error_msg,
                            "is_fatal": True,
                        },
                    }
                )
                return

            # Yield starting message
            await ctx.yield_output(
                f"data: ### {LOCALE_MSG.get('start_research', 'Starting Magentic research')}...\n\n"
            )

            all_results = []

            # Process each sub-topic with Magentic orchestration
            for idx, sub_topic_data in enumerate(sub_topics, 1):
                sub_topic_name = sub_topic_data.get("sub_topic", f"Topic {idx}")
                sub_topic_queries = sub_topic_data.get("queries", [question])

                # Extract context for this specific sub-topic from all search sources
                context_parts = []

                # Web search contexts
                sub_topic_web_contexts = research_data.get("sub_topic_web_contexts", {})
                if (
                    isinstance(sub_topic_web_contexts, dict)
                    and sub_topic_name in sub_topic_web_contexts
                ):
                    web_results = sub_topic_web_contexts[sub_topic_name]
                    if isinstance(web_results, list):
                        for result in web_results:
                            if isinstance(result, dict) and result.get("results"):
                                for item in result["results"]:
                                    title = item.get("title", "")
                                    snippet = item.get("snippet", "")
                                    url = item.get("url", "")
                                    context_parts.append(
                                        f"[Web] {title}\n{snippet}\nSource: {url}\n"
                                    )
                    elif isinstance(web_results, str):
                        context_parts.append(f"[Web Search Results]\n{web_results}\n")

                # YouTube contexts
                sub_topic_youtube_contexts = research_data.get(
                    "sub_topic_youtube_contexts", {}
                )
                if (
                    isinstance(sub_topic_youtube_contexts, dict)
                    and sub_topic_name in sub_topic_youtube_contexts
                ):
                    ytb_result = sub_topic_youtube_contexts[sub_topic_name]
                    if isinstance(ytb_result, dict) and ytb_result.get("videos"):
                        for video in ytb_result["videos"]:
                            title = video.get("title", "")
                            description = video.get("description", "")
                            url = video.get("url", "")
                            context_parts.append(
                                f"[YouTube] {title}\n{description}\nSource: {url}\n"
                            )

                # AI Search contexts
                sub_topic_ai_search_contexts = research_data.get(
                    "sub_topic_ai_search_contexts", {}
                )
                if (
                    isinstance(sub_topic_ai_search_contexts, dict)
                    and sub_topic_name in sub_topic_ai_search_contexts
                ):
                    ai_result = sub_topic_ai_search_contexts[sub_topic_name]
                    if isinstance(ai_result, dict) and ai_result.get("documents"):
                        for doc in ai_result["documents"]:
                            title = doc.get("title", "")
                            summary = doc.get("summary", "")
                            content = doc.get("content", "")

                            # Prefer summary, fallback to truncated content
                            if summary:
                                text_content = summary
                            elif content:
                                # Truncate to max_document_length
                                max_len = self.max_document_length
                                text_content = (
                                    content[:max_len] + "..."
                                    if len(content) > max_len
                                    else content
                                )
                            else:
                                text_content = ""

                            file_name = doc.get("file_name", "")
                            page_num = doc.get("page_number", "")
                            source_info = f"{file_name}" + (
                                f" (p.{page_num})" if page_num else ""
                            )

                            context_parts.append(
                                f"[AI Search] {title}\n{text_content}\nSource: {source_info}\n"
                            )

                # Combine all contexts for this sub-topic
                context = "\n".join(context_parts) if context_parts else ""

                logger.info(
                    f"[MagenticExecutor] Sub-topic '{sub_topic_name}' has {len(context_parts)} context items ({len(context)} chars)"
                )

                # Yield sub-topic start
                await ctx.yield_output(
                    f"data: ### 🎯 Magentic orchestration for '{sub_topic_name}' [{idx}/{len(sub_topics)}]\n\n"
                )

                # Execute Magentic orchestration for this sub-topic
                result = await self._execute_magentic_sub_topic(
                    sub_topic=sub_topic_name,
                    question=", ".join(sub_topic_queries),
                    sub_topic_contexts=context,
                    locale=locale,
                    max_tokens=max_tokens,
                    verbose=verbose,
                    ctx=ctx,
                )

                sub_topic_name = result.get("sub_topic", f"Topic {idx}")
                status = result.get("status", "unknown")
                RESULT_MSG = LOCALE_MESSAGES.get(locale, LOCALE_MESSAGES["ko-KR"])
                if status == "success":
                    await ctx.yield_output(
                        f"data: ### ✅ Completed '{sub_topic_name}' [{idx}/{len(sub_topics)}]\n\n"
                    )

                    final_answer = result.get("final_answer", "")
                    orchestration_rounds = result.get("orchestration_rounds", "N/A")
                    reviewer_score = result.get("reviewer_score", "N/A")
                    ready_to_publish = result.get("ready_to_publish", False)

                    # ✅ Safety check
                    if not final_answer:
                        final_answer = "No answer generated"
                        logger.warning(
                            f"[MagenticExecutor] No answer for '{sub_topic_name}'"
                        )

                    # Quality indicator emoji
                    quality_icon = "✅" if ready_to_publish else "⚠️"

                    # Yield sub-topic header with quality indicators
                    await ctx.yield_output(f"\n")
                    await ctx.yield_output(
                        f"data: ### {RESULT_MSG.get('write_research', 'Writing Answer')} for {sub_topic_name} "
                        f"{quality_icon} (Score: {reviewer_score}/5, Executed Rounds: {orchestration_rounds})\n\n"
                    )
                    await ctx.yield_output(f"## {sub_topic_name}\n\n")

                    # ✅ Stream answer in chunks
                    chunk_size = 100
                    for i in range(0, len(final_answer), chunk_size):
                        chunk = final_answer[i : i + chunk_size]
                        await ctx.yield_output(chunk)
                        await asyncio.sleep(0.01)

                    await ctx.yield_output("\n\n")
                else:
                    # Failed sub-topic
                    error_msg = result.get("error", "Unknown error")
                    await ctx.yield_output(f"\n## {sub_topic_name} ❌\n")
                    await ctx.yield_output(f"Error: {error_msg}\n\n")

            logger.info("✅ Magentic orchestration streaming complete")

        except Exception as e:
            error_msg = f"Magentic orchestration failed: {str(e)}"
            logger.error(f"[MagenticExecutor] {error_msg}")
            logger.exception(e)
            await ctx.yield_output(f"data: ### ❌ {error_msg}\n\n")

    async def _execute_magentic_sub_topic(
        self,
        sub_topic: str,
        question: str,
        sub_topic_contexts: str,
        locale: str,
        max_tokens: int,
        ctx: WorkflowContext[Dict[str, Any], str],
        verbose: bool,
    ) -> Dict[str, Any]:
        """Execute Magentic orchestration for a single sub-topic."""

        started_at = datetime.utcnow()
        current_date = started_at.strftime("%Y-%m-%d")

        # ⭐ Improved context size management with dynamic sizing
        MAX_CONTEXT_CHARS = min(
            self.context_max_chars, 20000
        )  # Cap at 20k for Magentic
        trimmed_context = sub_topic_contexts
        context_truncated = False

        if len(trimmed_context) > MAX_CONTEXT_CHARS:
            # Smart truncation: keep beginning and end
            keep_size = MAX_CONTEXT_CHARS // 2
            trimmed_context = (
                trimmed_context[:keep_size]
                + f"\n\n[... {len(sub_topic_contexts) - MAX_CONTEXT_CHARS} chars truncated ...]\n\n"
                + trimmed_context[-keep_size:]
            )
            context_truncated = True
            logger.warning(
                f"[MagenticExecutor] Context truncated from {len(sub_topic_contexts)} to {len(trimmed_context)} chars"
            )

        # Build specialized agent instructions using loaded prompts
        analyst_instructions = RESEARCH_ANALYST_PROMPT.format(
            current_date=current_date,
            locale=locale,
            sub_topic=sub_topic,
            question=question,
            contexts=trimmed_context,
        )

        writer_instructions = RESEARCH_WRITER_PROMPT.format(
            current_date=current_date,
            locale=locale,
            sub_topic=sub_topic,
            question=question,
            contexts=trimmed_context,
            max_tokens=max_tokens,
        )

        reviewer_instructions = RESEARCH_REVIEWER_PROMPT.format(
            current_date=current_date,
            locale=locale,
            sub_topic=sub_topic,
            question=question,
            contexts=trimmed_context,
            max_tokens=max_tokens,
        )

        # Create specialized agents for Magentic pattern (3 agents)
        analyst_agent = ChatAgent(
            chat_client=self.chat_client,
            name="ResearchAnalyst",
            description="Specialist in information synthesis and research analysis",
            instructions=analyst_instructions,
        )

        writer_agent = ChatAgent(
            chat_client=self.chat_client,
            name="ResearchWriter",
            description="Professional writer specializing in comprehensive research reports",
            instructions=writer_instructions,
        )

        reviewer_agent = ChatAgent(
            chat_client=self.chat_client,
            name="ResearchReviewer",
            description="Quality assurance specialist reviewing research outputs for accuracy and completeness",
            instructions=reviewer_instructions,
        )

        # State for tracking streaming events
        orchestrator_messages = []
        agent_responses = []
        streaming_buffer = []
        error_info = None
        final_answer = ""
        orchestration_rounds = 0
        first_token_sent = False
        current_agent = None
        workflow_status = None

        citations = []
        key_findings = []
        reviewer_score = "N/A"
        ready_to_publish = False

        # ⭐ Use verbose flag from parameter
        VERBOSE_MODE = verbose

        # ⭐ Error callback for better error handling
        def on_exception(exception: Exception) -> None:
            nonlocal error_info
            error_info = str(exception)
            logger.exception(
                f"[MagenticExecutor] Magentic workflow exception for '{sub_topic}'",
                exc_info=exception,
            )

        # Callback to process Magentic events
        async def on_magentic_event(event: MagenticCallbackEvent) -> None:
            nonlocal \
                orchestrator_messages, \
                agent_responses, \
                streaming_buffer, \
                orchestration_rounds, \
                first_token_sent, \
                current_agent, \
                workflow_status
            nonlocal \
                final_answer, \
                citations, \
                key_findings, \
                reviewer_score, \
                ready_to_publish

            try:
                if isinstance(event, MagenticOrchestratorMessageEvent):
                    # ✅ Use getattr to safely get text property (like in reference code)
                    message_text = (
                        getattr(event.message, "text", "") if event.message else ""
                    )

                    # Track orchestrator planning messages (internal)
                    orchestrator_messages.append(
                        {"kind": event.kind, "text": message_text}
                    )
                    orchestration_rounds += 1

                    # ✅ Always show round indicator (compact)
                    await ctx.yield_output(
                        f"data: ### 🔄 Orchestration Planning Rounds {orchestration_rounds}\n\n"
                    )

                    # ✅ VERBOSE: Show orchestrator planning details
                    if VERBOSE_MODE and message_text:
                        planning_text = json.dumps(
                            message_text, ensure_ascii=False, indent=2
                        )
                        truncated = (
                            planning_text[:100] + "... [truncated for display]"
                            if len(planning_text) > 100
                            else planning_text
                        )
                        await ctx.yield_output(
                            f"data: {send_step_with_code('💭 Planning: ', truncated)}\n\n"
                        )

                elif isinstance(event, MagenticAgentDeltaEvent):
                    # Track which agent is currently speaking
                    if current_agent != event.agent_id:
                        current_agent = event.agent_id
                        # ✅ Always show agent start (compact with emoji)
                        agent_emoji = (
                            "🔬"
                            if "analyst" in event.agent_id
                            else ("✍️" if "writer" in event.agent_id else "✅")
                        )
                        agent_name = (
                            "ResearchAnalyst"
                            if "analyst" in event.agent_id
                            else (
                                "ResearchWriter"
                                if "writer" in event.agent_id
                                else "Reviewer"
                            )
                        )
                        await ctx.yield_output(
                            f"data: ### {agent_emoji} [{agent_name}] working...\n\n"
                        )

                    # Send TTFT marker only once
                    if not first_token_sent:
                        first_token_sent = True
                        await ctx.yield_output(f"data: __TTFT_MARKER__\n\n")

                    # ⭐ DO NOT stream raw JSON to user (just collect in buffer)
                    streaming_buffer.append(
                        {"agent_id": event.agent_id, "text": event.text}
                    )

                elif isinstance(event, MagenticAgentMessageEvent):
                    # Agent completed a full response
                    if event.message is not None:
                        # ✅ Use getattr to safely get text property (like in reference code)
                        agent_text = getattr(event.message, "text", "")

                        agent_responses.append(
                            {
                                "agent_id": event.agent_id,
                                "role": event.message.role.value
                                if hasattr(event.message, "role")
                                else "unknown",
                                "text": agent_text,
                            }
                        )

                        # Use Reviewer output if available
                        if "reviewer" in event.agent_id.lower() and agent_text:
                            logger.info(
                                f"[MagenticExecutor] Reviewer agent completed, parsing output..."
                            )

                            cleaned_json = clean_and_validate_json(agent_text)
                            reviewer_output = json.loads(cleaned_json)

                            final_answer = reviewer_output.get(
                                "revised_answer_markdown", ""
                            )
                            citations = reviewer_output.get("citations", [])
                            reviewer_score = reviewer_output.get(
                                "reviewer_evaluation_score", "N/A"
                            )
                            ready_to_publish = reviewer_output.get(
                                "ready_to_publish", False
                            )

                            logger.info(
                                f"[MagenticExecutor] Parsed Reviewer output: "
                                f"answer_len={len(final_answer)}, score={reviewer_score}, ready={ready_to_publish}"
                            )

                        # # ✅ Always show completion checkmark
                        if VERBOSE_MODE:
                            agent_emoji = (
                                "🔬"
                                if "analyst" in event.agent_id
                                else ("✍️" if "writer" in event.agent_id else "✅")
                            )
                            await ctx.yield_output(
                                f"data: ### {agent_emoji} Complete ✓ \n\n"
                            )

                        # ✅ VERBOSE: Show agent output preview
                        if VERBOSE_MODE and agent_text:
                            preview = json.dumps(
                                agent_text, ensure_ascii=False, indent=2
                            )
                            truncated = (
                                preview[:100] + "... [truncated for display]"
                                if len(preview) > 100
                                else preview
                            )
                            agent_name = (
                                "ResearchAnalyst"
                                if "analyst" in event.agent_id
                                else (
                                    "ResearchWriter"
                                    if "writer" in event.agent_id
                                    else "Reviewer"
                                )
                            )
                            await ctx.yield_output(
                                f"data: {send_step_with_code(f'[{agent_name}] Preview', truncated)}\n\n"
                            )

                elif isinstance(event, MagenticFinalResultEvent):
                    # ✅ MagenticFinalResultEvent에서 최종 결과 처리
                    await ctx.yield_output(f"data: ### ✨ Finalizing...\n\n")

                    if event.message is not None:
                        # ✅ Extract text from ChatMessage
                        final_text = getattr(event.message, "text", "")

                        if final_text:
                            logger.info(
                                f"[MagenticExecutor] MagenticFinalResultEvent received: {len(final_text)} chars"
                            )

                            # ⭐ Parse JSON from final result
                            try:
                                final_answer_cleaned = clean_and_validate_json(
                                    final_text
                                )
                                parsed_answer = json.loads(final_answer_cleaned)

                                # ✅ Prefer Reviewer output over Writer output

                                answer_markdown = (
                                    parsed_answer.get("revised_answer_markdown", "")
                                    or parsed_answer.get("draft_answer_markdown", "")
                                    or parsed_answer.get("final_answer_markdown", "")
                                    or parsed_answer.get("answer_markdown", "")
                                    or parsed_answer.get("final_answer", "")
                                    or parsed_answer.get("answer", "")
                                )
                                if answer_markdown:
                                    final_answer = answer_markdown

                                if reviewer_score == "N/A":
                                    reviewer_score = parsed_answer.get(
                                        "reviewer_evaluation_score", "N/A"
                                    )

                                if not citations:
                                    citations = parsed_answer.get("citations", [])
                                key_findings = parsed_answer.get("key_findings", [])
                                reviewer_score = parsed_answer.get(
                                    "reviewer_evaluation_score", "N/A"
                                )
                                ready_to_publish = parsed_answer.get(
                                    "ready_to_publish", False
                                )

                                # ✅ Use parsed markdown as final answer
                                if answer_markdown:
                                    final_answer = answer_markdown
                                    logger.info(
                                        f"[MagenticExecutor] Parsed final answer: "
                                        f"{len(final_answer)} chars, score={reviewer_score}, ready={ready_to_publish}"
                                    )
                                else:
                                    # Fallback to raw text if no markdown found
                                    final_answer = final_text
                                    logger.warning(
                                        f"[MagenticExecutor] No markdown found, using raw text"
                                    )

                            except json.JSONDecodeError as e:
                                logger.warning(
                                    f"[MagenticExecutor] Failed to parse JSON from final result: {e}"
                                )
                                logger.warning(
                                    f"[MagenticExecutor] Raw final text: {final_text[:200]}..."
                                )
                                # Fallback: use raw text as-is
                                final_answer = final_text
                            except Exception as e:
                                logger.error(
                                    f"[MagenticExecutor] Error parsing final result: {e}"
                                )
                                final_answer = final_text

            except Exception as e:
                logger.error(f"Error in Magentic callback: {e}")
                logger.exception(e)

        # Build the Magentic workflow with 3 agents
        try:
            logger.info(
                f"[MagenticExecutor] Building Magentic workflow (3-agent pattern) for '{sub_topic}'"
            )
            workflow = (
                MagenticBuilder()
                .participants(
                    analyst=analyst_agent, writer=writer_agent, reviewer=reviewer_agent
                )
                .on_event(on_magentic_event, mode=MagenticCallbackMode.STREAMING)
                .on_exception(on_exception)
                .with_standard_manager(
                    chat_client=self.chat_client,
                    max_round_count=6,
                    max_stall_count=2,
                    max_reset_count=1,
                )
                .build()
            )

            # ⭐ Create detailed task description
            task = f"""You are coordinating a research team to analyze: "{sub_topic}"

# User Question
{question}

# Available Information Sources
{trimmed_context}

{"⚠️ Note: Context was truncated due to size. Focus on key information." if context_truncated else ""}

# Team Workflow (3-Phase Collaboration)

## Phase 1: ResearchAnalyst
- Analyze available sources carefully
- Extract key insights, data points, and patterns
- Identify source attribution for each claim
- Output: Structured JSON with key_insights, data_points, source_summary

## Phase 2: ResearchWriter
- Use analyst's findings to create comprehensive answer
- Structure content with clear markdown formatting
- Include proper citations with URLs/file names from sources
- Output: JSON with draft_answer_markdown, citations, key_findings

## Phase 3: ResearchReviewer
- Validate accuracy and completeness of the draft
- Check citation quality and source attribution
- Assess readiness to publish (ready_to_publish: true/false)
- Provide revised answer if needed
- Output: JSON with revised_answer_markdown, citations, reviewer_evaluation_score (1-5), ready_to_publish (boolean)

# Quality Standards
- Answer MUST be in {locale} language
- Target length: Under {max_tokens} tokens for final answer
- Use markdown: headings (#), bold (**), bullet lists (-)
- Include 1-2 emoji if natural
- Provide clickable reference links from sources
- Focus specifically on "{sub_topic}" within the broader question

# Critical Requirements
- Output valid JSON that can be parsed by json.loads()
- Use double quotes for all JSON strings
- No trailing commas in arrays or objects
- Escape special characters properly in strings
- Base all claims on provided contexts only
- Final answer output must be JSON with final_answer_markdown, citations, reviewer_evaluation_score, ready_to_publish fields
"""

            logger.info(
                f"[MagenticExecutor] Starting Magentic orchestration for '{sub_topic}'"
            )

            # ⭐ Execute workflow - all processing happens in on_magentic_event callback
            async for event in workflow.run_stream(task):
                # WorkflowOutputEvent is also emitted but we process everything in callback
                if isinstance(event, WorkflowOutputEvent):
                    logger.debug(
                        f"[MagenticExecutor] WorkflowOutputEvent received (processed in callback)"
                    )

            # ✅ After workflow completes, check if we got results
            if not final_answer:
                if not error_info:
                    error_info = "No output received from Magentic workflow"
                logger.error(f"[MagenticExecutor] {error_info}")

        except Exception as e:
            if not error_info:
                error_info = str(e)
            logger.error(
                f"[MagenticExecutor] Error in Magentic orchestration for '{sub_topic}': {e}"
            )
            logger.exception(e)
            await ctx.yield_output(f"data: ❌ Error: {e}\n\n")

        return {
            "status": "success" if error_info is None else "error",
            "sub_topic": sub_topic,
            "question": question,
            "final_answer": final_answer,
            "citations": citations,
            "reviewer_score": reviewer_score,
            "ready_to_publish": ready_to_publish,
            "orchestration_rounds": orchestration_rounds,
            "orchestrator_messages": len(orchestrator_messages),
            "agent_responses": len(agent_responses),
            "workflow_status": str(workflow_status) if workflow_status else None,
            "context_truncated": context_truncated,
            "error": error_info,
        }

