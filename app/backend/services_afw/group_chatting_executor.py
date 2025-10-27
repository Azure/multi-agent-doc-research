"""Group Chatting Executor using Microsoft Agent Framework.

Implements a writer/reviewer loop per sub-topic using Agent Framework patterns:
- ChatAgent for Writer and Reviewer roles
- Custom termination strategy (approval-based)
- Sequential turn-based conversation per sub-topic
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
import asyncio
from agent_framework import (
    ChatAgent,
    ChatMessage,
    TextContent,
    Role,
    Executor,
    WorkflowContext,
    handler,
)
from agent_framework.azure import AzureOpenAIChatClient
from langchain.prompts import load_prompt
from config.config import Settings
from i18n.locale_msg import LOCALE_MESSAGES
from utils.json_control import clean_and_validate_json, clean_duplicate_table_content

logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load prompt templates (reuse existing SK prompts)
RESEARCH_WRITER_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "research_writer_prompt.yaml"),
    encoding="utf-8",
)
RESEARCH_REVIEWER_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "research_reviewer_prompt.yaml"),
    encoding="utf-8",
)


class ApprovalTerminationStrategy:
    """Termination strategy that stops when approval is given."""

    def __init__(self, max_iterations: int = 10):
        self.max_iterations = max_iterations
        self.iteration_count = 0

    def should_terminate(self, messages: List[ChatMessage]) -> bool:
        """Check if the conversation should terminate."""
        self.iteration_count += 1

        # Terminate if max iterations reached
        if self.iteration_count >= self.max_iterations:
            logger.info(f"Reached maximum iterations ({self.max_iterations})")
            return True

        # Terminate if last message contains approval or final answer
        if messages and len(messages) > 0:
            last_message = messages[-1]
            if last_message.text:
                text_lower = last_message.text.lower()
                # Check for approval keywords
                if any(
                    keyword in text_lower
                    for keyword in ["approved", "approval granted", "final answer"]
                ):
                    logger.info(
                        f"Approval detected in iteration {self.iteration_count}"
                    )
                    return True

        return False

    def reset(self):
        """Reset the iteration counter."""
        self.iteration_count = 0


class GroupChattingExecutor(Executor):
    """
    Agent Framework Executor for group chatting (writer/reviewer collaboration).

    Processes multiple sub-topics, running writer/reviewer loops for each,
    and returns aggregated results.
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
            f"GroupChattingExecutor initialized with context_max_chars={context_max_chars}, max_document_length={max_document_length}"
        )

    @handler
    async def run_group_chat(
        self, research_data: Dict[str, Any], ctx: WorkflowContext[Dict[str, Any], str]
    ) -> None:
        """Run group chat for each sub-topic with streaming progress."""
        try:
            # ✅ 먼저 executor_error 체크 (MagenticExecutor와 동일)
            executor_error = research_data.get("executor_error")
            if executor_error and executor_error.get("is_fatal"):
                logger.error(
                    f"[GroupChattingExecutor] Fatal error detected from upstream executor: {executor_error}"
                )
                # ✅ Yield error dict to orchestrator
                await ctx.yield_output(research_data)
                return
            
            # Extract parameters (flexible fallback chain)
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
                f"[GroupChattingExecutor] Starting group chat for {len(sub_topics)} sub-topics"
            )
            logger.info(f"[GroupChattingExecutor] Question: {question[:100]}...")

            # Get locale messages
            LOCALE_MSG = LOCALE_MESSAGES.get(locale, LOCALE_MESSAGES["ko-KR"])

            # If no sub_topics, create default (MULTI_AGENT_FIX.md 패턴)
            if not sub_topics:
                logger.info(
                    "[GroupChattingExecutor] No sub_topics provided, creating default"
                )
                sub_topics = [{"sub_topic": "research report", "queries": [question]}]

            # Yield starting message
            await ctx.yield_output(
                f"data: ### {LOCALE_MSG.get('start_research', 'Starting research')}...\n\n"
            )

            all_results = []

            # Process each sub-topic
            for idx, sub_topic_data in enumerate(sub_topics, 1):
                sub_topic_name = sub_topic_data.get("sub_topic", f"Topic {idx}")
                sub_topic_queries = sub_topic_data.get("queries", [question])

                # Extract context for this specific sub-topic from all search sources
                context_parts = []

                # Web search contexts (SK-compatible format)
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
                    f"[GroupChattingExecutor] Sub-topic '{sub_topic_name}' has {len(context_parts)} context items ({len(context)} chars)"
                )

                # Yield sub-topic start
                await ctx.yield_output(
                    f"data: ### 📝 Writer phase for '{sub_topic_name}' [{idx}/{len(sub_topics)}]\n\n"
                )

                # Execute group chat for this sub-topic
                result = await self._execute_single_sub_topic(
                    sub_topic=sub_topic_name,
                    question=", ".join(sub_topic_queries),
                    sub_topic_contexts=context,
                    locale=locale,
                    max_tokens=max_tokens,
                    ctx=ctx,
                )

                sub_topic_name = result.get("sub_topic", f"Topic {idx}")
                status = result.get("status", "unknown")
                RESULT_MSG = LOCALE_MESSAGES.get(locale, LOCALE_MESSAGES["ko-KR"])

                if status == "success":
                    final_answer = result.get("final_answer", "")
                    reviewer_score = result.get("reviewer_score", "N/A")
                    ready_to_publish = result.get("ready_to_publish", False)

                    # ✅ Safety check
                    if not final_answer:
                        final_answer = "No answer generated"
                        logger.warning(
                            f"[GroupChattingExecutor] No answer for '{sub_topic_name}'"
                        )

                    ready_icon = "✅" if ready_to_publish else "⚠️"

                    # Yield sub-topic header with score
                    await ctx.yield_output(f"\n")
                    await ctx.yield_output(
                        f"data: ### {RESULT_MSG.get('write_research', 'Writing Answer')} for {sub_topic_name} {ready_icon} (Score: {reviewer_score})\n\n"
                    )
                    await ctx.yield_output(f"## {sub_topic_name}\n\n")

                    # ✅ Stream answer in chunks (SK style)
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

                # Yield completion for this sub-topic
                status = result.get("status", "unknown")
                if status == "success":
                    await ctx.yield_output(
                        f"data: ### ✅ Completed '{sub_topic_name}' [{idx}/{len(sub_topics)}]\n\n"
                    )
                else:
                    await ctx.yield_output(
                        f"data: ### ❌ Failed '{sub_topic_name}' [{idx}/{len(sub_topics)}]\n\n"
                    )

            # Yield overall completion
            success_count = sum(1 for r in all_results if r.get("status") == "success")
            await ctx.yield_output(
                f"data: ### ✅ Group chat completed: {success_count}/{len(all_results)} successful\n\n"
            )

        except Exception as e:
            error_msg = f"Group chat failed: {str(e)}"
            logger.error(f"[GroupChattingExecutor] {error_msg}")
            logger.exception(e)
            await ctx.yield_output(f"data: ### ❌ {error_msg}\n\n")

    async def _execute_single_sub_topic(
        self,
        sub_topic: str,
        question: str,
        sub_topic_contexts: str,
        locale: str,
        max_tokens: int,
        ctx: WorkflowContext[Dict[str, Any], str],
    ) -> Dict[str, Any]:
        """Execute writer/reviewer loop for a single sub-topic."""

        started_at = datetime.utcnow()
        current_date = started_at.strftime("%Y-%m-%d")
        max_rounds = 3  # Fixed for now

        # Context size guard
        MAX_CONTEXT_CHARS = 10000
        trimmed_context = sub_topic_contexts
        if len(trimmed_context) > MAX_CONTEXT_CHARS:
            trimmed_context = trimmed_context[:MAX_CONTEXT_CHARS] + "\n...[truncated]"

        # Build agent instructions
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

        # Create agents
        writer_agent = ChatAgent(
            chat_client=self.chat_client,
            name="Writer",
            instructions=writer_instructions,
        )

        reviewer_agent = ChatAgent(
            chat_client=self.chat_client,
            name="Reviewer",
            instructions=reviewer_instructions,
        )

        # Initialize termination
        termination = ApprovalTerminationStrategy(max_iterations=max_rounds)

        # Create task
        task = (
            f"Sub-topic: {sub_topic}\n"
            f"Question: {question}\n"
            f"Context: {trimmed_context}...\n\n"
            "Writer: Draft answer. Reviewer: Review and approve if satisfactory."
        )

        messages: List[ChatMessage] = [
            ChatMessage(role=Role.USER, contents=[TextContent(text=task)])
        ]

        conversation_history = []
        current_agent = writer_agent
        error_info = None
        final_answer = ""
        writer_count = 0
        reviewer_count = 0
        answer_markdown = ""
        reviewer_score = "N/A"
        ready_to_publish = False
        first_token_sent = False  # ✅ Track TTFT

        try:
            iteration = 0
            while not termination.should_terminate(messages):
                iteration += 1
                is_writer = current_agent == writer_agent

                if is_writer:
                    writer_count += 1
                    await ctx.yield_output(
                        f"data: ### ✅ Writer round {writer_count}: {sub_topic}\n\n"
                    )
                else:
                    reviewer_count += 1
                    if reviewer_count == 1:
                        await ctx.yield_output(
                            f"data: ### 🔍 Reviewer phase for '{sub_topic}'\n\n"
                        )
                    await ctx.yield_output(
                        f"data: ### 🔍 Reviewer round {reviewer_count}: {sub_topic}\n\n"
                    )

                # Get response from agent
                last_message_text = messages[-1].text if messages else task

                try:
                    # ✅ 직접 호출 (keepalive 없이)
                    response = await current_agent.run(last_message_text)
                except Exception as e:
                    logger.error(f"Agent {current_agent.name} failed: {e}")
                    raise

                # ✅ Send TTFT event on first response
                if not first_token_sent:
                    first_token_sent = True
                    await ctx.yield_output(f"data: __TTFT_MARKER__\n\n")

                # Record message
                response_message = ChatMessage(
                    role=Role.ASSISTANT,
                    contents=[TextContent(text=response.text)],
                    author_name=current_agent.name,
                )
                messages.append(response_message)

                conversation_history.append(
                    {
                        "agent": current_agent.name,
                        "content": response.text,
                    }
                )

                # Check for approval
                if not is_writer and "approved" in response.text.lower():
                    await ctx.yield_output(f"data: ✅ '{sub_topic}' approved!\n\n")

                # Switch agent
                current_agent = (
                    reviewer_agent if current_agent == writer_agent else writer_agent
                )

            # Extract final answer
            if messages and len(messages) > 1:
                final_answer = messages[-1].text

            # ✅ Parse JSON and extract answer_markdown
            answer_markdown = ""
            citations = []
            try:
                final_answer_cleaned = clean_and_validate_json(final_answer)

                # Parse JSON
                parsed_answer = json.loads(final_answer_cleaned)

                # Extract answer_markdown
                answer_markdown = parsed_answer.get(
                    "revised_answer_markdown", ""
                ) or parsed_answer.get("draft_answer_markdown", "")

                
                # ✅ Extract citations
                citations = parsed_answer.get("citations", [])

                # Also extract other useful fields
                reviewer_score = parsed_answer.get("reviewer_evaluation_score", "N/A")
                ready_to_publish = parsed_answer.get("ready_to_publish", False)

                logger.info(
                    f"[GroupChattingExecutor] Parsed answer for '{sub_topic}': {len(answer_markdown)} chars, score={reviewer_score}, ready={ready_to_publish}"
                )

            except json.JSONDecodeError as e:
                logger.warning(
                    f"[GroupChattingExecutor] Failed to parse JSON for '{sub_topic}': {e}"
                )
                logger.warning(
                    f"[GroupChattingExecutor] Raw response: {final_answer[:200]}..."
                )
                # Fallback: use raw response
                answer_markdown = final_answer
            except Exception as e:
                logger.error(
                    f"[GroupChattingExecutor] Unexpected error parsing answer: {e}"
                )
                answer_markdown = final_answer

        except Exception as e:
            error_info = str(e)
            logger.error(f"Error in group chat for '{sub_topic}': {e}")
            logger.exception(e)
            await ctx.yield_output(f"data: ❌ Error: {e}\n\n")

        return {
            "status": "success" if error_info is None else "error",
            "sub_topic": sub_topic,
            "question": question,
            "final_answer": answer_markdown,
            "citations": citations,
            "reviewer_score": reviewer_score if "reviewer_score" in locals() else "N/A",
            "ready_to_publish": ready_to_publish
            if "ready_to_publish" in locals()
            else False,
            "rounds_used": iteration,
            "writer_rounds": writer_count,
            "reviewer_rounds": reviewer_count,
            "error": error_info,
        }