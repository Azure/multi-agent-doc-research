"""Multi-agent research plugin for orchestrating writer and reviewer workflows.

This plugin coordinates parallel writer drafts followed by sequential reviews
using Azure OpenAI completions and project-specific prompts with progress callbacks.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union, Awaitable

import pytz
from langchain.prompts import load_prompt
from openai import AsyncAzureOpenAI
from semantic_kernel.functions import kernel_function

from config.config import Settings
from utils.json_control import clean_and_validate_json

logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))

RESEARCH_WRITER_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "research_writer_prompt.yaml"),
    encoding="utf-8",
)
RESEARCH_REVIEWER_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "research_reviewer_prompt.yaml"),
    encoding="utf-8",
)


def _safe_int(value: Union[str, int, None], default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Union[str, bool, None], default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return default


def _safe_json_loads(payload: Optional[str]) -> Any:
    if not payload:
        return None
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return None


@dataclass
class SubTopicTask:
    sub_topic: str
    question: str
    contexts: str
    max_tokens: int


class MultiAgentPlugin:
    """Semantic Kernel plugin that runs writer and reviewer agents per sub-topic with progress callbacks."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncAzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        )
        self.search_type = settings.AZURE_AI_SEARCH_SEARCH_TYPE or "hybrid"
        self.deployment_name = settings.AZURE_OPENAI_DEPLOYMENT_NAME
        self.default_max_tokens = settings.MAX_TOKENS
        self.context_max_chars = _safe_int(
            os.getenv("MULTI_AGENT_CONTEXT_MAX_CHARS"), 20000
        )
        self.writer_parallel_limit = max(
            1,
            _safe_int(os.getenv("MULTI_AGENT_WRITER_CONCURRENCY"), 4),
        )
        if isinstance(settings.TIME_ZONE, str):
            try:
                self.timezone = pytz.timezone(settings.TIME_ZONE)
            except Exception:
                self.timezone = pytz.UTC
        else:
            self.timezone = pytz.UTC

    @kernel_function(
        description=(
            "Run multi-agent research workflow: writers draft answers per sub-topic "
            "in parallel, reviewers validate sequentially, returning publish-ready "
            "results when all reviews pass."
        ),
        name="run_multi_agent",
    )
    async def run_multi_agent(
        self,
        question: str,
        sub_topic: Optional[str] = None,
        sub_topics: Optional[str] = None,
        sub_topic_contexts: Optional[str] = None,
        contexts: Optional[str] = None,
        locale: str = "en-US",
        max_tokens: Union[str, int] = 8000,
        current_date: Optional[str] = None,
        parallel: Union[str, bool, None] = None,
        parallel_limit: Union[str, int, None] = None,
        max_rounds: Optional[Union[str, int]] = None,  # kept for compatibility
    ) -> str:
        """Execute writer/reviewer pipeline and return aggregated JSON string.
        
        Note: This is a synchronous kernel function wrapper. For streaming progress,
        use run_multi_agent_with_callback() directly from the executor.
        """

        normalized_max_tokens = _safe_int(max_tokens, self.default_max_tokens)
        parallel_writers = _safe_bool(parallel, True)
        effective_parallel_limit = max(
            1,
            _safe_int(parallel_limit, self.writer_parallel_limit),
        )
        current_date = self._normalize_current_date(current_date)

        tasks = self._normalize_tasks(
            question=question,
            sub_topic=sub_topic,
            sub_topics=sub_topics,
            sub_topic_contexts=sub_topic_contexts,
            contexts=contexts,
            max_tokens=normalized_max_tokens,
        )

        if not tasks:
            logger.warning("[MultiAgent] No valid sub-topics resolved")
            return json.dumps(
                {
                    "status": "error",
                    "message": "No valid sub-topics provided",
                    "question": question,
                    "current_date": current_date,
                },
                ensure_ascii=False,
            )

        # Run without callback for kernel function compatibility
        return await self.run_multi_agent_with_callback(
            question=question,
            tasks=tasks,
            locale=locale,
            max_tokens=normalized_max_tokens,
            current_date=current_date,
            parallel=parallel_writers,
            parallel_limit=effective_parallel_limit,
            progress_callback=None,
        )

    async def run_multi_agent_with_callback(
        self,
        question: str,
        tasks: List[SubTopicTask],
        locale: str,
        max_tokens: int,
        current_date: str,
        parallel: bool,
        parallel_limit: int,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
    ) -> str:
        """Execute writer/reviewer pipeline with optional progress callback.
        
        Args:
            question: User's question
            tasks: List of SubTopicTask to process
            locale: Locale for responses
            max_tokens: Maximum tokens per response
            current_date: Current date string
            parallel: Whether to run writers in parallel
            parallel_limit: Concurrency limit for parallel writers
            progress_callback: Optional async callback(event_type: str, data: dict)
                Event types: "phase_start", "writer_progress", "writer_complete", 
                            "reviewer_progress", "reviewer_complete", "final_answer"
        
        Returns:
            JSON string with final results
        """
        
        # Phase 1: Writers
        if progress_callback:
            await progress_callback("phase_start", {
                "phase": "writer",
                "total_tasks": len(tasks),
                "message": f"📝 Writer Phase - Processing {len(tasks)} sub-topic(s)"
            })
        
        writer_results = await self._run_writers(
            tasks=tasks,
            locale=locale,
            current_date=current_date,
            parallel=parallel,
            parallel_limit=parallel_limit,
            progress_callback=progress_callback,
        )

        # Phase 2: Reviewers
        if progress_callback:
            await progress_callback("phase_start", {
                "phase": "reviewer",
                "total_tasks": len(writer_results),
                "message": f"🔍 Reviewer Phase - Validating {len(writer_results)} draft(s)"
            })
        
        reviewer_results = await self._run_reviewers(
            writer_results=writer_results,
            locale=locale,
            current_date=current_date,
            progress_callback=progress_callback,
        )

        # Phase 3: Final aggregation
        all_ready = all(
            r["review"].get("ready_to_publish")
            for r in reviewer_results
            if r["review"]["status"] == "success"
        ) and all(r["review"]["status"] == "success" for r in reviewer_results)

        if progress_callback:
            await progress_callback("phase_start", {
                "phase": "final",
                "all_ready": all_ready,
                "message": "✅ All sub-topics ready to publish!" if all_ready else "⚠️ Review completed with some concerns"
            })

        final_payload: Dict[str, Any] = {
            "status": "success" if all_ready else "completed_with_concerns",
            "question": question,
            "current_date": current_date,
            "locale": locale,
            "all_ready_to_publish": all_ready,
            "sub_topic_results": reviewer_results,
        }

        # Always generate final answer after review completion
        final_answers = {}
        for item in reviewer_results:
            sub_topic = item["sub_topic"]
            
            # ✅ Extract citations safely from writer or reviewer parsed data
            citations = []
            if item["review"]["status"] == "success" and item["review"]["parsed"]:
                # Prefer citations from reviewer if available
                citations = item["review"]["parsed"].get("citations", [])
                if not citations and item["writer"]["parsed"]:
                    # Fallback to writer's citations
                    citations = item["writer"]["parsed"].get("citations", [])
            elif item["writer"]["parsed"]:
                citations = item["writer"]["parsed"].get("citations", [])
            
            # Use revised answer if available, otherwise use draft
            if item["review"]["status"] == "success" and item["review"]["parsed"]:
                revised_answer = item["review"]["parsed"].get("revised_answer_markdown", "")
                if not revised_answer:  # Fallback to draft if no revised answer
                    revised_answer = item["writer"]["parsed"].get("draft_answer_markdown", "")
            else:
                revised_answer = item["writer"]["parsed"].get("draft_answer_markdown", "")
            
            final_answers[sub_topic] = revised_answer
            
            # Send individual answers through callback
            if progress_callback:
                await progress_callback("final_answer", {
                    "sub_topic": sub_topic,
                    "answer": revised_answer,
                    "citations": citations,
                    "ready_to_publish": item["review"].get("ready_to_publish", False),
                    "score": item["review"]["parsed"].get("reviewer_evaluation_score", "N/A") if item["review"].get("parsed") else "N/A"
                })
        
        final_payload["final_answer"] = json.dumps(
            {
                "question": question,
                "answers": final_answers,
            },
            ensure_ascii=False,
        )

        return json.dumps(final_payload, ensure_ascii=False)

    async def run_multi_agent_streaming(
        self,
        question: str,
        tasks: List[SubTopicTask],
        locale: str,
        max_tokens: int,
        current_date: str,
        parallel: bool,
        parallel_limit: int,
    ):
        """Execute writer/reviewer pipeline with direct streaming (AsyncGenerator).
        
        This yields progress messages and final results directly.
        """
        
        # Phase 1: Writers
        yield f"data: 📝 Writer Phase - Processing {len(tasks)} sub-topic(s)\n\n"
        
        writer_results = []
        completed = 0
        
        if parallel and len(tasks) > 1:
            # Parallel execution
            semaphore = asyncio.Semaphore(parallel_limit)
            
            async def _wrapped(task: SubTopicTask):
                nonlocal completed
                async with semaphore:
                    result = await self._invoke_writer(task, locale, current_date)
                    completed += 1
                    return result
            
            results = await asyncio.gather(
                *[_wrapped(task) for task in tasks],
                return_exceptions=True,
            )
            
            for task, result in zip(tasks, results):
                if isinstance(result, Exception):
                    yield f"data: ❌ Writer failed: {task.sub_topic}\n\n"
                    writer_results.append({
                        "sub_topic": task.sub_topic,
                        "question": task.question,
                        "writer": {"status": "error", "error": str(result), "parsed": None},
                        "task": task,
                    })
                else:
                    status_icon = "✅" if result["writer"]["status"] == "success" else "❌"
                    yield f"data: {status_icon} Writer [{completed}/{len(tasks)}]: {task.sub_topic}\n\n"
                    writer_results.append({"sub_topic": task.sub_topic, **result, "task": task})
        else:
            # Sequential execution
            for task in tasks:
                result = await self._invoke_writer(task, locale, current_date)
                completed += 1
                status_icon = "✅" if result["writer"]["status"] == "success" else "❌"
                yield f"data: {status_icon} Writer [{completed}/{len(tasks)}]: {task.sub_topic}\n\n"
                writer_results.append({"sub_topic": task.sub_topic, **result, "task": task})

        # Phase 2: Reviewers
        yield f"data: 🔍 Reviewer Phase - Validating {len(writer_results)} draft(s)\n\n"
        
        reviewer_results = []
        completed = 0
        
        for item in writer_results:
            writer_info = item.get("writer", {})
            task = item.get("task")
            sub_topic = item.get("sub_topic")
            
            if writer_info.get("status") != "success" or not writer_info.get("parsed"):
                yield f"data: ⏭️  Reviewer [{completed + 1}/{len(writer_results)}]: {sub_topic} (skipped - writer failed)\n\n"
                reviewer_results.append({
                    "sub_topic": sub_topic,
                    "question": item.get("question"),
                    "writer": writer_info,
                    "review": {"status": "skipped", "reason": "Writer output unavailable", "parsed": {}, "ready_to_publish": False},
                })
                completed += 1
                continue

            review_result = await self._invoke_reviewer(task, writer_info["parsed"], locale, current_date)
            completed += 1
            
            status_icon = "✅" if review_result.get("ready_to_publish") else "⚠️"
            score = review_result.get("parsed", {}).get("reviewer_evaluation_score", "N/A")
            yield f"data: {status_icon} Reviewer [{completed}/{len(writer_results)}]: {sub_topic} (score: {score})\n\n"
            
            reviewer_results.append({
                "sub_topic": sub_topic,
                "question": item.get("question"),
                "writer": writer_info,
                "review": review_result,
            })

        # Phase 3: Stream final answers - ALWAYS stream after review completion
        all_ready = all(
            r["review"].get("ready_to_publish")
            for r in reviewer_results
            if r["review"]["status"] == "success"
        ) and all(r["review"]["status"] == "success" for r in reviewer_results)

        if all_ready:
            yield f"data: ✅ All sub-topics ready to publish!\n\n"
        else:
            yield f"data: ⚠️ Review completed with some concerns\n\n"
        
        # Stream all final answers regardless of ready_to_publish status
        for item in reviewer_results:
            sub_topic = item["sub_topic"]
            
            # ✅ Extract citations safely
            citations = []
            if item["review"]["status"] == "success" and item["review"]["parsed"]:
                citations = item["review"]["parsed"].get("citations", [])
                if not citations and item["writer"]["parsed"]:
                    citations = item["writer"]["parsed"].get("citations", [])
            elif item["writer"]["parsed"]:
                citations = item["writer"]["parsed"].get("citations", [])
            
            # Use revised answer if available, otherwise use draft
            if item["review"]["status"] == "success" and item["review"]["parsed"]:
                revised_answer = item["review"]["parsed"].get("revised_answer_markdown", "")
                if not revised_answer:
                    revised_answer = item["writer"]["parsed"].get("draft_answer_markdown", "")
            else:
                revised_answer = item["writer"]["parsed"].get("draft_answer_markdown", "")
            
            if revised_answer:
                ready_status = "✅" if item["review"].get("ready_to_publish", False) else "⚠️"
                score = item["review"]["parsed"].get("reviewer_evaluation_score", "N/A") if item["review"].get("parsed") else "N/A"
                
                yield f"\n"
                yield f"data: ### 📝 Answer for {sub_topic} {ready_status} (Score: {score})\n\n"
                yield f"## {sub_topic}\n\n"
                
                # Stream answer in chunks
                chunk_size = 100
                for i in range(0, len(revised_answer), chunk_size):
                    chunk = revised_answer[i:i+chunk_size]
                    yield chunk
                    await asyncio.sleep(0.01)
                yield "\n\n"

        # Return final JSON
        final_payload = {
            "status": "success" if all_ready else "pending_review",
            "question": question,
            "current_date": current_date,
            "locale": locale,
            "all_ready_to_publish": all_ready,
            "sub_topic_results": reviewer_results,
        }
        
        if all_ready:
            final_answers = {
                item["sub_topic"]: item["review"]["parsed"].get("revised_answer_markdown")
                for item in reviewer_results
            }
            final_payload["final_answer"] = json.dumps({"question": question, "answers": final_answers}, ensure_ascii=False)
        else:
            final_payload["final_answer"] = ""

        # Don't yield the JSON, just return it for logging
        logger.info(f"Multi-agent completed: {final_payload['status']}")

    async def _run_writers(
        self,
        tasks: List[SubTopicTask],
        locale: str,
        current_date: str,
        parallel: bool,
        parallel_limit: int,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
    ) -> List[Dict[str, Any]]:
        """Run writers with optional progress callbacks."""
        semaphore = asyncio.Semaphore(parallel_limit)
        completed_count = 0
        total_count = len(tasks)

        async def _wrapped(task: SubTopicTask) -> Dict[str, Any]:
            nonlocal completed_count
            async with semaphore:
                result = await self._invoke_writer(task, locale, current_date)
                completed_count += 1
                
                if progress_callback:
                    await progress_callback("writer_progress", {
                        "sub_topic": task.sub_topic,
                        "completed": completed_count,
                        "total": total_count,
                        "status": result["writer"]["status"],
                        "success": result["writer"]["status"] == "success"
                    })
                
                return result

        if parallel and len(tasks) > 1:
            results = await asyncio.gather(
                *[asyncio.create_task(_wrapped(task)) for task in tasks],
                return_exceptions=True,
            )
            
            normalized: List[Dict[str, Any]] = []
            for task, result in zip(tasks, results):
                if isinstance(result, Exception):
                    logger.exception(
                        "[MultiAgent] Writer task failed for sub-topic '%s'", task.sub_topic
                    )
                    normalized.append(
                        {
                            "sub_topic": task.sub_topic,
                            "question": task.question,
                            "writer": {
                                "status": "error",
                                "error": f"{type(result).__name__}: {result}",
                                "raw_output": "",
                                "parsed": None,
                            },
                            "task": task,
                        }
                    )
                else:
                    normalized.append({"sub_topic": task.sub_topic, **result, "task": task})
        else:
            normalized = []
            for task in tasks:
                try:
                    result = await _wrapped(task)
                    normalized.append({"sub_topic": task.sub_topic, **result, "task": task})
                except Exception as exc:
                    logger.exception(
                        "[MultiAgent] Writer task failed for sub-topic '%s'", task.sub_topic
                    )
                    normalized.append(
                        {
                            "sub_topic": task.sub_topic,
                            "question": task.question,
                            "writer": {
                                "status": "error",
                                "error": f"{type(exc).__name__}: {exc}",
                                "raw_output": "",
                                "parsed": None,
                            },
                            "task": task,
                        }
                    )

        if progress_callback:
            await progress_callback("writer_complete", {
                "total": total_count,
                "successful": sum(1 for r in normalized if r["writer"]["status"] == "success"),
                "failed": sum(1 for r in normalized if r["writer"]["status"] == "error")
            })

        return normalized

    async def _run_reviewers(
        self,
        writer_results: List[Dict[str, Any]],
        locale: str,
        current_date: str,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
    ) -> List[Dict[str, Any]]:
        """Run reviewers sequentially with optional progress callbacks."""
        reviewed: List[Dict[str, Any]] = []
        total_count = len(writer_results)
        completed_count = 0

        for item in writer_results:
            writer_info = item.get("writer", {})
            task: SubTopicTask = item.get("task")
            sub_topic = item.get("sub_topic")
            
            if writer_info.get("status") != "success" or not writer_info.get("parsed"):
                reviewed.append(
                    {
                        "sub_topic": sub_topic,
                        "question": item.get("question"),
                        "writer": writer_info,
                        "review": {
                            "status": "skipped",
                            "reason": "Writer output unavailable",
                            "parsed": {},
                            "ready_to_publish": False,
                        },
                    }
                )
                completed_count += 1
                
                if progress_callback:
                    await progress_callback("reviewer_progress", {
                        "sub_topic": sub_topic,
                        "completed": completed_count,
                        "total": total_count,
                        "status": "skipped",
                        "reason": "Writer failed"
                    })
                continue

            review_result = await self._invoke_reviewer(
                task=task,
                writer_parsed=writer_info["parsed"],
                locale=locale,
                current_date=current_date,
            )
            
            completed_count += 1
            
            if progress_callback:
                await progress_callback("reviewer_progress", {
                    "sub_topic": sub_topic,
                    "completed": completed_count,
                    "total": total_count,
                    "status": review_result.get("status"),
                    "ready_to_publish": review_result.get("ready_to_publish"),
                    "score": review_result.get("parsed", {}).get("reviewer_evaluation_score", "N/A")
                })
            
            reviewed.append(
                {
                    "sub_topic": sub_topic,
                    "question": item.get("question"),
                    "writer": writer_info,
                    "review": review_result,
                }
            )

        if progress_callback:
            await progress_callback("reviewer_complete", {
                "total": total_count,
                "ready_to_publish": sum(1 for r in reviewed if r["review"].get("ready_to_publish")),
                "needs_revision": sum(1 for r in reviewed if not r["review"].get("ready_to_publish"))
            })

        return reviewed

    async def _invoke_writer(
        self, task: SubTopicTask, locale: str, current_date: str
    ) -> Dict[str, Any]:
        prompt = RESEARCH_WRITER_PROMPT.format(
            current_date=current_date,
            locale=locale,
            sub_topic=task.sub_topic,
            question=task.question,
            contexts=task.contexts,
            max_tokens=task.max_tokens,
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": task.question},
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.3,
                max_tokens=min(task.max_tokens, self.default_max_tokens),
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content.strip()
            
            # ✅ Use clean_and_validate_json for robust parsing
            try:
                cleaned_content = clean_and_validate_json(content)
                parsed = json.loads(cleaned_content)
            except Exception as json_err:
                logger.error(f"[MultiAgent] JSON parsing failed for writer '{task.sub_topic}'")
                logger.error(f"[MultiAgent] Raw content: {content[:500]}...")
                logger.error(f"[MultiAgent] Parse error: {json_err}")
                raise json_err
            
            return {
                "question": task.question,
                "writer": {
                    "status": "success",
                    "raw_output": content,
                    "parsed": parsed,
                },
            }
        except Exception as exc:
            logger.exception(
                "[MultiAgent] Writer call failed for sub-topic '%s'", task.sub_topic
            )
            return {
                "question": task.question,
                "writer": {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_output": "",
                    "parsed": None,
                },
            }

    async def _invoke_reviewer(
        self,
        task: SubTopicTask,
        writer_parsed: Dict[str, Any],
        locale: str,
        current_date: str,
    ) -> Dict[str, Any]:
        prompt = RESEARCH_REVIEWER_PROMPT.format(
            current_date=current_date,
            locale=locale,
            sub_topic=task.sub_topic,
            question=task.question,
            contexts=task.contexts,
            max_tokens=task.max_tokens,
        )

        user_message = (
            "Here is the writer draft JSON to review and improve:\n"
            f"{json.dumps(writer_parsed, ensure_ascii=False)}"
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.2,
                max_tokens=min(task.max_tokens, self.default_max_tokens),
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content.strip()
            
            # ✅ Use clean_and_validate_json for robust parsing
            try:
                cleaned_content = clean_and_validate_json(content)
                parsed = json.loads(cleaned_content)
            except Exception as json_err:
                logger.error(f"[MultiAgent] JSON parsing failed for reviewer '{task.sub_topic}'")
                logger.error(f"[MultiAgent] Raw content: {content[:500]}...")
                logger.error(f"[MultiAgent] Parse error: {json_err}")
                raise json_err
            
            ready = bool(parsed.get("ready_to_publish"))
            return {
                "status": "success",
                "raw_output": content,
                "parsed": parsed,
                "ready_to_publish": ready,
            }
        except Exception as exc:
            logger.exception(
                "[MultiAgent] Reviewer call failed for sub-topic '%s'", task.sub_topic
            )
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "parsed": {},
                "raw_output": "",
                "ready_to_publish": False,
            }

    def _normalize_tasks(
        self,
        question: str,
        sub_topic: Optional[str],
        sub_topics: Optional[str],
        sub_topic_contexts: Optional[str],
        contexts: Optional[str],
        max_tokens: int,
    ) -> List[SubTopicTask]:
        tasks: List[SubTopicTask] = []
        parsed_payload = _safe_json_loads(sub_topics)

        if isinstance(parsed_payload, dict) and "sub_topics" in parsed_payload:
            parsed_payload = parsed_payload.get("sub_topics")

        if isinstance(parsed_payload, list):
            for entry in parsed_payload:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("sub_topic") or entry.get("name") or "sub-topic")
                entry_question = str(entry.get("question") or question or "")
                entry_context = str(
                    entry.get("contexts")
                    or entry.get("context")
                    or entry.get("sub_topic_contexts")
                    or ""
                )
                entry_tokens = _safe_int(entry.get("max_tokens"), max_tokens)
                tasks.append(
                    SubTopicTask(
                        sub_topic=name,
                        question=entry_question,
                        contexts=self._trim_context(entry_context),
                        max_tokens=entry_tokens,
                    )
                )

        if tasks:
            return tasks

        context_map = _safe_json_loads(sub_topic_contexts)
        if isinstance(context_map, dict) and context_map:
            for name, ctx in context_map.items():
                tasks.append(
                    SubTopicTask(
                        sub_topic=str(name),
                        question=question,
                        contexts=self._trim_context(str(ctx)),
                        max_tokens=max_tokens,
                    )
                )
            if tasks:
                return tasks

        single_context = sub_topic_contexts or contexts or ""
        resolved_sub_topic = sub_topic or "research sub-topic"
        tasks.append(
            SubTopicTask(
                sub_topic=resolved_sub_topic,
                question=question or resolved_sub_topic,
                contexts=self._trim_context(single_context),
                max_tokens=max_tokens,
            )
        )
        return tasks

    def _trim_context(self, context: str) -> str:
        if not context:
            return ""
        if len(context) <= self.context_max_chars:
            return context
        return context[: self.context_max_chars] + "\n...[truncated]"

    def _normalize_current_date(self, current_date: Optional[str]) -> str:
        if current_date:
            return current_date
        return datetime.now(tz=self.timezone).strftime("%Y-%m-%d")


async def run_multi_agent(**kwargs) -> str:
    """Convenience function for quick invocation with environment settings."""

    settings = Settings()  # Loads from environment / .env
    plugin = MultiAgentPlugin(settings)
    return await plugin.run_multi_agent(**kwargs)