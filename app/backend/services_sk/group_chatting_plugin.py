"""Group Chatting Plugin rebuilt using Semantic Kernel 1.37.0 orchestration pattern.

Implements a writer/reviewer loop per sub-topic using:
- GroupChatOrchestration
- RoundRobinGroupChatManager
- InProcessRuntime

Compatible with existing caller signature:
  sub_topics (string for a single sub-topic), question, contexts, locale,
  max_rounds, max_tokens, current_date

Returns a JSON string.
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from semantic_kernel.functions import kernel_function
from semantic_kernel.agents import (
    ChatCompletionAgent,
    GroupChatOrchestration,
    RoundRobinGroupChatManager,
)
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatMessageContent
from langchain.prompts import load_prompt
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


class GroupChattingPlugin:
    def __init__(self, settings: Settings):
        self.settings = settings
        # Single shared completion service instance (key-based auth)
        self.chat_service = AzureChatCompletion(
            deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            api_key=settings.AZURE_OPENAI_API_KEY,
            endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )

    @kernel_function(
        description="Execute multi-agent (writer/reviewer) research conversation for one sub-topic.",
        name="group_chat",
    )
    async def group_chat(
        self,
        sub_topic: str,         # kept name for upstream compatibility (single sub-topic string)
        question: str,
        sub_topic_contexts: str,           # upstream passes 'contexts'
        locale: str = "ko-KR",
        max_rounds: int = 3,
        max_tokens: int = 8000,
        current_date: Optional[str] = None,
    ) -> str:
        """
        Orchestrate a writer/reviewer refinement loop for a single sub-topic.
        Returns JSON string with conversation artifacts.
        """
        started_at = datetime.utcnow()
        if current_date is None:
            current_date = started_at.strftime("%Y-%m-%d")

        # Normalize numeric inputs (caller may pass str)
        try:
            max_rounds = int(max_rounds)
        except Exception:
            max_rounds = 1
        if max_rounds < 1:
            max_rounds = 1

        try:
            max_tokens = int(max_tokens)
        except Exception:
            max_tokens = 1000

        # Make max_rounds odd so writer gets final word (sample pattern)
        if max_rounds % 2 == 0:
            max_rounds += 1

        
        logger.info(
            "[GroupChat-Orchestration] Start sub_topic='%s' rounds=%d", sub_topic, max_rounds
        )

        # Context size guard (coarse - actual tokenization happens inside model)
        MAX_CONTEXT_CHARS = 15000
        trimmed_context = sub_topic_contexts
        if len(trimmed_context) > MAX_CONTEXT_CHARS:
            trimmed_context = trimmed_context[:MAX_CONTEXT_CHARS] + "\n...[truncated]"

        # Build agent instruction prompts - 동적 플레이스홀더 제거
        writer_instructions = RESEARCH_WRITER_PROMPT.format(
            current_date=current_date,
            locale=locale,
            sub_topic=sub_topic,
            question=question,
            contexts=trimmed_context,
            max_tokens=max_tokens
        )
        reviewer_instructions = RESEARCH_REVIEWER_PROMPT.format(
            current_date=current_date,
            locale=locale,
            sub_topic=sub_topic,
            question=question,
            contexts=trimmed_context,
            max_tokens=max_tokens
        )

        # Create agents with better instructions
        writer_agent = ChatCompletionAgent(
            name="Writer",
            description="Research content writer producing structured JSON answers. Do not include any explanations or notes, only the JSON.",
            instructions=writer_instructions + "\n\nIMPORTANT: You must output ONLY valid JSON. No explanatory text before or after.",
            service=self.chat_service,
        )
        reviewer_agent = ChatCompletionAgent(
            name="Reviewer",
            description="Reviewer improving writer output and producing final JSON. Do not include any explanations or notes, only the JSON.",
            instructions=reviewer_instructions + "\n\nIMPORTANT: You must output ONLY valid JSON. No explanatory text before or after.",
            service=self.chat_service,
            
        )

        # Capture message flow
        messages: List[Dict[str, Any]] = []
        writer_messages: List[Dict[str, Any]] = []
        reviewer_messages: List[Dict[str, Any]] = []

        def agent_response_callback(message: ChatMessageContent) -> None:
            # This callback is synchronous; just record contents
            role = getattr(message, "role", "assistant") or "assistant"
            name = getattr(message, "name", "unknown")
            content = message.content if message.content is not None else ""
            
            # JSON 형식 검증 로깅 추가
            if content.strip():
                try:
                    # JSON인지 간단히 체크
                    content_stripped = content.strip()
                    if content_stripped.startswith('{') and content_stripped.endswith('}'):
                        json.loads(content_stripped)
                        logger.info(f"[GroupChat] Agent {name} produced valid JSON")
                    else:
                        logger.warning(f"[GroupChat] Agent {name} produced non-JSON content: {content[:100]}...")
                except json.JSONDecodeError:
                    logger.warning(f"[GroupChat] Agent {name} produced invalid JSON: {content[:100]}...")
            
            record = {
                "idx": len(messages),
                "time": datetime.utcnow().isoformat(),
                "role": role,
                "agent": name,
                "content": content,
            }
            messages.append(record)
            if name == "Writer":
                writer_messages.append(record)
            elif name == "Reviewer":
                reviewer_messages.append(record)

        # Build task prompt (user message)
        task = (
            f"Sub-topic: {sub_topic}\n"
            f"Original Question: {question}\n"
            f"Locale: {locale}\n"
            f"Date: {current_date}\n\n"
            f"Context (may be truncated):\n{trimmed_context}\n\n"
            "Follow the roles: Writer drafts/improves structured answer. Reviewer critiques/refines "
            "and if satisfactory implicitly finalizes by providing best final answer."
        )

        # Configure orchestration (RoundRobin ensures writer then reviewer alternating)
        orchestration = GroupChatOrchestration(
            members=[writer_agent, reviewer_agent],
            manager=RoundRobinGroupChatManager(max_rounds=max_rounds),
            agent_response_callback=agent_response_callback,
        )

        runtime = InProcessRuntime()
        error_info: Optional[str] = None
        final_answer = ""
        rounds_used = 0

        class LoggingSuppress:
            """로깅 레벨을 임시로 조정하는 컨텍스트 매니저"""
            def __init__(self, logger_names, level=logging.ERROR):
                self.logger_names = logger_names if isinstance(logger_names, list) else [logger_names]
                self.level = level
                self.original_levels = {}
            
            def __enter__(self):
                for logger_name in self.logger_names:
                    logger_obj = logging.getLogger(logger_name)
                    self.original_levels[logger_name] = logger_obj.level
                    logger_obj.setLevel(self.level)
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                for logger_name, original_level in self.original_levels.items():
                    logging.getLogger(logger_name).setLevel(original_level)

        try:
            with LoggingSuppress([
                "semantic_kernel.agents.runtime.in_process_runtime",
                "semantic_kernel.agents.runtime",
                "semantic_kernel.agents"
            ]):
                runtime.start()
                orchestration_future = await orchestration.invoke(task=task, runtime=runtime)
                raw_final_answer = await orchestration_future.get()
                rounds_used = len(messages)

                logger.info(f"[GroupChat] Raw orchestration result type: {type(raw_final_answer)}")
                
                # StreamingChatMessageContent 객체에서 content 추출
                if hasattr(raw_final_answer, 'content'):
                    final_answer = str(raw_final_answer.content)
                else:
                    final_answer = str(raw_final_answer)
                    
                logger.info(f"[GroupChat] Extracted content length: {len(final_answer)}")
                
                # JSON 검증 및 정리
                final_answer = clean_and_validate_json(final_answer)

                logger.info(f"[GroupChat] Cleaned final answer=====================: {final_answer}")
                
        except Exception as e:
            error_info = f"{type(e).__name__}: {e}"
            logger.exception("[GroupChat-Orchestration] Error sub_topic='%s': %s", sub_topic, error_info)
            final_answer = f"Orchestration failed: {str(e)}"
        finally:
            try:
                await runtime.stop_when_idle()
            except Exception:
                pass

        # Ensure all values are JSON serializable
        try:
            started_at_str = started_at.isoformat() if started_at else datetime.utcnow().isoformat()
            ended_at_str = datetime.utcnow().isoformat()
        except Exception as dt_error:
            logger.error(f"[GroupChat] DateTime serialization error: {dt_error}")
            started_at_str = "unknown"
            ended_at_str = "unknown"

        result = {
            "status": "success" if error_info is None else "error",
            "sub_topic": str(sub_topic),  # Ensure string
            "question": str(question),    # Ensure string
            "final_answer": str(final_answer),  # Ensure string
            "rounds_used": int(rounds_used),    # Ensure int
            "max_rounds_requested": int(max_rounds),  # Ensure int
            "started_at": started_at_str,
            "ended_at": ended_at_str,
        }
        if error_info:
            result["error"] = str(error_info)  # Ensure string

        try:
            json_result = json.dumps(result, ensure_ascii=True)
            logger.info(f"[GroupChat] Successfully serialized result, length: {len(json_result)}")
            return json_result
        except Exception as serial_error:
            logger.error(f"[GroupChat] Serialization error: {serial_error}")
            logger.error(f"[GroupChat] Problematic result keys: {list(result.keys())}")
            
            # Try to identify the problematic field
            for key, value in result.items():
                try:
                    json.dumps({key: value})
                except Exception as field_error:
                    logger.error(f"[GroupChat] Field '{key}' cannot be serialized: {field_error}, type: {type(value)}")
            
            return json.dumps(
                {
                    "status": "error",
                    "sub_topic": str(sub_topic),
                    "error": f"serialization_failure: {str(serial_error)}",
                },
                ensure_ascii=False,
            )

    