"""
Plan and Search Orchestrator using Microsoft Agent Framework.

This is the main orchestrator that coordinates intent analysis, search planning,
and response generation using MS Agent Framework workflows and executors.
"""

import json
import logging
import os
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Dict, Any
import asyncio
import pytz
import base64

from config.config import Settings
from i18n.locale_msg import LOCALE_MESSAGES
from langchain.prompts import load_prompt
from model.models import ChatMessage
from utils.enum import SearchEngine
from utils.yield_message import (
    send_step_with_code,
    send_step_with_input,
    send_step_with_code_and_input,
)

# Microsoft Agent Framework imports
from agent_framework import (
    ChatAgent,
    ChatMessage as AFChatMessage,
    TextContent,
    Role,
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowOutputEvent,
    AgentRunUpdateEvent,
    handler,
    ai_function,
)
from agent_framework.azure import AzureOpenAIChatClient
from typing_extensions import Never
from typing import Annotated

from services_afw.ai_search_executor import AISearchExecutor
from services_afw.grounding_executor import GroundingExecutor
from services_afw.youtube_executor import YouTubeMCPExecutor
from services_afw.group_chatting_executor import GroupChattingExecutor
from services_afw.magentic_executor import MagenticExecutor

from services_afw.web_search_executor import WebSearchExecutor

logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))

# Load prompts (reuse from SK implementation)
INTENT_ANALYZE_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "intent_analyze_prompt.yaml"),
    encoding="utf-8",
)
RESEARCH_PLANNER_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "research_planner_prompt.yaml"),
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
GENERAL_PLANNER_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "general_planner_prompt.yaml"),
    encoding="utf-8",
)
GENERAL_ANSWER_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "general_answer_prompt.yaml"),
    encoding="utf-8",
)


class IntentAnalyzerExecutor(Executor):
    """
    Executor that analyzes user intent and generates enriched queries.

    This is the first stage of the workflow that determines what the user
    wants to do and prepares the query for downstream executors.
    Uses intent_analyze_prompt.yaml for consistent behavior with SK implementation.
    """

    def __init__(self, id: str, chat_client: AzureOpenAIChatClient, settings: Settings):
        super().__init__(id=id)
        self._chat_client = chat_client
        self._settings = settings

        # Set timezone
        if isinstance(settings.TIME_ZONE, str):
            self.timezone = pytz.timezone(settings.TIME_ZONE)
        else:
            self.timezone = pytz.UTC

    @handler
    async def analyze(
        self,
        workflow_input: Dict[str, Any],
        ctx: WorkflowContext[Dict[str, Any], str],  # Added str for yield_output
    ) -> None:
        """
        Analyze user intent from the conversation history.

        Args:
            workflow_input: Dict with 'messages' and 'metadata'
            ctx: Workflow context for sending analysis results
        """
        try:
            logger.info("IntentAnalyzerExecutor: Starting intent analysis")

            # Extract messages and metadata from input
            messages = workflow_input.get("messages", [])
            metadata = workflow_input.get("metadata", {})

            # Get locale and verbose from metadata
            locale = metadata.get("locale", "ko-KR")
            verbose = metadata.get("verbose", False)
            LOCALE_MSG = LOCALE_MESSAGES.get(locale, LOCALE_MESSAGES["ko-KR"])

            # Extract last user message
            last_user_message = next(
                (msg["content"] for msg in reversed(messages) if msg["role"] == "user"),
                "No question provided",
            )

            if last_user_message == "No question provided":
                await ctx.send_message(
                    {
                        "error": "No question provided",
                        "user_intent": "unknown",
                        "enriched_query": "",
                        "search_query": "",
                        "metadata": metadata,
                    }
                )
                return

            # Get locale from metadata
            current_date = datetime.now(tz=self.timezone).strftime("%Y-%m-%d")

            # Use intent_analyze_prompt.yaml (same as SK implementation)
            intent_prompt = INTENT_ANALYZE_PROMPT.format(
                current_date=current_date,
                original_query=last_user_message,
                locale=locale,
            )

            logger.info(f"IntentAnalyzerExecutor: Analyzing query with locale={locale}")

            # Call LLM for intent analysis
            af_messages = [
                AFChatMessage(role=Role.SYSTEM, text=intent_prompt),
                AFChatMessage(role=Role.USER, text=last_user_message),
            ]

            response = await self._chat_client.get_response(messages=af_messages)
            response_text = response.messages[-1].text

            # Parse JSON response
            try:
                # Extract JSON from markdown code block if present
                if "```json" in response_text:
                    response_text = (
                        response_text.split("```json")[1].split("```")[0].strip()
                    )
                elif "```" in response_text:
                    response_text = (
                        response_text.split("```")[1].split("```")[0].strip()
                    )

                intent_data = json.loads(response_text)
                intent_data["original_query"] = last_user_message

                # Validate required keys and set defaults (matching SK implementation)
                required_keys = ["user_intent", "enriched_query"]
                for key in required_keys:
                    if key not in intent_data:
                        raise ValueError(f"Missing required key: {key}")

                # Validate intent value
                valid_intents = [
                    "research",
                    "general_query",
                    "small_talk",
                    "tool_calling",
                ]
                if intent_data["user_intent"] not in valid_intents:
                    logger.warning(
                        f"Invalid intent detected: {intent_data['user_intent']}, defaulting to general_query"
                    )
                    intent_data["user_intent"] = "general_query"

                # Set default values for optional fields (matching SK implementation)
                intent_data.setdefault("confidence", 0.8)
                intent_data.setdefault("keywords", [])
                intent_data.setdefault("target_info", "general information")
                intent_data.setdefault("tool_name", "")
                intent_data.setdefault("search_query", intent_data["enriched_query"])

                logger.info(
                    f"IntentAnalyzerExecutor: Intent = {intent_data.get('user_intent')}"
                )
                logger.info(
                    f"IntentAnalyzerExecutor: Enriched query = {intent_data.get('enriched_query')}"
                )
                logger.info(
                    f"IntentAnalyzerExecutor: Confidence = {intent_data.get('confidence')}"
                )

                # Add metadata to intent_data for next executors
                intent_data["metadata"] = metadata

                # ✅ Yield progress message (SK compatible format)
                if verbose:
                    intent_data_str = json.dumps(
                        intent_data, ensure_ascii=False, indent=2
                    )
                    await ctx.yield_output(
                        f"data: {send_step_with_code(LOCALE_MSG['analyze_complete'], intent_data_str)}\n\n"
                    )

                # Send results to next executor
                await ctx.send_message(intent_data)

            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"IntentAnalyzerExecutor: Failed to parse JSON: {e}")
                logger.error(
                    f"IntentAnalyzerExecutor: Response text: {response_text[:500]}"
                )
                # Fallback (matching SK implementation)
                fallback_data = {
                    "user_intent": "general_query",
                    "enriched_query": last_user_message,
                    "search_query": last_user_message,
                    "original_query": last_user_message,
                    "confidence": 0.5,
                    "keywords": [],
                    "target_info": "general information",
                    "tool_name": "",
                    "metadata": metadata,
                }
                await ctx.yield_output(
                    f"data: {send_step_with_code(LOCALE_MSG['analyze_complete'], fallback_data)}\n\n"
                )
                await ctx.send_message(fallback_data)

        except Exception as e:
            logger.error(f"IntentAnalyzerExecutor: Error during analysis: {e}")
            await ctx.send_message(
                {
                    "error": str(e),
                    "user_intent": "general_query",
                    "enriched_query": last_user_message
                    if "last_user_message" in locals()
                    else "",
                    "search_query": last_user_message
                    if "last_user_message" in locals()
                    else "",
                    "original_query": last_user_message
                    if "last_user_message" in locals()
                    else "",
                    "confidence": 0.5,
                    "keywords": [],
                    "target_info": "general information",
                    "tool_name": "",
                    "metadata": metadata,
                }
            )


class TaskPlannerExecutor(Executor):
    """
    Executor that creates a search plan with multiple queries and sub-topics.

    Takes the intent analysis results and generates a structured search plan
    that will guide the search executors.
    """

    def __init__(self, id: str, chat_client: AzureOpenAIChatClient, settings: Settings):
        super().__init__(id=id)
        self._chat_client = chat_client
        self._settings = settings

    @handler
    async def plan(
        self,
        intent_data: Dict[str, Any],
        ctx: WorkflowContext[Dict[str, Any], str],  # Added str for yield_output
    ) -> None:
        """
        Generate a search plan based on intent analysis.

        Args:
            intent_data: Intent analysis results from IntentAnalyzerExecutor
            ctx: Workflow context for sending search plan
        """
        try:
            logger.info("SearchPlannerExecutor: Generating search plan")

            # Get metadata for verbose and locale
            metadata = intent_data.get("metadata", {})
            locale = metadata.get("locale", "ko-KR")
            verbose = metadata.get("verbose", False)
            LOCALE_MSG = LOCALE_MESSAGES.get(locale, LOCALE_MESSAGES["ko-KR"])

            user_intent = intent_data.get("user_intent", "general_query")
            enriched_query = intent_data.get("enriched_query", "")

            # Skip planning for small talk
            if user_intent == "small_talk":
                await ctx.send_message(
                    {
                        **intent_data,
                        "search_queries": [],
                        "sub_topics": [],
                        "planning_done": False,
                    }
                )
                return

            current_date = datetime.now().strftime("%Y-%m-%d")

            # Use appropriate planner prompt based on user intent
            if user_intent == "research":
                planner_prompt = RESEARCH_PLANNER_PROMPT.format(
                    planner_max_plans=self._settings.PLANNER_MAX_PLANS,
                    current_date=current_date,
                    enriched_query=enriched_query,
                    locale=locale,
                )
            else:
                planner_prompt = GENERAL_PLANNER_PROMPT.format(
                    planner_max_plans=self._settings.PLANNER_MAX_PLANS,
                    current_date=current_date,
                    enriched_query=enriched_query,
                    locale=locale,
                )

            logger.info(
                f"Using {'RESEARCH' if user_intent == 'research' else 'GENERAL'} planner prompt"
            )

            # Call LLM for planning
            af_messages = [
                AFChatMessage(
                    role=Role.SYSTEM,
                    contents=[
                        TextContent(
                            text="You are an expert search planner. Always respond with valid JSON."
                        )
                    ],
                ),
                AFChatMessage(
                    role=Role.USER, contents=[TextContent(text=planner_prompt)]
                ),
            ]

            response = await self._chat_client.get_response(messages=af_messages)
            response_text = response.messages[-1].text

            # Parse JSON response
            try:
                # Extract JSON from markdown code block if present
                if "```json" in response_text:
                    response_text = (
                        response_text.split("```json")[1].split("```")[0].strip()
                    )
                elif "```" in response_text:
                    response_text = (
                        response_text.split("```")[1].split("```")[0].strip()
                    )

                plan_data = json.loads(response_text)
                search_queries_data = plan_data.get("search_queries", [])

                # Extract flat queries and sub-topics
                flat_queries = []
                sub_topics = []

                for item in search_queries_data:
                    if (
                        isinstance(item, dict)
                        and "sub_topic" in item
                        and "queries" in item
                    ):
                        sub_topics.append(item)
                        flat_queries.extend(item["queries"])
                    elif isinstance(item, str):
                        # Fallback format
                        flat_queries.append(item)
                        sub_topics.append({"sub_topic": "research", "queries": [item]})

                if not sub_topics:
                    # Fallback
                    sub_topics = [
                        {"sub_topic": "research report", "queries": [enriched_query]}
                    ]
                    flat_queries = [enriched_query]

                logger.info(
                    f"SearchPlannerExecutor: Generated {len(sub_topics)} sub-topics with {len(flat_queries)} total queries"
                )

                # ✅ Yield progress message (SK compatible format)
                plan_result = {
                    "sub_topics": sub_topics,
                    "search_queries": flat_queries,
                    "planning_done": True,
                }

                if verbose:
                    plan_data_str = json.dumps(
                        plan_result, ensure_ascii=False, indent=2
                    )
                    await ctx.yield_output(
                        f"data: {send_step_with_code(LOCALE_MSG['plan_done'], plan_data_str)}\n\n"
                    )
                else:
                    await ctx.yield_output(
                        f"data: {send_step_with_code(LOCALE_MSG['plan_done'], plan_data_str)}\n\n"
                    )

                # Send complete data to next executor
                await ctx.send_message(
                    {
                        **intent_data,
                        "search_queries": flat_queries,
                        "sub_topics": sub_topics,
                        "planning_done": True,
                    }
                )

            except json.JSONDecodeError as e:
                logger.error(f"SearchPlannerExecutor: Failed to parse JSON: {e}")
                # Fallback
                await ctx.yield_output(f"data: ### {LOCALE_MSG['plan_done']}\n\n")
                await ctx.send_message(
                    {
                        **intent_data,
                        "search_queries": [enriched_query],
                        "sub_topics": [
                            {
                                "sub_topic": "research report",
                                "queries": [enriched_query],
                            }
                        ],
                        "planning_done": False,
                    }
                )

        except Exception as e:
            logger.error(f"SearchPlannerExecutor: Error during planning: {e}")
            enriched_query = intent_data.get("enriched_query", "")
            await ctx.send_message(
                {
                    **intent_data,
                    "search_queries": [enriched_query],
                    "sub_topics": [
                        {"sub_topic": "research report", "queries": [enriched_query]}
                    ],
                    "planning_done": False,
                    "error": str(e),
                }
            )


class ResponseGeneratorExecutor(Executor):
    """
    Executor that generates the final response using all gathered context.

    This is the final stage that synthesizes all search results, AI search results,
    and other context into a comprehensive answer.
    """

    def __init__(self, id: str, chat_client: AzureOpenAIChatClient, settings: Settings):
        super().__init__(id=id)
        self._chat_client = chat_client
        self._settings = settings

    @handler
    async def generate(
        self, context_data: Dict[str, Any], ctx: WorkflowContext[Never, str]
    ) -> None:
        """
        Generate final response using all context.

        Args:
            context_data: Dictionary with intent, plan, and search contexts
            ctx: Workflow context for yielding final output
        """
        try:
            logger.info("ResponseGeneratorExecutor: Generating final response")

            # Get metadata for verbose and locale
            metadata = context_data.get("metadata", {})
            locale = metadata.get("locale", "ko-KR")
            LOCALE_MSG = LOCALE_MESSAGES.get(locale, LOCALE_MESSAGES["ko-KR"])

            # ✅ Yield starting message for final answer generation
            await ctx.yield_output(f"data: ### {LOCALE_MSG['answering']}\n\n")

            original_query = context_data.get("original_query", "")
            enriched_query = context_data.get("enriched_query", original_query)
            user_intent = context_data.get("user_intent", "general_query")

            all_contexts = context_data.get("all_contexts", [])
            current_date = datetime.now().strftime("%Y-%m-%d")

            # Build context string
            context_str = (
                "\n\n".join(all_contexts)
                if all_contexts
                else "No additional context available."
            )

            # Use GENERAL_ANSWER_PROMPT from YAML file (same as SK)
            response_prompt_text = GENERAL_ANSWER_PROMPT.format(
                current_date=current_date,
                contexts=context_str,
                question=enriched_query,
                locale=locale,
            )

            logger.info(f"Using GENERAL_ANSWER_PROMPT for {user_intent} intent")

            # Generate response
            af_messages = [
                AFChatMessage(
                    role=Role.SYSTEM, contents=[TextContent(text=response_prompt_text)]
                ),
                AFChatMessage(
                    role=Role.USER, contents=[TextContent(text=enriched_query)]
                ),
            ]

            response = await self._chat_client.get_response(messages=af_messages)
            final_response = response.messages[-1].text

            logger.info("ResponseGeneratorExecutor: Response generated successfully")

            # Yield final output
            await ctx.yield_output(final_response)

        except Exception as e:
            logger.error(f"ResponseGeneratorExecutor: Error generating response: {e}")
            await ctx.yield_output(f"Error generating response: {str(e)}")


class PlanSearchOrchestratorAFW:
    """
    Plan and Search Orchestrator using Microsoft Agent Framework.

    This orchestrator coordinates the entire workflow:
    1. Intent analysis
    2. Search planning
    3. Web search (optional)
    4. AI search (optional)
    5. YouTube search (optional)
    6. Response generation
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        if isinstance(settings.TIME_ZONE, str):
            self.timezone = pytz.timezone(settings.TIME_ZONE)
        else:
            self.timezone = pytz.UTC

        # Initialize Azure OpenAI Chat Client for Agent Framework
        self.chat_client = AzureOpenAIChatClient(
            deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )

        logger.info(
            f"PlanSearchOrchestrator initialized with Azure OpenAI deployment: {settings.AZURE_OPENAI_DEPLOYMENT_NAME}"
        )

    async def generate_response(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        research: bool = True,
        planning: bool = True,
        search_engine: SearchEngine = SearchEngine.BING_SEARCH_CRAWLING,
        stream: bool = False,
        elapsed_time: bool = True,
        locale: Optional[str] = "en-US",
        include_web_search: bool = True,
        include_ytb_search: bool = True,
        include_mcp_server: bool = True,
        include_ai_search: bool = True,
        multi_agent_type: str = "vanilla",
        verbose: Optional[bool] = False,
    ) -> AsyncGenerator[str, None]:
        """
        Generate response using Microsoft Agent Framework workflows.

        This method orchestrates the entire response generation pipeline using
        AF executors and workflows.
        """
        try:
            start_time = datetime.now(tz=self.timezone)
            if elapsed_time:
                logger.info(f"Starting response generation at {start_time}")

            # Convert messages to dict format
            messages_dict = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]
            last_user_message = next(
                (
                    msg["content"]
                    for msg in reversed(messages_dict)
                    if msg["role"] == "user"
                ),
                "No question provided",
            )

            # Get locale messages
            LOCALE_MSG = LOCALE_MESSAGES.get(locale, LOCALE_MESSAGES["ko-KR"])
            if last_user_message == "No question provided":
                yield LOCALE_MSG["input_needed"]
                return

            if max_tokens is None:
                max_tokens = self.settings.MAX_TOKENS
            if temperature is None:
                temperature = self.settings.DEFAULT_TEMPERATURE

            if stream:
                yield f"data: ### {LOCALE_MSG['analyzing']}\n\n"

            # Create executors
            intent_analyzer = IntentAnalyzerExecutor(
                id="intent_analyzer",
                chat_client=self.chat_client,
                settings=self.settings,
            )

            # Build workflow - executors now handle their own progress via ctx.yield_output()
            workflow_builder = WorkflowBuilder()
            workflow_builder.set_start_executor(intent_analyzer)

            # Track the last executor in the pipeline
            last_executor = intent_analyzer

            # Add planning stage if enabled
            if planning:
                search_planner = TaskPlannerExecutor(
                    id="search_planner",
                    chat_client=self.chat_client,
                    settings=self.settings,
                )
                workflow_builder.add_edge(last_executor, search_planner)
                last_executor = search_planner

            # Phase 2: Add optional search executors (independent of planning)
            if include_web_search:
                web_search_executor = WebSearchExecutor(
                    id="web_search",
                    bing_api_key=os.getenv("BING_API_KEY"),
                    bing_endpoint=os.getenv("BING_ENDPOINT"),
                    bing_custom_config_id=os.getenv("BING_CUSTOM_CONFIG_ID"),
                )
                workflow_builder.add_edge(last_executor, web_search_executor)
                last_executor = web_search_executor

            if include_ai_search:
                ai_search_executor = AISearchExecutor(
                    id="ai_search",
                    search_endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
                    search_key=os.getenv("AZURE_AI_SEARCH_API_KEY"),
                    index_name=os.getenv("AZURE_AI_SEARCH_INDEX_NAME"),
                    openai_endpoint=self.settings.AZURE_OPENAI_ENDPOINT,
                    openai_key=self.settings.AZURE_OPENAI_API_KEY,
                    embedding_deployment=os.getenv(
                        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"
                    ),
                    openai_api_version=self.settings.AZURE_OPENAI_API_VERSION,
                    search_type=os.getenv("AZURE_AI_SEARCH_SEARCH_TYPE", "hybrid"),  # ✅ 환경변수에서 search_type 전달
                )
                workflow_builder.add_edge(last_executor, ai_search_executor)
                last_executor = ai_search_executor

            if include_ytb_search and include_mcp_server:
                youtube_executor = YouTubeMCPExecutor(
                    id="youtube_search",
                    api_key=os.getenv("YOUTUBE_API_KEY"),
                    max_results=10,
                )
                workflow_builder.add_edge(last_executor, youtube_executor)
                last_executor = youtube_executor

            # Add grounding search for web-based queries (alternative to web search)
            if os.getenv("BING_GROUNDING_PROJECT_ENDPOINT"):
                grounding_executor = GroundingExecutor(
                    id="grounding_search",
                    settings=self.settings,
                    project_endpoint=os.getenv("BING_GROUNDING_PROJECT_ENDPOINT"),
                    connection_id=os.getenv("BING_GROUNDING_CONNECTION_ID"),
                    agent_model_deployment_name=os.getenv(
                        "BING_GROUNDING_AGENT_MODEL_DEPLOYMENT_NAME"
                    ),
                    max_results=5,
                    market=locale if locale else "ko-KR",
                )
                workflow_builder.add_edge(last_executor, grounding_executor)
                last_executor = grounding_executor

            # Add multi-agent research for deep research tasks
            if research:
                multi_agent_executor = None
                if multi_agent_type == "MS Agent Framework Magentic":
                    # Use Magentic orchestration pattern
                    multi_agent_executor = MagenticExecutor(
                        id="magentic_research",
                        chat_client=self.chat_client,
                        settings=self.settings,
                        context_max_chars=400000,  # ✅ SK와 동일한 MAX_CONTEXT_LENGTH
                        max_document_length=10000,  # ✅ 문서당 최대 길이
                        writer_parallel_limit=4,
                    )
                    logger.info("🎯 Using Magentic orchestration pattern for research")
                elif multi_agent_type == "MS Agent Framework GroupChat":
                    multi_agent_executor = GroupChattingExecutor(
                        id="group_chatting_research",
                        chat_client=self.chat_client,
                        settings=self.settings,
                        context_max_chars=400000,
                        max_document_length=10000,
                        writer_parallel_limit=4,
                    )
                    logger.info("💬 Using Group Chat pattern for research")
                else:
                    # Default to GroupChatting
                    multi_agent_executor = GroupChattingExecutor(
                        id="group_chatting_research",
                        chat_client=self.chat_client,
                        settings=self.settings,
                        context_max_chars=400000,
                        max_document_length=10000,
                        writer_parallel_limit=4,
                    )
                    logger.info("💬 Using default Group Chat pattern for research")

                if multi_agent_executor:
                    workflow_builder.add_edge(last_executor, multi_agent_executor)
                    last_executor = multi_agent_executor

                    # ✅ For research intent, multi-agent executor is the FINAL executor
                    # Don't add ResponseGeneratorExecutor - output directly from research executor
                    if multi_agent_type == "MS Agent Framework Magentic":
                        logger.info(
                            "🔬 Research mode: MagenticExecutor will be the final node"
                        )
                    else:
                        logger.info(
                            "🔬 Research mode: GroupChattingExecutor will be the final node"
                        )
            else:
                # ✅ For general queries, add ResponseGeneratorExecutor as final node
                response_generator = ResponseGeneratorExecutor(
                    id="response_generator",
                    chat_client=self.chat_client,
                    settings=self.settings,
                )
                workflow_builder.add_edge(last_executor, response_generator)
                logger.info(
                    "💬 General query mode: ResponseGeneratorExecutor is the final node"
                )

            # Build the workflow
            workflow = workflow_builder.build()

            # Execute workflow with locale metadata
            logger.info(f"Executing Agent Framework workflow with locale={locale}...")

            # Create input with metadata embedded
            # Agent Framework passes the input directly to the first handler
            # We'll pass a dict that includes both messages and metadata
            workflow_input = {
                "messages": messages_dict,
                "metadata": {
                    "locale": locale,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "include_web_search": include_web_search,
                    "include_ai_search": include_ai_search,
                    "include_ytb_search": include_ytb_search,
                    "verbose": verbose,
                },
            }

            if stream:
                ttft_time = None
                first_answer_token_yielded = False

                # Stream workflow events
                async for event in workflow.run_stream(workflow_input):
                    event_data = None
                    if hasattr(event, "data"): 
                        event_data = event.data
                    # ✅ 이제 event_data가 정의된 후에 체크
                    if event_data and isinstance(event_data, dict):
                        executor_error = event_data.get("executor_error")
                        if executor_error and executor_error.get("is_fatal"):
                            executor_name = executor_error.get("executor", "Unknown")
                            error_type = executor_error.get("error_type", "unknown")
                            error_message = executor_error.get("error_message", "Unknown error")

                            logger.error(
                                f"🔴 Fatal error from {executor_name} executor: {error_type}"
                            )
                            logger.error(f"   Error details: {error_message}")

                            # ✅ 기존 progress message처럼 ### 포맷으로 전송 (frontend가 이미 처리 가능)
                            yield f"data: ### ❌ {executor_name.upper()} 오류 발생\n\n"
                            yield f"data: ### error type: {error_type}\n\n"
                            yield f"data: ### {error_message[:300]}\n\n"
                            yield f"data: ### Terminate the task\n\n"
                            yield "data: [DONE]\n\n"

                            logger.info(f"🛑 Workflow terminated due to fatal error from {executor_name}")
                            return  # ✅ 워크플로우 즉시 종료

                    # Handle progress messages
                    if event_data and isinstance(event_data, dict):
                        progress_msg = event_data.get("_progress_message")
                        if progress_msg:
                            yield f"data: {progress_msg}\n\n"
                            logger.info(f"📢 Progress: {progress_msg[:80]}...")
                            event_data.pop("_progress_message", None)

                    # Handle different event types
                    if isinstance(event, WorkflowOutputEvent):
                        logger.info(f"🎯 WorkflowOutputEvent received")
                        output_text = (
                            event.output if hasattr(event, "output") else event_data
                        )

                        if output_text:
                            # ✅ Check for TTFT marker
                            if (
                                isinstance(output_text, str)
                                and "__TTFT_MARKER__" in output_text
                            ):
                                if not first_answer_token_yielded:
                                    ttft_time = (
                                        datetime.now(tz=self.timezone) - start_time
                                    )
                                    first_answer_token_yielded = True
                                    logger.info(
                                        f"⏱️ TTFT (from GroupChattingExecutor): {ttft_time.total_seconds():.2f}s"
                                    )
                                # Don't yield the marker to frontend
                                continue

                            # Check if this is a progress message
                            is_progress_message = isinstance(
                                output_text, str
                            ) and output_text.strip().startswith("data: ###")

                            if is_progress_message:
                                yield f"{output_text}"
                                logger.info(f"📢 Progress: {output_text[:80]}...")
                            else:
                                # ✅ Final answer (from ResponseGenerator OR GroupChattingExecutor)
                                if not first_answer_token_yielded:
                                    ttft_time = (
                                        datetime.now(tz=self.timezone) - start_time
                                    )
                                    first_answer_token_yielded = True
                                    logger.info(
                                        f"⏱️ TTFT: {ttft_time.total_seconds():.2f}s"
                                    )

                                yield f"{output_text}"

                    elif isinstance(event, AgentRunUpdateEvent):
                        logger.info(f"📝 AgentRunUpdateEvent received")
                        if event_data:
                            yield f"data: {event_data}\n\n"

                # Add elapsed time at the end
                if elapsed_time:
                    total_elapsed = datetime.now(tz=self.timezone) - start_time
                    elapsed_msg = LOCALE_MSG.get("elapsed_time", "Elapsed Time")

                    if ttft_time is not None:
                        logger.info(
                            f"✅ Response completed - TTFT: {ttft_time.total_seconds():.2f}s, Total: {total_elapsed.total_seconds():.2f}s"
                        )
                        yield "\n"
                        yield f"doc research response generated successfully in {ttft_time.total_seconds():.2f} seconds \n"
                        yield f"\n"
                        yield f"data: ### ⏱️ TTFT: {ttft_time.total_seconds():.2f}s | Total: {total_elapsed.total_seconds():.2f}s\n\n"
                    else:
                        logger.warning(
                            f"⚠️ No final answer token detected - Total elapsed: {total_elapsed.total_seconds():.2f}s"
                        )
                        yield f"data: ### {elapsed_msg}: {total_elapsed.total_seconds():.2f}s\n\n"
            else:
                # Non-streaming execution
                events = await workflow.run(workflow_input)

                final_output = None
                for event in events:
                    # Check for progress messages
                    event_data = event.data if hasattr(event, "data") else None
                    if event_data and isinstance(event_data, dict):
                        progress_msg = event_data.get("_progress_message")
                        if progress_msg:
                            yield f"data: {progress_msg}\n\n"
                            event_data.pop("_progress_message", None)

                    # Get final output
                    if isinstance(event, WorkflowOutputEvent):
                        final_output = (
                            event.output if hasattr(event, "output") else event_data
                        )

                if final_output:
                    yield f"{final_output}\n\n"

                # Add elapsed time
                if elapsed_time:
                    elapsed = (
                        datetime.now(tz=self.timezone) - start_time
                    ).total_seconds()
                    elapsed_msg = LOCALE_MSG.get("elapsed_time", "Elapsed Time")
                    yield f"\ndata: ### {elapsed_msg}: {elapsed:.2f}s\n\n"
        except Exception as e:
            logger.error(
                f"Error in PlanSearchOrchestrator.generate_response: {str(e)}",
                exc_info=True,
            )
            yield f"Error generating response: {str(e)}"