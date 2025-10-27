import json
import logging
import os
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Dict, Any, Callable, Awaitable
import asyncio
import pytz
from config.config import Settings
from i18n.locale_msg import LOCALE_MESSAGES
from langchain.prompts import load_prompt
from model.models import ChatMessage
from openai import AsyncAzureOpenAI
from utils.enum import SearchEngine
import base64

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments

from .search_plugin import SearchPlugin
from services_sk.youtube_plugin import YouTubePlugin
from services_sk.youtube_mcp_plugin import YouTubeMCPPlugin
from .intent_plan_plugin import IntentPlanPlugin
from .grounding_plugin import GroundingPlugin
from .ai_search_plugin import AISearchPlugin
from .unified_file_upload_plugin import UnifiedFileUploadPlugin
from .group_chatting_plugin import GroupChattingPlugin
from .vanilla_multi_agent_plugin import MultiAgentPlugin
from utils.json_control import clean_duplicate_table_content


logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))

# Load prompts
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


class PlanSearchOrchestratorSK:
    """
    Plan and Search Executor using Semantic Kernel.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        if isinstance(settings.TIME_ZONE, str):
            self.timezone = pytz.timezone(settings.TIME_ZONE)
        else:
            self.timezone = pytz.UTC

        # Initialize OpenAI client for legacy operations
        self.client = AsyncAzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        )

        # Initialize Semantic Kernel
        self.kernel = Kernel()

        # Add Azure OpenAI chat completion service
        self.chat_completion = AzureChatCompletion(
            deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            api_key=settings.AZURE_OPENAI_API_KEY,
            base_url=settings.AZURE_OPENAI_ENDPOINT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )
        self.kernel.add_service(self.chat_completion)

        # Initialize plugins
        bing_api_key = getattr(settings, "BING_API_KEY", None)
        bing_endpoint = getattr(settings, "BING_ENDPOINT", None)
        
        logger.info(f"Initializing SearchPlugin with:")
        logger.info(
            f"  - bing_api_key from settings: {'SET' if bing_api_key else 'NOT SET'}"
        )
        logger.info(f"  - bing_endpoint from settings: {bing_endpoint}")

        self.search_plugin = SearchPlugin(
            bing_api_key=bing_api_key, bing_endpoint=bing_endpoint
        )
        self.intent_plan_plugin = IntentPlanPlugin(settings)
        self.ai_search_plugin = AISearchPlugin(settings)
        self.unified_file_upload_plugin = UnifiedFileUploadPlugin()
        self.group_chatting_plugin = GroupChattingPlugin(settings)
        self.multi_agent_plugin = MultiAgentPlugin(settings)

        self.kernel.add_plugin(self.intent_plan_plugin, plugin_name="intent_plan")
        self.kernel.add_plugin(self.ai_search_plugin, plugin_name="ai_search")
        self.kernel.add_plugin(
            self.unified_file_upload_plugin, plugin_name="file_upload"
        )
        self.kernel.add_plugin(self.group_chatting_plugin, plugin_name="sk_group_chat")
        self.kernel.add_plugin(
            self.multi_agent_plugin, plugin_name="vanilla_multi_agent"
        )

        self.deployment_name = settings.AZURE_OPENAI_DEPLOYMENT_NAME
        self.query_deployment_name = settings.AZURE_OPENAI_QUERY_DEPLOYMENT_NAME
        self.planner_max_plans = settings.PLANNER_MAX_PLANS

        logger.debug(
            f"RiskSearchExecutor initialized with Azure OpenAI deployment: {self.deployment_name}"
        )

    @staticmethod
    def send_step_with_code(step_name: str, code: str) -> str:
        """Send a step with code content"""
        encoded_code = base64.b64encode(code.encode("utf-8")).decode("utf-8")
        return f"### {step_name}#code#{encoded_code}"

    @staticmethod
    def send_step_with_input(step_name: str, description: str) -> str:
        """Send a step with input description"""
        return f"### {step_name}#input#{description}"

    @staticmethod
    def send_step_with_code_and_input(
        step_name: str, code: str, description: str
    ) -> str:
        """Send a step with both code and input description"""
        encoded_code = base64.b64encode(code.encode("utf-8")).decode("utf-8")
        return f"### {step_name}#input#{description}#code#{encoded_code}"

    def _extract_search_queries(self, search_queries_data) -> List[str]:
        """
        Extract flat list of query strings from search plan data.
        Supports both old format (list of strings) and new format (list of sub-topic dicts).
        """
        queries = []

        if isinstance(search_queries_data, list):
            for item in search_queries_data:
                if isinstance(item, str):
                    # Old format: direct string
                    queries.append(item)
                elif isinstance(item, dict) and "queries" in item:
                    # New format: sub-topic with queries array
                    queries.extend(item["queries"])

        # Fallback to single string if no queries found
        return queries if queries else [str(search_queries_data)]

    def _extract_sub_topics(self, search_queries_data) -> List[Dict[str, Any]]:
        """
        Extract sub-topics with their queries for structured processing.
        Returns list of {sub_topic: str, queries: List[str]} dictionaries.
        """
        sub_topics = []

        if isinstance(search_queries_data, list):
            for item in search_queries_data:
                if isinstance(item, str):
                    # Old format: treat as single sub-topic
                    sub_topics.append(
                        {"sub_topic": "research report", "queries": [item]}
                    )
                elif (
                    isinstance(item, dict) and "sub_topic" in item and "queries" in item
                ):
                    # New format: use as-is
                    sub_topics.append(
                        {"sub_topic": item["sub_topic"], "queries": item["queries"]}
                    )

        # Fallback for empty or invalid data
        if not sub_topics:
            sub_topics = [
                {"sub_topic": "research report", "queries": [str(search_queries_data)]}
            ]

        return sub_topics

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
        Generate response using semantic kernel with search and/or MCP plugins.

        Args:
            messages: Chat messages history
            max_tokens: Maximum tokens for response
            temperature: Temperature for response generation
            query_rewrite: Whether to rewrite the query
            planning: Whether to use planning to search
            search_engine: Search engine to use
            stream: Whether to stream response
            elapsed_time: Whether to include elapsed time
            locale: Locale for search and response
            include_web_search: Whether to include web search results
            include_ytb_search: Whether to include YouTube search results
            include_mcp_server: Whether to include MCP server integration
            include_ai_search: Whether to include AI search results from uploaded documents
            multi_agent_type: The type of multi-agent system to use
            verbose: Whether to include verbose context information,

        """
        try:
            start_time = datetime.now(tz=self.timezone)
            if elapsed_time:
                logger.info(f"Starting risk search response generation at {start_time}")
                ttft_time = None

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

            current_date = datetime.now(tz=self.timezone).strftime("%Y-%m-%d")

            if max_tokens is None:
                max_tokens = self.settings.MAX_TOKENS
            if temperature is None:
                temperature = self.settings.DEFAULT_TEMPERATURE

            if stream:
                yield f"data: ### {LOCALE_MSG['analyzing']}\n\n"

            # Intent analysis and query rewriting using IntentPlugin
            enriched_query = last_user_message
            search_queries = []
            sub_topics = []
            resource_group_name = None

            try:
                # Use IntentPlugin for intent analysis
                intent_function = self.kernel.get_function(
                    "intent_plan", "analyze_intent"
                )
                intent_result = await intent_function.invoke(
                    self.kernel,
                    KernelArguments(
                        original_query=last_user_message, locale=locale, temperature=0.3
                    ),
                )

                if intent_result and intent_result.value:
                    intent_data = json.loads(intent_result.value)
                    user_intent = intent_data.get("user_intent", "general_query")
                    enriched_query = intent_data.get(
                        "enriched_query", last_user_message
                    )
                    search_queries = [
                        intent_data.get("search_query", last_user_message)
                    ]
                    resource_group_name = intent_data.get("resource_group_name")

                    logger.info("=" * 60)
                    logger.info("---Intent analysis result--")
                    logger.info(f"User intent: {user_intent}")
                    logger.info(f"Enriched query: {enriched_query}")
                    logger.info(f"Search queries: {search_queries}")
                    logger.info(f"resource group name: {resource_group_name}")
                    logger.info("=" * 60)

                if verbose and stream:
                    intent_data_str = (
                        json.dumps(intent_data, ensure_ascii=False, indent=2)
                        if intent_data
                        else "{}"
                    )
                    yield f"data: {self.send_step_with_code(LOCALE_MSG['analyze_complete'], intent_data_str)}\n\n"

                if user_intent == "small_talk":
                    # Small talk does not require search
                    planning = False
                    include_web_search = False
                    include_ytb_search = False

                    if stream:
                        yield f"data: ### {LOCALE_MSG['intent_small_talk']}\n\n"

                if planning:
                    if stream:
                        yield f"data: ### {LOCALE_MSG['task_planning']}\n\n"

                    # Generate search plan using IntentPlanPlugin
                    plan_function = self.kernel.get_function(
                        "intent_plan", "generate_search_plan"
                    )
                    plan_result = await plan_function.invoke(
                        self.kernel,
                        KernelArguments(
                            user_intent=user_intent,
                            enriched_query=enriched_query,
                            locale=locale,
                            temperature=0.7,
                        ),
                    )

                    if plan_result and plan_result.value:
                        plan_data = json.loads(plan_result.value)
                        # Extract both flat queries and structured sub-topics
                        raw_search_queries = plan_data.get(
                            "search_queries", [enriched_query]
                        )
                        search_queries = self._extract_search_queries(
                            raw_search_queries
                        )
                        sub_topics = self._extract_sub_topics(raw_search_queries)

                        logger.info(f"Search plan: {plan_data}")
                        logger.info(f"Extracted search queries: {search_queries}")
                        logger.info(f"Extracted sub-topics: {sub_topics}")
                    else:
                        # Fallback
                        search_queries = [enriched_query]
                        sub_topics = []
                        sub_topics.append(
                            {
                                "sub_topic": "research report",
                                "queries": [enriched_query],
                            }
                        )

                    if verbose and stream:
                        plan_data_str = (
                            json.dumps(plan_data, ensure_ascii=False, indent=2)
                            if plan_data
                            else "{}"
                        )
                        yield f"data: {self.send_step_with_code(LOCALE_MSG['plan_done'], plan_data_str)}\n\n"
                else:
                    # No planning, use enriched query directly
                    search_queries = [enriched_query]
                    sub_topics = []
                    sub_topics.append(
                        {"sub_topic": "research report", "queries": [enriched_query]}
                    )

            except Exception as e:
                logger.error(f"Error during intent analysis: {e}")
                # Fallback to original query
                search_queries = [enriched_query]
                sub_topics = []
                sub_topics.append(
                    {"sub_topic": "research report", "queries": [enriched_query]}
                )
                if stream:
                    yield f"data: ### Intent analysis failed, using fallback\n\n"

            # Collect contexts
            all_contexts = []

            # Web search context by sub-topic (NEW APPROACH - SIMPLE)
            if include_web_search and sub_topics:
                try:
                    sub_topic_web_contexts = {}  # Store results by sub-topic

                    if search_engine == SearchEngine.BING_GROUNDING:
                        # Use grounding plugin for BING_GROUNDING (keep existing flat approach for grounding)
                        logger.info("Using GroundingPlugin for BING_GROUNDING search")

                        text_appended_query = f"{LOCALE_MSG['searching']}...<br>"
                        for i, query in enumerate(search_queries):
                            text_appended_query += (
                                f"{i}: {LOCALE_MSG['search_keyword']}: {query} <br>"
                            )

                        if stream:
                            yield f"data: ### {text_appended_query}\n\n"

                        grounding_function = self.kernel.get_function(
                            "grounding", "grounding_search_multi_query"
                        )

                        # Convert search_queries list to JSON string for the plugin
                        search_queries_json = json.dumps(search_queries)

                        grounding_result = await grounding_function.invoke(
                            self.kernel,
                            KernelArguments(
                                search_queries=search_queries_json,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                locale=locale,
                            ),
                        )

                        if grounding_result and grounding_result.value:
                            all_contexts.append(
                                f"=== Grounding Search ===\n{grounding_result.value}"
                            )
                            logger.info("Successfully got grounding search results")
                        else:
                            logger.warning("No grounding search results obtained")

                    elif (
                        search_engine == SearchEngine.BING_SEARCH_CRAWLING
                        or search_engine == SearchEngine.BING_GROUNDING_CRAWLING
                    ):
                        # Process each sub-topic for web search
                        logger.info(
                            f"Using SearchPlugin for {search_engine} search by sub-topic"
                        )
                        search_function = self.kernel.get_function(
                            "search", "search_single_query"
                        )

                        for sub_topic_idx, sub_topic_data in enumerate(sub_topics):
                            sub_topic_name = sub_topic_data["sub_topic"]
                            sub_topic_queries = sub_topic_data["queries"]

                            if stream:
                                yield f"data: ### {LOCALE_MSG['searching']} - {sub_topic_name} ({sub_topic_idx + 1}/{len(sub_topics)})\n\n"

                            sub_topic_results = []

                            # Search with each query in this sub-topic
                            for query in sub_topic_queries:
                                logger.info(
                                    f"Web search for sub-topic '{sub_topic_name}' with query: {query}"
                                )

                                search_result = await search_function.invoke(
                                    self.kernel,
                                    KernelArguments(
                                        query=str(query),
                                        locale=locale,
                                        max_results=3,  # Reduced per query
                                        max_context_length=3000,  # Reduced context length
                                    ),
                                )

                                if search_result and search_result.value:
                                    sub_topic_results.append(search_result.value)
                                    logger.info(
                                        f"Added web search result for sub-topic '{sub_topic_name}', query: {query}"
                                    )
                                else:
                                    logger.warning(
                                        f"No web search result for sub-topic '{sub_topic_name}', query: {query}"
                                    )

                            # Store results for this sub-topic
                            if sub_topic_results:
                                sub_topic_web_contexts[sub_topic_name] = "\n\n".join(
                                    sub_topic_results
                                )
                                logger.info(
                                    f"Added {len(sub_topic_results)} web search results for sub-topic: {sub_topic_name}"
                                )

                        # Combine all sub-topic web search results
                        if sub_topic_web_contexts:
                            combined_web_context = ""
                            for (
                                sub_topic_name,
                                context,
                            ) in sub_topic_web_contexts.items():
                                combined_web_context += (
                                    f"=== {sub_topic_name} ===\n{context}\n\n"
                                )

                            all_contexts.append(
                                f"=== Web Search ===\n{combined_web_context}"
                            )
                            logger.info(
                                f"Final web search: {len(sub_topic_web_contexts)} sub-topics processed"
                            )

                    if verbose and stream and sub_topic_web_contexts:
                        truncated_for_display = (
                            str(sub_topic_web_contexts)[:200]
                            + "... [truncated for display]"
                        )
                        yield f"data: {self.send_step_with_code(LOCALE_MSG['search_done'], truncated_for_display)}\n\n"

                except Exception as e:
                    logger.error(f"Error during web search by sub-topic: {str(e)}")
                    if stream:
                        yield f"data: ### Error during web search: {str(e)}\n\n"

            # YouTube search context by sub-topic (NEW APPROACH - SIMPLE)
            if include_ytb_search and sub_topics:
                try:
                    sub_topic_youtube_contexts = {}  # Store results by sub-topic

                    # Process each sub-topic for YouTube search
                    for sub_topic_idx, sub_topic_data in enumerate(sub_topics):
                        sub_topic_name = sub_topic_data["sub_topic"]
                        sub_topic_queries = sub_topic_data["queries"]

                        if stream:
                            yield f"data: ### {LOCALE_MSG['searching_YouTube']} - {sub_topic_name} ({sub_topic_idx + 1}/{len(sub_topics)})\n\n"

                        sub_topic_results = []

                        # Search with each query in this sub_topic
                        for query in sub_topic_queries:
                            # Use kernel to invoke youtube / youtube_mcp plugin
                            if include_mcp_server:
                                youtube_search_function = self.kernel.get_function(
                                    "youtube_mcp", "search_youtube_videos"
                                )
                            else:
                                youtube_search_function = self.kernel.get_function(
                                    "youtube", "search_youtube_videos"
                                )

                            mcp_args = KernelArguments()
                            mcp_args["query"] = str(query)

                            mcp_result = await youtube_search_function.invoke(
                                self.kernel, mcp_args
                            )

                            if mcp_result and mcp_result.value:
                                sub_topic_results.append(mcp_result.value)
                                logger.info(
                                    f"Added YouTube search result for sub-topic '{sub_topic_name}', query: {query}"
                                )
                            else:
                                logger.warning(
                                    f"No YouTube search result for sub-topic '{sub_topic_name}', query: {query}"
                                )

                        # Store results for this sub-topic
                        if sub_topic_results:
                            sub_topic_youtube_contexts[sub_topic_name] = "\n\n".join(
                                sub_topic_results
                            )
                            logger.info(
                                f"Added {len(sub_topic_results)} YouTube search results for sub-topic: {sub_topic_name}"
                            )

                    # Combine all sub-topic YouTube search results
                    if sub_topic_youtube_contexts:
                        combined_youtube_context = ""
                        for (
                            sub_topic_name,
                            context,
                        ) in sub_topic_youtube_contexts.items():
                            combined_youtube_context += (
                                f"=== {sub_topic_name} ===\n{context}\n\n"
                            )

                        all_contexts.append(
                            f"=== Youtube Search ===\n{combined_youtube_context}"
                        )
                        logger.info(
                            f"Final YouTube search: {len(sub_topic_youtube_contexts)} sub-topics processed"
                        )

                    if verbose and stream and sub_topic_youtube_contexts:
                        truncated_for_display = (
                            str(sub_topic_youtube_contexts)[:200]
                            + "... [truncated for display]"
                        )
                        yield f"data: {self.send_step_with_code(LOCALE_MSG['YouTube_done'], truncated_for_display)}\n\n"

                except Exception as e:
                    logger.error(f"Error during YouTube search by sub-topic: {str(e)}")
                    if stream:
                        yield f"data: ### Error during YouTube search: {str(e)}\n\n"

            # AI search context by sub-topic (EXISTING APPROACH - KEEP AS IS)
            if include_ai_search and sub_topics:
                try:
                    sub_topic_ai_search_contexts = {}  # Store results by sub-topic
                    seen_documents = set()  # 중복 문서 방지용
                    MAX_CONTEXT_LENGTH = 400000  # 40만자로 제한
                    MAX_DOCUMENT_LENGTH = 10000  # 문서당 10000자로 제한
                    current_total_length = 0

                    # Process each sub-topic
                    for sub_topic_idx, sub_topic_data in enumerate(sub_topics):
                        sub_topic_name = sub_topic_data["sub_topic"]
                        sub_topic_queries = sub_topic_data["queries"]

                        if stream:
                            yield f"data: ### {LOCALE_MSG['searching_ai_search']} - {sub_topic_name} ({sub_topic_idx + 1}/{len(sub_topics)})\n\n"

                        sub_topic_results = []

                        # Search with each query in this sub-topic
                        for query in sub_topic_queries:
                            ai_search_function = self.kernel.get_function(
                                "ai_search", "search_documents"
                            )
                            ai_search_result = await ai_search_function.invoke(
                                self.kernel,
                                KernelArguments(
                                    query=str(query),
                                    top_k=3,
                                    include_content=True,
                                ),
                            )

                            if ai_search_result and ai_search_result.value:
                                search_data = (
                                    json.loads(ai_search_result.value)
                                    if isinstance(ai_search_result.value, str)
                                    else ai_search_result.value
                                )

                                if search_data.get(
                                    "status"
                                ) == "success" and search_data.get("documents"):
                                    documents = search_data["documents"]

                                    for doc in documents[:5]:  # 각 쿼리당 5개 문서만
                                        doc_id = (
                                            doc.get("id")
                                            or doc.get("title")
                                            or doc.get("url", f"doc_{query}")
                                        )

                                        if doc_id in seen_documents:
                                            continue

                                        seen_documents.add(doc_id)

                                        if "content" in doc and doc["content"]:
                                            content = doc["content"]
                                            if len(content) > MAX_DOCUMENT_LENGTH:
                                                content = (
                                                    content[:MAX_DOCUMENT_LENGTH]
                                                    + "... [truncated]"
                                                )

                                            if (
                                                current_total_length + len(content)
                                                <= MAX_CONTEXT_LENGTH
                                            ):
                                                sub_topic_results.append(content)
                                                current_total_length += len(content)
                                            else:
                                                break

                                        if current_total_length >= MAX_CONTEXT_LENGTH:
                                            break

                            if current_total_length >= MAX_CONTEXT_LENGTH:
                                break

                        # Store results for this sub-topic
                        if sub_topic_results:
                            sub_topic_ai_search_contexts[sub_topic_name] = "\n\n".join(
                                sub_topic_results
                            )
                            logger.info(
                                f"Added {len(sub_topic_results)} AI search documents for sub-topic: {sub_topic_name}"
                            )

                        if current_total_length >= MAX_CONTEXT_LENGTH:
                            logger.warning(
                                f"Reached maximum context length, stopping at sub-topic: {sub_topic_name}"
                            )
                            break

                    # Combine all sub-topic AI search results
                    if sub_topic_ai_search_contexts:
                        combined_ai_context = ""
                        for (
                            sub_topic_name,
                            context,
                        ) in sub_topic_ai_search_contexts.items():
                            combined_ai_context += (
                                f"=== {sub_topic_name} ===\n{context}\n\n"
                            )

                        all_contexts.append(
                            f"=== Document Context ===\n{combined_ai_context}"
                        )
                        logger.info(
                            f"Final AI search: {len(sub_topic_ai_search_contexts)} sub-topics, {current_total_length} total chars"
                        )

                    if verbose and stream and sub_topic_ai_search_contexts:
                        truncated_for_display = (
                            str(sub_topic_ai_search_contexts)[:200]
                            + "... [truncated for display]"
                        )
                        yield f"data: {self.send_step_with_code(LOCALE_MSG['ai_search_context_done'], truncated_for_display)}\n\n"

                except Exception as e:
                    logger.error(f"Error during AI search by sub-topic: {str(e)}")
                    if stream:
                        yield f"data: ### Error processing AI search: {str(e)}\n\n"

            if stream:
                yield f"data: ### {LOCALE_MSG['answering']}\n\n"

            if not all_contexts:
                all_contexts.append("No relevant context found.")

            contexts_text = "\n".join(all_contexts)

            # Generate final answer
            if research and user_intent == "research":
                # Use group chat for research intent
                try:
                    yield f"data: ### {LOCALE_MSG['start_research']}\n"

                    # Process each sub-topic SEPARATELY with its own context
                    for sub_topic_idx, sub_topic_data in enumerate(sub_topics):
                        sub_topic_name = sub_topic_data["sub_topic"]
                        sub_topic_queries = sub_topic_data["queries"]

                        # Extract ONLY this sub-topic's context
                        sub_topic_context = ""
                        sub_topic_group_chat_result = None

                        # Web search context for this sub-topic only
                        if (
                            "sub_topic_web_contexts" in locals()
                            and sub_topic_name in sub_topic_web_contexts
                        ):
                            sub_topic_context += f"=== Web Search Results ===\n{sub_topic_web_contexts[sub_topic_name]}\n\n"

                        # YouTube search context for this sub-topic only
                        if (
                            "sub_topic_youtube_contexts" in locals()
                            and sub_topic_name in sub_topic_youtube_contexts
                        ):
                            sub_topic_context += f"=== YouTube Search Results ===\n{sub_topic_youtube_contexts[sub_topic_name]}\n\n"

                        # AI search context for this sub-topic only
                        if (
                            "sub_topic_ai_search_contexts" in locals()
                            and sub_topic_name in sub_topic_ai_search_contexts
                        ):
                            sub_topic_context += f"=== Document Context ===\n{sub_topic_ai_search_contexts[sub_topic_name]}\n\n"

                        # If no specific context, use fallback
                        if not sub_topic_context.strip():
                            sub_topic_context = (
                                "No specific context available for this sub-topic."
                            )

                        sub_topic_group_chat_result_str = None
                        yield f"data: ### {LOCALE_MSG['organize_research']} for {sub_topic_name} \n"

                        if multi_agent_type == "Semantic Kernel GroupChat":
                            # Execute group chat with ONLY this sub-topic's context
                            group_chat_function = self.kernel.get_function(
                                "sk_group_chat", "group_chat"
                            )
                            sub_topic_group_chat_result = await group_chat_function.invoke(
                                self.kernel,
                                KernelArguments(
                                    sub_topic=sub_topic_name,
                                    question=", ".join(
                                        sub_topic_queries
                                    ),  # Convert list to string
                                    sub_topic_contexts=sub_topic_context,  # Only this sub-topic's context
                                    locale=locale,
                                    max_rounds="1",
                                    max_tokens=str(40000),
                                    current_date=current_date,
                                ),
                            )
                            sub_topic_group_chat_result_str = (
                                sub_topic_group_chat_result.value
                            )
                        else:
                            # Prepare tasks using the plugin's method
                            tasks = self.multi_agent_plugin._normalize_tasks(
                                question=", ".join(sub_topic_queries),
                                sub_topic=sub_topic_name,
                                sub_topics=None,
                                sub_topic_contexts=sub_topic_context,
                                contexts=None,
                                max_tokens=40000,
                            )

                            # Call multi-agent plugin
                            multi_agent_function = self.kernel.get_function(
                                "vanilla_multi_agent", "run_multi_agent"
                            )
                            sub_topic_group_chat_result = (
                                await multi_agent_function.invoke(
                                    self.kernel,
                                    KernelArguments(
                                        sub_topic=sub_topic_name,
                                        question=", ".join(sub_topic_queries),
                                        sub_topic_contexts=sub_topic_context,
                                        locale=locale,
                                        max_rounds="4",
                                        max_tokens=str(40000),
                                        current_date=current_date,
                                    ),
                                )
                            )
                            sub_topic_group_chat_result_str = (
                                sub_topic_group_chat_result.value
                            )

                        # Parse and stream results
                        if sub_topic_group_chat_result_str:
                            try:
                                group_chat_data = json.loads(sub_topic_group_chat_result_str)
                                logger.info(
                                    f"Multi-agent result for {sub_topic_name}: status={group_chat_data.get('status')}"
                                )

                                # ✅ Check if this is Semantic Kernel GroupChat format (simple structure)
                                if multi_agent_type == "Semantic Kernel GroupChat":
                                    # SK GroupChat format: {"status": "success", "sub_topic": "...", "final_answer": "JSON string"}
                                    if group_chat_data.get("status") == "success":
                                        final_answer_json = group_chat_data.get("final_answer", "")
                                        
                                        # ✅ Parse final_answer JSON to extract markdown
                                        try:
                                            final_answer_data = json.loads(final_answer_json)
                                            # Extract answer_markdown (try revised first, then draft)
                                            answer_markdown = final_answer_data.get("revised_answer_markdown", "") or final_answer_data.get("draft_answer_markdown", "")
                                            if stream and answer_markdown:
                                                # Set TTFT
                                                if 'ttft_time' not in locals():
                                                    ttft_time = datetime.now(tz=self.timezone) - start_time
                                                
                                                yield f"\n"
                                                yield f"data: ### {LOCALE_MSG.get('write_research', 'Writing Answer')} for {sub_topic_name}\n\n"
                                                yield f"## {sub_topic_name}\n\n"
                                                
                                                # Stream answer in chunks
                                                chunk_size = 100
                                                for i in range(0, len(answer_markdown), chunk_size):
                                                    chunk = answer_markdown[i : i + chunk_size]
                                                    yield chunk
                                                    await asyncio.sleep(0.01)
                                                yield "\n\n"
                                            elif stream:
                                                # No markdown found, log warning
                                                logger.warning(f"No answer_markdown found in final_answer for {sub_topic_name}")
                                                yield f"data: ⚠️ No answer content generated for {sub_topic_name}\n\n"
                                                
                                        except json.JSONDecodeError as je:
                                            logger.error(f"Failed to parse final_answer JSON for {sub_topic_name}: {je}")
                                            logger.error(f"Raw final_answer: {final_answer_json[:200]}...")
                                            if stream:
                                                yield f"data: ❌ Error parsing answer format\n\n"
                                    else:
                                        error_msg = group_chat_data.get("error", "Unknown error")
                                        logger.error(f"SK GroupChat failed for {sub_topic_name}: {error_msg}")
                                        if stream:
                                            yield f"data: ❌ Error: {error_msg}\n\n"
                                    
                                    continue  # Skip vanilla multi-agent processing
                                
                                # ✅ Vanilla multi-agent format processing (existing code)
                                # Get sub-topic results for detailed info
                                sub_results = group_chat_data.get("sub_topic_results", [])

                                # Stream writer progress
                                if stream:
                                    for idx, result in enumerate(sub_results, 1):
                                        topic = result.get("sub_topic", "Unknown")
                                        writer_status = result.get("writer", {}).get(
                                            "status"
                                        )
                                        if writer_status == "success":
                                            yield f"data: ### ✅ Writer [{idx}/{len(sub_results)}]: {topic}\n\n"
                                        else:
                                            yield f"data: ### ❌ Writer [{idx}/{len(sub_results)}]: {topic} (failed)\n\n"

                                    
                                    # Stream reviewer progress
                                    for idx, result in enumerate(sub_results, 1):
                                        topic = result.get("sub_topic", "Unknown")
                                        review = result.get("review", {})
                                        review_status = review.get("status")

                                        if review_status == "skipped":
                                            yield f"data: ⏭️  Reviewer [{idx}/{len(sub_results)}]: {topic} (skipped - writer failed)\n\n"
                                        elif review_status == "success":
                                            ready = review.get(
                                                "ready_to_publish", False
                                            )
                                            score = review.get("parsed", {}).get(
                                                "reviewer_evaluation_score", "N/A"
                                            )
                                            status_icon = "✅" if ready else "⚠️"
                                            yield f"data: ### {status_icon} Reviewer [{idx}/{len(sub_results)}]: {topic} (score: {score})\n\n"
                                        else:
                                            yield f"data: ### ❌ Reviewer [{idx}/{len(sub_results)}]: {topic} (error)\n\n"

                                # Check overall status
                                all_ready = group_chat_data.get(
                                    "all_ready_to_publish", False
                                )
                        
                                # Stream final answers (always, regardless of ready_to_publish)
                                final_answer_str = group_chat_data.get(
                                    "final_answer", ""
                                )
                                if final_answer_str and stream:
                                    try:
                                        final_answer_data = json.loads(final_answer_str)
                                        answers = final_answer_data.get("answers", {})
                                        ttft_time = (
                                            datetime.now(tz=self.timezone) - start_time
                                        )

                                        for topic, answer in answers.items():
                                            # Get detailed info for this topic
                                            topic_result = next(
                                                (
                                                    r
                                                    for r in sub_results
                                                    if r.get("sub_topic") == topic
                                                ),
                                                None,
                                            )

                                            ready_icon = "✅"
                                            score = "N/A"
                                            if topic_result:
                                                ready_to_publish = topic_result.get(
                                                    "review", {}
                                                ).get("ready_to_publish", False)
                                                ready_icon = (
                                                    "✅" if ready_to_publish else "⚠️"
                                                )
                                                score = (
                                                    topic_result.get("review", {})
                                                    .get("parsed", {})
                                                    .get(
                                                        "reviewer_evaluation_score",
                                                        "N/A",
                                                    )
                                                )

                                            yield f"\n"
                                            yield f"data: ### {LOCALE_MSG.get('write_research', 'Writing Answer')} for {topic} {ready_icon} (Score: {score})\n\n"
                                            yield f"## {topic}\n\n"

                                            # Stream answer in chunks
                                            chunk_size = 100
                                            for i in range(0, len(answer), chunk_size):
                                                chunk = answer[i : i + chunk_size]
                                                yield chunk
                                                await asyncio.sleep(0.01)
                                            yield "\n\n"
                                    except json.JSONDecodeError as je:
                                        logger.error(
                                            f"Failed to parse final_answer for {sub_topic_name}: {je}"
                                        )
                                        if stream:
                                            yield f"data: ❌ Error parsing final answer\n\n"

                            except json.JSONDecodeError as e:
                                logger.error(
                                    f"Failed to parse multi-agent JSON for {sub_topic_name}: {e}"
                                )
                                if stream:
                                    yield f"data: ❌ Error parsing results for {sub_topic_name}\n\n"
                        else:
                            logger.warning(
                                f"No multi-agent result for {sub_topic_name}"
                            )
                            if stream:
                                yield f"data: ⚠️ No result generated for {sub_topic_name}\n\n"

                except Exception as e:
                    logger.error(f"Error during group chat for research: {str(e)}")
                    if stream:
                        yield f"data: ### Error during research analysis: {str(e)}"
            else:
                # General query processing
                answer_messages = [
                    {
                        "role": "system",
                        "content": GENERAL_ANSWER_PROMPT.format(
                            current_date=current_date,
                            contexts=contexts_text,
                            question=enriched_query,
                            locale=locale,
                        ),
                    },
                    {"role": "user", "content": enriched_query},
                ]

                response = await self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=answer_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream,
                )
                ttft_time = datetime.now(tz=self.timezone) - start_time

                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield f"{chunk.choices[0].delta.content}"

            yield "\n"  # clear previous md formatting

            if elapsed_time and ttft_time is not None:
                logger.info(
                    f"Doc research response generated successfully in {ttft_time.total_seconds()} seconds"
                )
                yield "\n"
                yield f"doc research response generated successfully in {ttft_time.total_seconds()} seconds \n"

        except Exception as e:
            error_msg = f"Doc research error: {str(e)}"
            logger.error(error_msg)
            yield f"Error: {str(e)}"

    async def cleanup(self):
        """Clean up resources"""
        try:
            # AzureMCPPlugin 정리
            if hasattr(self.youtube_plugin, "cleanup"):
                await self.youtube_plugin.cleanup()
            if hasattr(self.youtube_mcp_plugin, "cleanup"):
                await self.youtube_mcp_plugin.cleanup()

            # IntentPlugin 정리
            if hasattr(self.intent_plan_plugin, "cleanup"):
                await self.intent_plan_plugin.cleanup()

            # GroundingPlugin 정리
            if hasattr(self.grounding_plugin, "cleanup"):
                await self.grounding_plugin.cleanup()

            # GroupChattingPlugin 정리
            if hasattr(self.group_chatting_plugin, "cleanup"):
                await self.group_chatting_plugin.cleanup()

            # OpenAI 클라이언트 정리
            if hasattr(self.client, "close"):
                await self.client.close()

            # 잠시 대기하여 연결이 완전히 정리되도록 함
            await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")