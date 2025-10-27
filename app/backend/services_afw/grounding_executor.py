"""
Grounding Executor using Microsoft Agent Framework.

This executor performs grounding search using Azure AI Agents with Bing Grounding Tool.
Migrated from services_sk/grounding_plugin.py.
"""

import os
import json
import base64
import logging
import asyncio
import pytz
from datetime import datetime
from typing import Dict, Any, Optional

from agent_framework import Executor, WorkflowContext, handler
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.ai.agents.models import BingGroundingTool, MessageRole, RunStatus
from azure.ai.agents import AgentsClient
from langchain.prompts import PromptTemplate
from i18n.locale_msg import LOCALE_MESSAGES
from utils.yield_message import (
    send_step_with_code,
    send_step_with_input,
    send_step_with_code_and_input,
)


from config.config import Settings

logger = logging.getLogger(__name__)

# Prompt template for grounding search
SEARCH_GENERATE_PROMPT_TEMPLATE = """
You are an intelligent chatbot that provides guidance on various topics in **Markdown** format based on real-time web search results.

🎯 Objective:
- Provide users with accurate and reliable answers based on the latest web information.
- Actively utilize web search results to generate rich, detailed, and specific answers.
- Respond in Markdown format, including 1-2 emojis to increase readability and friendliness.

📌 Guidelines:  
1. don't response with any greeting messages, just response with the answer to the user's question.
2. Always generate answers based on search results and avoid making unfounded assumptions.  
3. Always include reference links and format them using the Markdown `[text](URL)` format.  
4. When providing product price information, base it on the official website's prices and links.
"""

SEARCH_GEN_PROMPT = PromptTemplate(
    template=SEARCH_GENERATE_PROMPT_TEMPLATE,
)


class GroundingExecutor(Executor):
    """
    Executor for performing grounding search using Azure AI Agents with Bing Grounding Tool.

    This executor provides conversational AI-powered search that:
    - Uses Bing Grounding Tool for real-time web information
    - Generates comprehensive answers with citations
    - Supports multiple search queries in one request
    """

    def __init__(
        self,
        id: str,
        settings: Optional[Settings] = None,
        project_endpoint: Optional[str] = None,
        connection_id: Optional[str] = None,
        agent_model_deployment_name: Optional[str] = None,
        max_results: int = 5,
        market: str = "ko-KR",
        set_lang: str = "ko",
    ):
        """
        Initialize the GroundingExecutor.

        Args:
            id: Executor ID
            settings: Settings object
            project_endpoint: Azure AI project endpoint
            connection_id: Bing Grounding connection ID
            agent_model_deployment_name: Agent model deployment name
            max_results: Maximum search results
            market: Market for search (e.g., ko-KR, en-US)
            set_lang: Language setting
        """
        super().__init__(id=id)

        self.settings = settings or Settings()

        # Set timezone
        if isinstance(self.settings.TIME_ZONE, str):
            self.timezone = pytz.timezone(self.settings.TIME_ZONE)
        else:
            self.timezone = pytz.UTC

        # Azure AI configuration
        self.project_endpoint = project_endpoint or os.getenv(
            "BING_GROUNDING_PROJECT_ENDPOINT"
        )
        self.connection_id = connection_id or os.getenv("BING_GROUNDING_CONNECTION_ID")
        self.agent_model_deployment_name = agent_model_deployment_name or os.getenv(
            "BING_GROUNDING_AGENT_MODEL_DEPLOYMENT_NAME"
        )
        self.max_results = max_results
        self.market = market
        self.set_lang = set_lang
        self.search_gen_agent_id_env = os.getenv("SEARCH_GEN_AGENT_ID")

        # Initialize credentials
        self.creds = self._get_azure_credential()

        # Initialize Azure AI Agents client
        self.agents_client = AgentsClient(
            endpoint=self.project_endpoint,
            credential=self.creds,
        )

        # Initialize Bing Grounding Tool
        self.bing_tool = BingGroundingTool(
            connection_id=self.connection_id,
            market=self.market,
            set_lang=self.set_lang,
            count=int(self.max_results),
        )

        # Initialize or get existing agent
        self.search_gen_agent = self._initialize_agent()

        logger.info(
            f"GroundingExecutor initialized with agent: {self.search_gen_agent.id}"
        )

    def _get_azure_credential(self):
        """Get appropriate Azure credential based on the environment."""
        try:
            if os.getenv("APP_USER_ASSIGNED_MANAGED_IDENTITY_CLIENT_ID"):
                logger.info("Using Managed Identity credential")
                return ManagedIdentityCredential(
                    client_id=os.getenv("APP_USER_ASSIGNED_MANAGED_IDENTITY_CLIENT_ID")
                )
            else:
                logger.info("Using DefaultAzureCredential")
                return DefaultAzureCredential()
        except Exception as e:
            logger.warning(f"Error initializing Azure credential: {str(e)}")
            logger.info("Falling back to DefaultAzureCredential")
            return DefaultAzureCredential()

    def _initialize_agent(self):
        """Initialize or get existing Azure AI Agent."""
        try:
            if self.search_gen_agent_id_env:
                logger.info(
                    f"Retrieving existing agent: {self.search_gen_agent_id_env}"
                )
                agent = self.agents_client.agents.get(self.search_gen_agent_id_env)
                logger.info(f"Retrieved agent: {agent.id}")
            else:
                logger.info("Creating new agent with Bing Grounding Tool")
                agent = self.agents_client.agents.create(
                    model=self.agent_model_deployment_name,
                    name="Search Generation Agent",
                    instructions=SEARCH_GEN_PROMPT.template,
                    tools=[self.bing_tool.definition],
                )
                logger.info(f"Created new agent: {agent.id}")

            return agent
        except Exception as e:
            logger.error(f"Error initializing agent: {str(e)}")
            raise

    @handler
    async def grounding_search(
        self,
        search_data: Dict[str, Any],
        ctx: WorkflowContext[Dict[str, Any], str],  # Added str for yield_output
    ) -> None:
        """
        Perform grounding search using multiple queries.

        Args:
            search_data: Dictionary with search parameters:
                - search_queries: JSON string or list of search queries
                - max_tokens: Maximum tokens for response (default: 1024)
                - temperature: Temperature for generation (default: 0.7)
                - locale: Locale for search and response (default: ko-KR)
            ctx: Workflow context for sending results
        """
        try:
            # Get metadata for verbose and locale
            metadata = search_data.get("metadata", {})
            locale = metadata.get("locale", "ko-KR")
            verbose = metadata.get("verbose", False)
            LOCALE_MSG = LOCALE_MESSAGES.get(locale, LOCALE_MESSAGES["ko-KR"])

            search_queries = search_data.get("search_queries", "")
            max_tokens = search_data.get("max_tokens", 1024)
            temperature = search_data.get("temperature", 0.7)

            # ✅ Yield starting message
            await ctx.yield_output(f"data: ### {LOCALE_MSG['searching']}\n\n")

            logger.info(f"[GroundingExecutor] Starting grounding search")
            logger.info(f"[GroundingExecutor] Queries: {search_queries}")

            # Parse search queries if JSON string
            if isinstance(search_queries, str):
                try:
                    queries_data = json.loads(search_queries)
                    if isinstance(queries_data, list):
                        # Extract queries from list of dicts or use directly
                        if queries_data and isinstance(queries_data[0], dict):
                            queries = [
                                q.get("query", q.get("search_query", ""))
                                for q in queries_data
                            ]
                        else:
                            queries = queries_data
                        search_queries_str = ", ".join(queries)
                    else:
                        search_queries_str = search_queries
                except json.JSONDecodeError:
                    search_queries_str = search_queries
            else:
                search_queries_str = search_queries

            # Execute grounding search
            result = await self._execute_grounding_search(
                search_queries=search_queries_str,
                max_tokens=max_tokens,
                temperature=temperature,
                locale=locale,
            )

            result_data = {
                "search_queries": search_queries_str,
                "result": result,
                "locale": locale,
            }

            logger.info(f"[GroundingExecutor] Search completed successfully")

            # ✅ Yield completion message (SK compatible format with results)
            if verbose and result:
                results_str = json.dumps(result_data, ensure_ascii=False, indent=2)
                truncated = (
                    results_str[:200] + "... [truncated for display]"
                    if len(results_str) > 200
                    else results_str
                )
                encoded_code = base64.b64encode(
                    f"```json\n{truncated}\n```".encode("utf-8")
                ).decode("utf-8")
                await ctx.yield_output(
                    f"data: {send_step_with_code(LOCALE_MSG['search_done'], encoded_code)}\n\n"
                )
            else:
                await ctx.yield_output(f"data: ### {LOCALE_MSG['search_done']}\n\n")

            # Send results to next executor
            await ctx.send_message({**search_data, "grounding_results": result_data})

        except Exception as e:
            error_msg = f"Grounding search failed: {str(e)}"
            logger.error(f"[GroundingExecutor] {error_msg}")
            await ctx.send_message(
                {
                    **search_data,
                    "grounding_results": {
                        "error": error_msg,
                        "search_queries": search_data.get("search_queries", ""),
                        "result": "",
                    },
                }
            )

    async def _execute_grounding_search(
        self,
        search_queries: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        locale: str = "ko-KR",
    ) -> str:
        """Execute the actual grounding search using Azure AI Agent."""
        thread = None
        try:
            # Create thread
            thread = self.agents_client.threads.create()
            logger.info(f"Created thread, ID: {thread.id}")

            # Define the user prompt template
            SEARCH_GEN_USER_PROMPT_TEMPLATE = """
please provide as rich and specific an answer and reference links as possible for the following search queries: {search_keyword}
Today is {current_date}. Results should be based on the recent information available. 

If multiple queries are provided, provide rich, detailed search results for each query and clearly separate them.
Format the response with clear section headers for each query.
Include relevant reference links in Markdown format [text](URL).

return the answer in the following format:
For each query, provide:
- Query: [the search query]
- Answer: [comprehensive answer with references]
- References: [list of relevant links]
            """

            # Create prompt template
            SEARCH_GEN_USER_PROMPT = PromptTemplate(
                template=SEARCH_GEN_USER_PROMPT_TEMPLATE,
                input_variables=["search_keyword", "current_date", "max_results"],
            )

            current_date = datetime.now(tz=self.timezone).strftime("%Y-%m-%d")

            search_gen_instruction = SEARCH_GEN_USER_PROMPT.format(
                search_keyword=search_queries,
                current_date=current_date,
                max_results=self.max_results,
            )

            # Create message to thread
            logger.info(f"Final user instruction: {search_gen_instruction}")
            message = self.agents_client.messages.create(
                thread_id=thread.id,
                role=MessageRole.USER,
                content=search_gen_instruction,
            )
            logger.info(f"Created message, ID: {message.id}")

            # Create and poll run with exponential backoff
            run = self.agents_client.runs.create(
                thread_id=thread.id,
                agent_id=self.search_gen_agent.id,
            )
            logger.info(f"Created run, ID: {run.id}")

            # Enhanced polling with exponential backoff
            max_wait_time = 120  # 2 minutes
            poll_interval = 1
            max_poll_interval = 10
            elapsed = 0

            while elapsed < max_wait_time:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

                run = self.agents_client.runs.get(thread_id=thread.id, run_id=run.id)
                logger.info(f"Run status: {run.status} (elapsed: {elapsed}s)")

                if run.status == RunStatus.COMPLETED:
                    logger.info("Run completed successfully")
                    break
                elif run.status in [
                    RunStatus.FAILED,
                    RunStatus.CANCELLED,
                    RunStatus.EXPIRED,
                ]:
                    error_msg = f"Run ended with status: {run.status}"
                    if hasattr(run, "last_error") and run.last_error:
                        error_msg += f", error: {run.last_error}"
                    logger.error(error_msg)
                    return f"Error: {error_msg}"

                # Exponential backoff
                poll_interval = min(poll_interval * 1.5, max_poll_interval)

            if run.status != RunStatus.COMPLETED:
                error_msg = f"Run did not complete within {max_wait_time}s (status: {run.status})"
                logger.error(error_msg)
                return f"Error: {error_msg}"

            # Get messages
            messages = self.agents_client.messages.list(thread_id=thread.id)
            logger.info(f"Retrieved {len(messages.data)} messages")

            # Extract assistant messages
            assistant_messages = [
                msg for msg in messages.data if msg.role == MessageRole.ASSISTANT
            ]

            if not assistant_messages:
                return "No response generated"

            # Get the latest assistant message
            latest_message = assistant_messages[0]

            # Extract text content
            if hasattr(latest_message, "content") and latest_message.content:
                content_parts = []
                for content in latest_message.content:
                    if hasattr(content, "text") and content.text:
                        if hasattr(content.text, "value"):
                            content_parts.append(content.text.value)
                        else:
                            content_parts.append(str(content.text))

                result = "\n\n".join(content_parts)
                logger.info(f"Extracted response ({len(result)} chars)")
                return result

            return "No content in response"

        except asyncio.TimeoutError:
            error_msg = "Grounding search timed out"
            logger.error(error_msg)
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"Grounding search error: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"
        finally:
            # Clean up thread
            if thread:
                try:
                    self.agents_client.threads.delete(thread.id)
                    logger.info(f"Deleted thread: {thread.id}")
                except Exception as e:
                    logger.warning(f"Failed to delete thread: {e}")

    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, "search_gen_agent") and self.search_gen_agent:
                # Only delete if we created it (not if retrieved from env)
                if not self.search_gen_agent_id_env:
                    self.agents_client.agents.delete(self.search_gen_agent.id)
                    logger.info(f"Deleted agent: {self.search_gen_agent.id}")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
