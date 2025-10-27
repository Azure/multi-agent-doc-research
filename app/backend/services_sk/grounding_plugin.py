import logging
import json
import os
import asyncio
import pytz
from datetime import datetime
from semantic_kernel.functions import kernel_function
from langchain.prompts import PromptTemplate
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.ai.agents.models import BingGroundingTool, MessageRole
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import RunStatus
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


class GroundingPlugin:
    """
    A plugin for performing grounding search using Azure AI Agents with Bing Grounding Tool.
    This plugin is designed to work with Semantic Kernel and includes all necessary Azure AI functionality.
    """

    def __init__(self):
        """Initialize the GroundingPlugin with Azure AI Agent configuration."""
        self.settings = Settings()

        # Set timezone
        if isinstance(self.settings.TIME_ZONE, str):
            self.timezone = pytz.timezone(self.settings.TIME_ZONE)
        else:
            self.timezone = pytz.UTC

        # Azure AI configuration
        self.project_endpoint = os.getenv("BING_GROUNDING_PROJECT_ENDPOINT")
        self.connection_id = os.getenv("BING_GROUNDING_CONNECTION_ID")
        self.agent_model_deployment_name = os.getenv(
            "BING_GROUNDING_AGENT_MODEL_DEPLOYMENT_NAME"
        )
        self.max_results = int(os.getenv("BING_GROUNDING_MAX_RESULTS", 5))
        self.market = os.getenv("BING_GROUNDING_MARKET", "ko-KR")
        self.set_lang = os.getenv("BING_GROUNDING_SET_LANG", "ko")
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

        logger.info("GroundingPlugin initialized successfully")

    def _get_azure_credential(self):
        """Get appropriate Azure credential based on the environment."""
        try:
            if os.getenv("APP_USER_ASSIGNED_MANAGED_IDENTITY_CLIENT_ID"):
                logger.info(
                    "Detected Azure environment, using ManagedIdentityCredential"
                )
                client_id = os.getenv("APP_USER_ASSIGNED_MANAGED_IDENTITY_CLIENT_ID")
                if client_id:
                    logger.info(
                        f"Using user-assigned managed identity with client ID: {client_id}"
                    )
                    return ManagedIdentityCredential(client_id=client_id)
                else:
                    logger.info("Using system-assigned managed identity")
                    return ManagedIdentityCredential()
            else:
                logger.info("Using DefaultAzureCredential for local development")
                return DefaultAzureCredential()

        except Exception as e:
            logger.warning(f"Error initializing Azure credential: {str(e)}")
            logger.info("Falling back to DefaultAzureCredential")
            return DefaultAzureCredential()

    def _initialize_agent(self):
        """Initialize or get existing Azure AI Agent."""
        try:
            if self.search_gen_agent_id_env:
                # Try to get existing agent
                agent = self.agents_client.get_agent(self.search_gen_agent_id_env)
                logger.info(f"Using existing search-gen-agent, ID: {agent.id}")

                if agent is None:
                    logger.error("search_gen_agent is None, creating new agent")
                    agent = self.agents_client.create_agent(
                        model=self.agent_model_deployment_name,
                        name="grounding-search-agent",
                        instructions=SEARCH_GEN_PROMPT.format(),
                        tools=self.bing_tool.definitions,
                    )
                else:
                    # Update existing agent
                    self.agents_client.update_agent(
                        agent.id,
                        model=self.agent_model_deployment_name,
                        name="grounding-search-agent",
                        instructions=SEARCH_GEN_PROMPT.format(),
                        tools=self.bing_tool.definitions,
                    )
            else:
                # Create new agent
                agent = self.agents_client.create_agent(
                    model=self.agent_model_deployment_name,
                    name="grounding-search-agent",
                    instructions=SEARCH_GEN_PROMPT.format(),
                    tools=self.bing_tool.definitions,
                )
                logger.info(f"Created new search-gen-agent, ID: {agent.id}")

            return agent

        except Exception as e:
            logger.error(f"Error initializing agent: {str(e)}")
            raise

    @kernel_function(
        description="Perform grounding search using multiple queries to get comprehensive results",
        name="grounding_search_multi_query",
    )
    async def grounding_search_multi_query(
        self,
        search_queries: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        locale: str = "ko-KR",
    ) -> str:
        """
        Perform grounding search using multiple queries to get comprehensive results.

        Args:
            search_queries: JSON string containing list of search queries or search query info
            max_tokens: Maximum tokens for response generation
            temperature: Temperature for response generation
            locale: Locale for search and response

        Returns:
            Combined search results from all queries
        """
        try:
            logger.info(f"Starting grounding search with queries: {search_queries}")

            if not self.creds or not self.project_endpoint:
                logger.error("Bing Grounding credentials are missing")
                return "Error: Azure credentials are not properly configured"

            # Parse search queries
            if isinstance(search_queries, str):
                try:
                    # Try to parse as JSON first
                    parsed_queries = json.loads(search_queries)
                    if isinstance(parsed_queries, list):
                        queries_list = parsed_queries
                    elif isinstance(parsed_queries, dict):
                        # If it's a dict, look for common keys
                        if "search_queries" in parsed_queries:
                            queries_list = parsed_queries["search_queries"]
                        elif "queries" in parsed_queries:
                            queries_list = parsed_queries["queries"]
                        else:
                            # Use the values or fallback to the string itself
                            queries_list = (
                                list(parsed_queries.values())
                                if parsed_queries
                                else [search_queries]
                            )
                    else:
                        queries_list = [str(parsed_queries)]
                except json.JSONDecodeError:
                    # If not JSON, treat as a single query
                    queries_list = [search_queries]
            elif isinstance(search_queries, list):
                queries_list = search_queries
            else:
                queries_list = [str(search_queries)]

            logger.info(f"Parsed queries list: {queries_list}")

            # Format the queries for the agent
            if len(queries_list) == 1:
                formatted_queries = queries_list[0]
            else:
                # Format multiple queries as a numbered list
                formatted_queries = "\n".join(
                    [f"{i+1}. {query}" for i, query in enumerate(queries_list)]
                )

            logger.info(f"Formatted queries for agent: {formatted_queries}")

            # Execute the grounding search
            result = await self._execute_grounding_search(
                search_queries=formatted_queries,
                max_tokens=max_tokens,
                temperature=temperature,
                locale=locale,
            )

            return result

        except Exception as e:
            error_msg = f"Error in grounding_search_multi_query: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

    async def _execute_grounding_search(
        self,
        search_queries: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        locale: str = "ko-KR",
    ) -> str:
        """Execute the actual grounding search using Azure AI Agent with improved error handling."""

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
            logger.info(
                f"Final user instruction on Azure AI Agent: {search_gen_instruction}"
            )
            message = self.agents_client.messages.create(
                thread_id=thread.id,
                role=MessageRole.USER,
                content=search_gen_instruction,
            )
            logger.info(f"Created message, ID: {message.id}")

            # Enhanced polling function with exponential backoff
            async def get_agent_message_with_enhanced_retry(
                thread_id, role, max_retries=20, initial_delay=0.3
            ):
                """Enhanced polling with exponential backoff and longer timeout."""
                delay = initial_delay
                for attempt in range(max_retries):
                    try:
                        response_message = (
                            self.agents_client.messages.get_last_message_by_role(
                                thread_id=thread_id,
                                role=role,
                            )
                        )
                        if response_message is not None and getattr(
                            response_message, "content", None
                        ):
                            logger.info(f"Agent message found on attempt {attempt+1}")
                            return response_message

                        # Exponential backoff with jitter
                        actual_delay = delay * (
                            1 + 0.1 * (attempt % 3)
                        )  # Add some jitter
                        logger.info(
                            f"Agent message not found (attempt {attempt+1}/{max_retries}), retrying after {actual_delay:.2f}s..."
                        )
                        await asyncio.sleep(actual_delay)

                        # Exponential backoff up to 3 seconds
                        delay = min(delay * 1.5, 3.0)

                    except Exception as e:
                        logger.warning(
                            f"Error checking for agent message (attempt {attempt+1}): {e}"
                        )
                        await asyncio.sleep(delay)

                return None

            # Create run with better error handling
            logger.info("Creating agent run in thread with tools")
            run = self.agents_client.runs.create(
                thread_id=thread.id, agent_id=self.search_gen_agent.id
            )
            logger.info(f"Created run, ID: {run.id}")

            # Poll run status with timeout
            max_run_time = 60  # 60 seconds timeout
            start_time = datetime.now()

            while True:
                run_status = self.agents_client.runs.get(
                    thread_id=thread.id, run_id=run.id
                )

                logger.info(f"Run status: {run_status.status}")

                if run_status.status == RunStatus.COMPLETED:
                    logger.info("Run completed successfully")
                    break
                elif run_status.status == RunStatus.FAILED:
                    error_msg = f"Run failed: {run_status.last_error}"
                    logger.error(error_msg)
                    return f"Error: {error_msg}"
                elif run_status.status in [RunStatus.CANCELLED, RunStatus.EXPIRED]:
                    error_msg = f"Run {run_status.status.lower()}"
                    logger.error(error_msg)
                    return f"Error: {error_msg}"
                elif run_status.status in [
                    RunStatus.QUEUED,
                    RunStatus.IN_PROGRESS,
                    RunStatus.REQUIRES_ACTION,
                ]:
                    # Check timeout
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed > max_run_time:
                        logger.error(f"Run timeout after {elapsed:.1f} seconds")
                        # Try to cancel the run
                        try:
                            self.agents_client.runs.cancel(
                                thread_id=thread.id, run_id=run.id
                            )
                        except:
                            pass
                        return "Error: Request timeout. Please try again with a simpler query."

                    # Handle requires_action status
                    if run_status.status == RunStatus.REQUIRES_ACTION:
                        logger.info("Run requires action, processing tool calls...")
                        # Handle tool calls if needed

                    await asyncio.sleep(1.0)  # Wait before next status check
                else:
                    logger.warning(f"Unexpected run status: {run_status.status}")
                    await asyncio.sleep(1.0)

            # Get the agent's response message with enhanced retry
            response_message = await get_agent_message_with_enhanced_retry(
                thread_id=thread.id,
                role=MessageRole.AGENT,
                max_retries=20,
                initial_delay=0.3,
            )

            if response_message is None or not getattr(
                response_message, "content", None
            ):
                logger.error("Agent response message not found after enhanced retries.")
                return (
                    "Error: No response received from search service. Please try again."
                )

            logger.info(
                f"Agent response message received: {len(str(response_message.content))} characters"
            )

            # Extract text content from response
            result_text = ""
            if response_message.content:
                for content_item in response_message.content:
                    if hasattr(content_item, "type") and content_item.type == "text":
                        if hasattr(content_item, "text") and hasattr(
                            content_item.text, "value"
                        ):
                            result_text += content_item.text.value
                    elif (
                        isinstance(content_item, dict)
                        and content_item.get("type") == "text"
                    ):
                        text_content = content_item.get("text", {}).get("value", "")
                        result_text += text_content

            if not result_text:
                logger.warning("No text content found in response")
                return "Error: Empty response received. Please try again."

            logger.info(f"Extracted result text: {len(result_text)} characters")
            return result_text

        except asyncio.TimeoutError:
            logger.error("Timeout occurred during grounding search")
            return "Error: Request timeout. Please try again."
        except Exception as e:
            logger.error(f"Error during grounding search execution: {str(e)}")
            return f"Error during search execution: {str(e)}"
        finally:
            # Clean up thread
            if thread:
                try:
                    self.agents_client.threads.delete(thread.id)
                    logger.info(f"Cleaned up thread: {thread.id}")
                except Exception as e:
                    logger.warning(f"Error cleaning up thread: {e}")

    def delete_agent(self):
        """Delete the Azure AI Agent."""
        try:
            if self.search_gen_agent:
                self.agents_client.delete_agent(self.search_gen_agent.id)
                logger.info(f"Deleted search-gen-agent, ID: {self.search_gen_agent.id}")
        except Exception as e:
            logger.error(f"Error deleting agent: {str(e)}")

    async def cleanup(self):
        """Clean up resources."""
        try:
            self.delete_agent()
            logger.info("GroundingPlugin cleanup completed")
        except Exception as e:
            logger.error(f"Error during GroundingPlugin cleanup: {e}")
