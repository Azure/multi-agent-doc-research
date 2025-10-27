"""
Intent Analysis Plugin for Semantic Kernel
Provides intent analysis and search planning capabilities as kernel functions.
"""

import os
import json
import logging
import pytz
from datetime import datetime

from semantic_kernel.functions import kernel_function
from openai import AsyncAzureOpenAI
from config.config import Settings
from langchain.prompts import load_prompt

logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
# Load prompts
INTENT_ANALYZE_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "intent_analyze_prompt.yaml"),
    encoding="utf-8",
)
RESEARCH_PLAN_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "research_planner_prompt.yaml"),
    encoding="utf-8",
)
GENERAL_PLAN_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "general_planner_prompt.yaml"),
    encoding="utf-8",
)


class IntentPlanPlugin:
    """
    Semantic Kernel plugin for intent analysis and search planning.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the IntentPlugin.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.client = AsyncAzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        )

        self.deployment_name = settings.AZURE_OPENAI_DEPLOYMENT_NAME
        self.query_deployment_name = settings.AZURE_OPENAI_QUERY_DEPLOYMENT_NAME
        self.planner_max_plans = settings.PLANNER_MAX_PLANS
        self.max_tokens = settings.MAX_TOKENS

        if isinstance(settings.TIME_ZONE, str):
            self.timezone = pytz.timezone(settings.TIME_ZONE)
        else:
            self.timezone = pytz.UTC

    @kernel_function(
        description="Analyze user intent and rewrite query for better search",
        name="analyze_intent",
    )
    async def analyze_intent(
        self, original_query: str, locale: str = "en-US", temperature: float = 0.3
    ) -> str:
        """
        Analyze user intent and rewrite query accordingly.

        Args:
            original_query: The original user query
            locale: Locale for analysis (default: ko-KR)
            temperature: Temperature parameter for the LLM (0.0 to 1.0)

        Returns:
            JSON string with intent analysis results
        """
        try:
            current_date = datetime.now(tz=self.timezone).strftime("%Y-%m-%d")

            logger.info(
                f"Analyzing intent for query: {original_query} with locale: {locale}"
            )

            # Make the API call
            response = await self.client.chat.completions.create(
                model=self.query_deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": INTENT_ANALYZE_PROMPT.format(
                            current_date=current_date,
                            original_query=original_query,
                            locale=locale,
                        ),
                    },
                    {"role": "user", "content": original_query},
                ],
                temperature=0.2,
                max_tokens=500,
                response_format={"type": "json_object"},
            )

            # Parse and validate the response
            result = json.loads(response.choices[0].message.content.strip())
            logger.info(f"Intent analysis result: {result}")

            # Ensure required keys are present
            required_keys = ["user_intent", "enriched_query"]
            for key in required_keys:
                if key not in result:
                    raise ValueError(f"API response missing required key: {key}")

            # Validate intent value
            valid_intents = [
                "research",
                "general_query",
                "small_talk",
                "tool_calling",
            ]
            if result["user_intent"] not in valid_intents:
                logger.warning(
                    f"Invalid intent detected: {result['user_intent']}, defaulting to general_query"
                )
                result["user_intent"] = "general_query"

            # Set default values for optional fields
            result.setdefault("confidence", 0.8)
            result.setdefault("keywords", [])
            result.setdefault("target_info", "general information")

            return json.dumps(result)

        except Exception as e:
            logger.error(f"Error analyzing intent: {str(e)}")
            # Fallback to simple classification
            return self._fallback_intent_analysis(original_query, locale)

    @kernel_function(
        description="Generate search plan based on user intent",
        name="generate_search_plan",
    )
    async def generate_search_plan(
        self,
        user_intent: str,
        enriched_query: str,
        locale: str = "en-US",
        temperature: float = 0.7,
    ) -> str:
        """
        Generate search plan based on user intent.

        Args:
            user_intent: Detected user intent
            enriched_query: Enriched query from intent analysis
            locale: Locale for planning
            temperature: Temperature for planning

        Returns:
            JSON string with search plan
        """
        try:
            current_date = datetime.now(tz=self.timezone).strftime("%Y-%m-%d")

            logger.info(
                f"Generating search plan for intent: {user_intent} with query: {enriched_query} and locale: {locale}"
            )

            if user_intent == "research":
                planner_prompt = RESEARCH_PLAN_PROMPT.format(
                    planner_max_plans=self.planner_max_plans,
                    current_date=current_date,
                    enriched_query=enriched_query,
                    locale=locale,
                )

            else:
                # General query
                planner_prompt = GENERAL_PLAN_PROMPT.format(
                    planner_max_plans=self.planner_max_plans,
                    current_date=current_date,
                    enriched_query=enriched_query,
                    locale=locale,
                )

            logger.info(f"Generating search plan for intent: {user_intent}")

            response = await self.client.chat.completions.create(
                model=self.query_deployment_name,
                messages=[
                    {"role": "system", "content": planner_prompt},
                    {"role": "user", "content": enriched_query},
                ],
                temperature=temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content.strip())
            result["plan_type"] = user_intent

            logger.info(f"Search plan generated: {result}")
            return json.dumps(result)

        except Exception as e:
            logger.error(f"Error generating search plan: {str(e)}")
            # Fallback plan
            fallback_result = {"search_queries": [enriched_query]}

            return json.dumps(fallback_result)

    def _fallback_intent_analysis(self, query: str, locale: str = "en-US") -> str:
        """
        Fallback intent analysis using keyword matching.

        Args:
            query: User query
            locale: Locale for analysis

        Returns:
            JSON string with basic intent classification
        """
        # Default to general query
        result = {
            "user_intent": "general_query",
            "enriched_query": query,
            "confidence": 0.5,
            "keywords": [],
            "target_info": "general information",
        }
        return json.dumps(result)

    async def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self.client, "close"):
                await self.client.close()
        except Exception as e:
            logger.error(f"Error during IntentPlugin cleanup: {e}")
