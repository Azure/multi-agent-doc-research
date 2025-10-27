"""
AI Search Executor for Document Retrieval using Microsoft Agent Framework.

This executor provides various search capabilities for documents stored in Azure AI Search,
including hybrid search, semantic search, vector search, and traditional text search.
Migrated from services_sk/ai_search_plugin.py.
"""

import os
import json
import base64
import logging
from typing import List, Dict, Any, Optional, Annotated

from agent_framework import Executor, WorkflowContext, handler, ai_function
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    VectorizedQuery,
    QueryType,
    QueryCaptionType,
    QueryAnswerType,
)
from openai import AzureOpenAI
from i18n.locale_msg import LOCALE_MESSAGES
from utils.yield_message import (
    send_step_with_code,
    send_step_with_input,
    send_step_with_code_and_input,
)

logger = logging.getLogger(__name__)


class AISearchExecutor(Executor):
    """
    Executor for searching documents in Azure AI Search with multiple search methods.

    This executor supports:
    - Hybrid search (text + vector)
    - Semantic search
    - Vector search
    - Traditional text search
    """

    def __init__(
        self,
        id: str,
        search_endpoint: Optional[str] = None,
        search_key: Optional[str] = None,
        index_name: Optional[str] = None,
        openai_endpoint: Optional[str] = None,
        openai_key: Optional[str] = None,
        embedding_deployment: Optional[str] = None,
        openai_api_version: Optional[str] = None,
        search_type: Optional[str] = None,
    ):
        """
        Initialize the AI search executor with required clients.

        Args:
            id: Executor ID
            search_endpoint: Azure AI Search endpoint
            search_key: Azure AI Search API key
            index_name: Azure AI Search index name
            openai_endpoint: Azure OpenAI endpoint for embeddings
            openai_key: Azure OpenAI API key
            embedding_deployment: Azure OpenAI embedding deployment name
            openai_api_version: Azure OpenAI API version
            search_type: Search type (semantic, hybrid, vector, text)
        """
        super().__init__(id=id)

        # AI Search setup
        self.search_endpoint = search_endpoint or os.getenv("AZURE_AI_SEARCH_ENDPOINT")
        self.search_key = search_key or os.getenv("AZURE_AI_SEARCH_API_KEY")
        self.index_name = index_name or os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
        self.search_type = search_type or os.getenv("AZURE_AI_SEARCH_SEARCH_TYPE", "hybrid")

        # OpenAI setup for embeddings
        self.openai_endpoint = openai_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.openai_key = openai_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.embedding_deployment = embedding_deployment or os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"
        )
        self.openai_api_version = openai_api_version or os.getenv(
            "AZURE_OPENAI_API_VERSION"
        )

        # Initialize clients
        self._init_clients()

        logger.info(f"AISearchExecutor initialized with index: {self.index_name}")

    def _init_clients(self):
        """Initialize Azure clients."""
        # Search client
        from azure.identity import DefaultAzureCredential

        if self.search_key:
            search_credential = AzureKeyCredential(self.search_key)
        else:
            search_credential = DefaultAzureCredential()

        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.index_name,
            credential=search_credential,
        )

        # OpenAI client
        self.openai_client = AzureOpenAI(
            api_version=self.openai_api_version,
            azure_endpoint=self.openai_endpoint,
            api_key=self.openai_key,
        )

    @handler
    async def search_documents(
        self,
        search_data: Dict[str, Any],
        ctx: WorkflowContext[Dict[str, Any], str],
    ) -> None:
        """Search documents in Azure AI Search for each sub-topic."""
        try:
            # Get metadata for verbose and locale
            metadata = search_data.get("metadata", {})
            locale = metadata.get("locale", "ko-KR")
            verbose = metadata.get("verbose", False)
            LOCALE_MSG = LOCALE_MESSAGES.get(locale, LOCALE_MESSAGES["ko-KR"])

            # ✅ Yield starting message
            await ctx.yield_output(f"data: ### {LOCALE_MSG['searching_ai_search']}\n\n")

            sub_topics = search_data.get("sub_topics", [])
            # ✅ Use self.search_type as default (from env or init param)
            search_type = search_data.get("search_type", self.search_type)
            logger.info(f"[AISearchExecutor] Using search_type: {search_type}")
            filters = search_data.get("filters")
            top_k = search_data.get("top_k", 5)
            include_content = search_data.get("include_content", True)
            document_type = search_data.get("document_type")
            industry = search_data.get("industry")
            company = search_data.get("company")
            report_year = search_data.get("report_year")

            if not sub_topics:
                sub_topics = [
                    {
                        "sub_topic": "research report",
                        "queries": [search_data.get("enriched_query")],
                    }
                ]

            logger.info(
                f"[AISearchExecutor] Searching AI Search for {len(sub_topics)} sub-topics (type: {search_type})"
            )

            # Search by sub-topic
            sub_topic_results = {}

            for sub_topic_data in sub_topics:
                sub_topic_name = sub_topic_data.get("sub_topic", "research")
                queries = sub_topic_data.get("queries", [])

                sub_topic_documents = []

                for query in queries:
                    logger.info(
                        f"[AISearchExecutor] Searching for '{query}' in sub-topic '{sub_topic_name}'"
                    )

                    try:
                        # Generate query vector
                        query_vector = self._generate_embedding(query)

                        # Build filter expression
                        filter_expression = self._build_filters(
                            filters, document_type, industry, company, report_year
                        )

                        # Execute search
                        search_results = self._execute_search(
                            query=query,
                            query_vector=query_vector,
                            search_type=search_type,
                            filter_expression=filter_expression,
                            top_k=top_k,
                            include_content=include_content,
                        )

                        # Process results
                        documents = self._process_search_results(
                            search_results, include_content
                        )
                        sub_topic_documents.extend(documents)

                    except Exception as search_error:
                        error_str = str(search_error)
                        logger.error(f"[AISearchExecutor] AI Search failed: {error_str}")

                        # ✅ 에러를 orchestrator로 전달
                        await ctx.send_message(
                            {
                                **search_data,
                                "sub_topic_ai_search_contexts": {},
                                "executor_error": {
                                    "executor": "ai_search",
                                    "error_type": "search_api_failure",  # 더 명확한 타입
                                    "error_message": error_str,
                                    "is_fatal": True,
                                },
                            }
                        )
                        return  # ✅ 즉시 종료

                if sub_topic_documents:
                    # Store results keyed by sub_topic name
                    sub_topic_results[sub_topic_name] = {
                        "search_type": search_type,
                        "documents": sub_topic_documents,
                        "total_results": len(sub_topic_documents),
                    }

            logger.info(
                f"[AISearchExecutor] Completed AI Search for {len(sub_topic_results)} sub-topics"
            )

            # ✅ Yield completion message (SK compatible format with results)
            if verbose and sub_topic_results:
                results_str = json.dumps(
                    sub_topic_results, ensure_ascii=False, indent=2
                )
                truncated = (
                    results_str[:200] + "... [truncated for display]"
                    if len(results_str) > 200
                    else results_str
                )
                await ctx.yield_output(
                    f"data: {send_step_with_code(LOCALE_MSG['ai_search_context_done'], truncated)}\n\n"
                )
            else:
                await ctx.yield_output(
                    f"data: ### {LOCALE_MSG['ai_search_context_done']}\n\n"
                )

            # Send results to next executor (using SK-compatible key name)
            await ctx.send_message(
                {**search_data, "sub_topic_ai_search_contexts": sub_topic_results}
            )

        except Exception as e:
            error_msg = f"AI Search fatal error: {str(e)}"
            logger.error(f"[AISearchExecutor] {error_msg}")

            # ✅ 최상위 예외도 동일한 형식으로 전달
            await ctx.send_message(
                {
                    **search_data,
                    "sub_topic_ai_search_contexts": {},
                    "executor_error": {
                        "executor": "ai_search",
                        "error_type": "fatal_exception",
                        "error_message": str(e),
                        "is_fatal": True,
                    },
                }
            )

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Azure OpenAI."""
        try:
            response = self.openai_client.embeddings.create(
                input=text, model=self.embedding_deployment
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []

    def _build_filters(
        self,
        filters: Optional[str],
        document_type: Optional[str],
        industry: Optional[str],
        company: Optional[str],
        report_year: Optional[str],
    ) -> Optional[str]:
        """Build OData filter expression."""
        filter_parts = []

        if filters:
            filter_parts.append(filters)
        if document_type:
            filter_parts.append(f"document_type eq '{document_type}'")
        if industry:
            filter_parts.append(f"industry eq '{industry}'")
        if company:
            filter_parts.append(f"company eq '{company}'")
        if report_year:
            filter_parts.append(f"report_year eq '{report_year}'")

        return " and ".join(filter_parts) if filter_parts else None

    def _execute_search(
        self,
        query: str,
        query_vector: List[float],
        search_type: str,
        filter_expression: Optional[str],
        top_k: int,
        include_content: bool,
    ):
        """Execute search based on search type."""
        select_fields = self._get_select_fields(include_content)

        vector_queries = [
            VectorizedQuery(
                vector=query_vector, k_nearest_neighbors=top_k, fields="content_vector"
            ),
            VectorizedQuery(
                vector=query_vector, k_nearest_neighbors=top_k, fields="summary_vector"
            ),
        ]

        if search_type == "hybrid":
            # Hybrid search: text + vector
            return self.search_client.search(
                search_text=query,
                vector_queries=vector_queries,
                filter=filter_expression,
                select=select_fields,
                top=top_k,
                query_type=QueryType.FULL,
                semantic_configuration_name="semantic-config",
            )

        elif search_type == "semantic":
            # Semantic search with captions
            return self.search_client.search(
                search_text=query,
                filter=filter_expression,
                select=select_fields,
                top=top_k,
                query_type=QueryType.SEMANTIC,
                semantic_configuration_name="semantic-config",
                query_caption=QueryCaptionType.EXTRACTIVE,
                query_answer=QueryAnswerType.EXTRACTIVE,
            )

        elif search_type == "vector":
            # Pure vector search
            return self.search_client.search(
                search_text=None,
                vector_queries=vector_queries,
                filter=filter_expression,
                select=select_fields,
                top=top_k,
            )

        elif search_type == "text":
            # Traditional text search
            return self.search_client.search(
                search_text=query,
                filter=filter_expression,
                select=select_fields,
                top=top_k,
            )

        else:
            raise ValueError(f"Unknown search type: {search_type}")

    def _get_select_fields(self, include_content: bool) -> str:
        """Get select fields for search query."""
        base_fields = "docId,title,file_name,summary,document_type,industry,company,report_year,page_number,upload_date,keywords"

        if include_content:
            return f"{base_fields},content"
        else:
            return base_fields

    def _process_search_results(
        self, search_results, include_content: bool
    ) -> List[Dict[str, Any]]:
        """Process search results into a standardized format."""
        documents = []

        for result in search_results:
            doc = {
                "docId": result.get("docId"),
                "title": result.get("title"),
                "file_name": result.get("file_name"),
                "summary": result.get("summary"),
                "document_type": result.get("document_type"),
                "industry": result.get("industry"),
                "company": result.get("company"),
                "report_year": result.get("report_year"),
                "page_number": result.get("page_number"),
                "upload_date": result.get("upload_date"),
                "keywords": result.get("keywords"),
                "score": result.get("@search.score"),
            }

            if include_content:
                doc["content"] = result.get("content")

            # Add semantic captions if available
            if self.search_type=="semantic" and hasattr(result, "@search.captions"):
                doc["captions"] = [
                    {"text": caption.text, "highlights": caption.highlights}
                    for caption in result["@search.captions"]
                ]

            documents.append(doc)

        return documents
