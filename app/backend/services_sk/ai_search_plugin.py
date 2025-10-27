"""
AI Search Plugin for Document Retrieval

This plugin provides various search capabilities for documents stored in Azure AI Search,
including hybrid search, semantic search, vector search, and traditional text search.
Optimized for IR reports and market research documents.
"""
import os
from typing import List, Dict, Any, Optional
from semantic_kernel.functions import kernel_function
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    VectorizedQuery,
    QueryType,
    QueryCaptionType,
    QueryAnswerType
)
from openai import AzureOpenAI

from config.config import Settings


class AISearchPlugin:
    """Plugin for searching documents in Azure AI Search with multiple search methods."""
    
    def __init__(self, settings: Settings):
        """Initialize the AI search plugin with required clients."""
        # AI Search setup
        self.search_endpoint = settings.AZURE_AI_SEARCH_ENDPOINT
        self.search_key = settings.AZURE_AI_SEARCH_API_KEY
        self.index_name = settings.AZURE_AI_SEARCH_INDEX_NAME
        
        # OpenAI setup for embeddings
        self.openai_endpoint = settings.AZURE_OPENAI_ENDPOINT
        self.openai_key = settings.AZURE_OPENAI_API_KEY
        self.embedding_deployment = settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
        self.openai_api_version = settings.AZURE_OPENAI_API_VERSION
        self.search_type = settings.AZURE_AI_SEARCH_SEARCH_TYPE or "hybrid"
        # Initialize clients
        self._init_clients()
    
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
            credential=search_credential
        )
        
        # OpenAI client
        self.openai_client = AzureOpenAI(
            api_version=self.openai_api_version,
            azure_endpoint=self.openai_endpoint,
            api_key=self.openai_key
        )
    
    @kernel_function(
        description="Search documents in Azure AI Search using various search methods",
        name="search_documents",
    )
    def search_documents(
        self,
        query: str,
        filters: Optional[str] = None,
        top_k: int = 5,
        include_content: bool = True,
        document_type: Optional[str] = None,
        industry: Optional[str] = None,
        company: Optional[str] = None,
        report_year: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search documents in Azure AI Search using various search methods.
        
        Args:
            query: Search query
            search_type: Type of search (hybrid, semantic, vector, text)
            filters: OData filter expression
            top_k: Number of results to return
            include_content: Whether to include full content in results
            document_type: Filter by document type
            industry: Filter by industry
            company: Filter by company
            report_year: Filter by report year
        
        Returns:
            Dict containing search results and metadata
        """
        try:
            # Generate query embedding for vector search
            query_vector = self._generate_embedding(query)
            
            # Build filter expression
            filter_expression = self._build_filters(
                filters, document_type, industry, company, report_year
            )

            # Configure search based on search type
            search_results = self._execute_search(
                query, query_vector, self.search_type, filter_expression, top_k, include_content
            )
            
            # Process results
            documents = self._process_search_results(search_results, include_content)
            
            # Get search answers if available
            answers = self._extract_answers(search_results)
            
            return {
                "status": "success",
                "query": query,
                "search_type": self.search_type,
                "filter": filter_expression,
                "total_results": len(documents),
                "documents": documents,
                "answers": answers,
                "message": f"Found {len(documents)} documents for query: {query}"
            }
            
        except Exception as e:
            error_msg = f"Error searching documents: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "status": "error",
                "message": error_msg
            }
    
    def get_document_by_id(self, doc_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific document by its ID.
        
        Args:
            doc_id: Document ID
        
        Returns:
            Dict containing document data or error message
        """
        try:
            document = self.search_client.get_document(key=doc_id)
            return {
                "status": "success",
                "document": dict(document)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Document not found: {str(e)}"
            }
    
    def list_documents(
        self,
        document_type: Optional[str] = None,
        industry: Optional[str] = None,
        company: Optional[str] = None,
        report_year: Optional[str] = None,
        top: int = 20,
        order_by: str = "uploadDate desc"
    ) -> Dict[str, Any]:
        """
        List documents with optional filtering.
        
        Args:
            document_type: Filter by document type
            industry: Filter by industry
            company: Filter by company
            report_year: Filter by report year
            top: Number of documents to return
            order_by: Sort order
        
        Returns:
            Dict containing list of documents
        """
        try:
            # Build filter
            filter_expression = self._build_filters(
                None, document_type, industry, company, report_year, None, None
            )
            
            results = self.search_client.search(
                search_text="*",
                filter=filter_expression,
                top=top,
                select="docId,file_name,document_type,industry,company,report_year,page_number,upload_date,summary,keywords",
                order_by=[order_by] if order_by else None
            )
            
            documents = [dict(result) for result in results]
            
            return {
                "status": "success",
                "total_results": len(documents),
                "documents": documents
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error listing documents: {str(e)}"
            }
    
    def get_document_statistics(self) -> Dict[str, Any]:
        """Get statistics about documents in the index."""
        try:
            # Get total count
            total_results = self.search_client.search(search_text="*", top=0, include_total_count=True)
            total_count = total_results.get_count()
            
            # Get counts by document type
            doc_type_results = self.search_client.search(
                search_text="*",
                facets=["documentType,count:10", "industry,count:10", "company,count:10", "reportYear,count:10"],
                top=0
            )
            
            facets = dict(doc_type_results.get_facets()) if hasattr(doc_type_results, 'get_facets') else {}
            
            return {
                "status": "success",
                "total_documents": total_count,
                "facets": facets,
                "message": f"Index contains {total_count} documents"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error getting statistics: {str(e)}"
            }
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Azure OpenAI."""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_deployment
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Warning: Failed to generate embedding: {str(e)}")
            # Return a zero vector as fallback
            return [0.0] * 1536  # Default embedding dimension
    
    def _build_filters(
        self,
        filters: Optional[str],
        document_type: Optional[str],
        industry: Optional[str],
        company: Optional[str],
        report_year: Optional[str]
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
        include_content: bool
    ):
        """Execute search based on search type."""
        select_fields = self._get_select_fields(include_content)

        vector_queries = [
                VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=top_k,
                    fields="content_vector"
                ),
                VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=top_k,
                    fields="summary_vector"
                )
            ]
        
        if search_type == "hybrid":
            # Hybrid search: combines text and vector search
            
            return self.search_client.search(
                search_text=query,
                vector_queries=vector_queries,
                filter=filter_expression,
                query_type=QueryType.FULL,
                semantic_configuration_name="semantic-config",
                query_caption=QueryCaptionType.EXTRACTIVE,
                query_answer=QueryAnswerType.EXTRACTIVE,
                top=top_k,
                select=select_fields
            )
            
        elif search_type == "semantic":
            # Semantic search only
            return self.search_client.search(
                search_text=query,
                vector_queries=vector_queries,
                filter=filter_expression,
                query_type=QueryType.SEMANTIC,
                semantic_configuration_name="semantic-config",
                query_caption=QueryCaptionType.EXTRACTIVE,
                query_answer=QueryAnswerType.EXTRACTIVE,
                top=top_k,
                select=select_fields
            )
            
        elif search_type == "vector":
            # Vector search only
            return self.search_client.search(
                search_text=None,
                vector_queries=vector_queries,
                filter=filter_expression,
                top=top_k,
                select=select_fields
            )
            
        elif search_type == "text":
            # Text search only
            return self.search_client.search(
                search_text=query,
                filter=filter_expression,
                top=top_k,
                select=select_fields
            )
        
        else:
            raise ValueError(f"Unsupported search type: {search_type}")
    
    def _get_select_fields(self, include_content: bool) -> str:
        """Get select fields for search query."""
        base_fields = "docId,title,file_name,summary,document_type,industry,company,report_year,page_number,upload_date,keywords"
        
        if include_content:
            return base_fields + ",content"
        else:
            return base_fields
    
    def _process_search_results(self, search_results, include_content: bool) -> List[Dict[str, Any]]:
        """Process search results into a standardized format."""
        documents = []
        
        for result in search_results:
            doc = dict(result)
            
            # Add search metadata
            if hasattr(result, '@search.score'):
                doc['search_score'] = result['@search.score']
            if self.search_type=="semantic" and hasattr(result, '@search.reranker_score'):
                doc['reranker_score'] = result['@search.reranker_score']
            if self.search_type=="semantic" and hasattr(result, '@search.captions'):
                doc['captions'] = [caption.text for caption in result['@search.captions']]
            
            # Truncate content if needed for response size
            if include_content and 'content' in doc and len(doc['content']) > 2000:
                doc['content_preview'] = doc['content'][:2000] + "..."
                if not include_content:
                    del doc['content']
            
            documents.append(doc)
        
        return documents
    
    def _extract_answers(self, search_results) -> List[Dict[str, Any]]:
        """Extract answers from search results if available."""
        answers = []
        try:
            if hasattr(search_results, 'get_answers') and search_results.get_answers():
                answers = [
                    {"text": answer.text, "score": answer.score} 
                    for answer in search_results.get_answers()
                ]
        except Exception:
            # Answers not available or error occurred
            pass
        
        return answers


# Convenience functions for backward compatibility
def ai_search_plugin(query: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to search documents."""
    plugin = AISearchPlugin()
    return plugin.search_documents(query=query, **kwargs)


def get_document_by_id(doc_id: str) -> Dict[str, Any]:
    """Convenience function to get document by ID."""
    plugin = AISearchPlugin()
    return plugin.get_document_by_id(doc_id)


def list_documents(**kwargs) -> Dict[str, Any]:
    """Convenience function to list documents."""
    plugin = AISearchPlugin()
    return plugin.list_documents(**kwargs)


def get_document_statistics() -> Dict[str, Any]:
    """Convenience function to get document statistics."""
    plugin = AISearchPlugin()
    return plugin.get_document_statistics()
