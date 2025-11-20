"""
GraphRAG Query Helper

Simplified query functions for local and global search using GraphRAG SDK.
Based on graphrag.query.api module.
"""

import logging
from typing import Any, Dict, Tuple, Optional

import pandas as pd
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.query.factory import (
    get_global_search_engine,
    get_local_search_engine,
)
from graphrag.query.indexer_adapters import (
    read_indexer_communities,
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.utils.api import (
    get_embedding_store,
    load_search_prompt,
)
from graphrag.config.embeddings import entity_description_embedding

logger = logging.getLogger(__name__)


async def local_search(
    config: GraphRagConfig,
    entities: pd.DataFrame,
    communities: pd.DataFrame,
    community_reports: pd.DataFrame,
    text_units: pd.DataFrame,
    relationships: pd.DataFrame,
    covariates: pd.DataFrame | None,
    community_level: int,
    response_type: str,
    query: str,
    generate_answer: bool = False,  # ⭐ 새로운 파라미터
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Perform a local search and return the response and context data.
    
    Args:
        config: GraphRAG configuration
        entities: Entities DataFrame
        communities: Communities DataFrame
        community_reports: Community reports DataFrame
        text_units: Text units DataFrame
        relationships: Relationships DataFrame
        covariates: Covariates DataFrame (optional)
        community_level: Community level to search at
        response_type: Response type
        query: Search query
        generate_answer: If True, generate LLM answer; if False, return only context (default: False)
        
    Returns:
        Tuple of (response_text or None, context_data)
    """
    logger.info(f"Executing local search: {query} (generate_answer={generate_answer})")
    
    # Get vector store configuration
    vector_store_args = {}
    for index, store in config.vector_store.items():
        vector_store_args[index] = store.model_dump()
    
    # Get embedding store
    description_embedding_store = get_embedding_store(
        config_args=vector_store_args,
        embedding_name=entity_description_embedding,
    )
    
    # Read indexer data
    entities_ = read_indexer_entities(entities, communities, community_level)
    covariates_ = read_indexer_covariates(covariates) if covariates is not None else []
    reports = read_indexer_reports(community_reports, communities, community_level)
    text_units_ = read_indexer_text_units(text_units)
    relationships_ = read_indexer_relationships(relationships)
    
    # Load prompt
    prompt = load_search_prompt(config.root_dir, config.local_search.prompt)
    
    # Create search engine
    search_engine = get_local_search_engine(
        config=config,
        reports=reports,
        text_units=text_units_,
        entities=entities_,
        relationships=relationships_,
        covariates={"claims": covariates_},
        description_embedding_store=description_embedding_store,
        response_type=response_type,
        system_prompt=prompt,
    )
    
    # Build context first (this is fast, not async)
    context_result = search_engine.context_builder.build_context(
        query=query,
        **search_engine.context_builder_params,
    )
    
    # Debug: Log context_result structure
    chunks_len = len(context_result.context_chunks) if hasattr(context_result, 'context_chunks') else 0
    records_count = len(context_result.context_records) if hasattr(context_result, 'context_records') else 0
    logger.info(f"Context result: chunks={chunks_len}, records={records_count}")
    
    # Extract and structure context_records for LLM usage
    # context_records is a dict with keys like 'entities', 'relationships', 'reports', 'sources'
    structured_records = {}
    if hasattr(context_result, "context_records") and isinstance(context_result.context_records, dict):
        for key, value in context_result.context_records.items():
            if isinstance(value, pd.DataFrame) and not value.empty:
                # Convert DataFrame to list of dicts for JSON serialization
                structured_records[key] = value.to_dict(orient="records")
                logger.info(f"  - {key}: {len(value)} records")
            elif isinstance(value, list):
                structured_records[key] = value
                logger.info(f"  - {key}: {len(value)} items")
            else:
                structured_records[key] = str(value) if value is not None else None
    
    # Prepare context data with both raw chunks (for LLM) and structured records (for analysis)
    context_data = {
        "context_chunks": context_result.context_chunks if hasattr(context_result, "context_chunks") else "",
        "context_records": structured_records,  # Structured data for programmatic access
        "llm_calls": context_result.llm_calls if hasattr(context_result, "llm_calls") else 0,
        "prompt_tokens": context_result.prompt_tokens if hasattr(context_result, "prompt_tokens") else 0,
        "output_tokens": context_result.output_tokens if hasattr(context_result, "output_tokens") else 0,
    }
    
    # Optionally generate LLM answer
    full_response = None
    if generate_answer:
        logger.info("Generating LLM answer...")
        full_response = ""
        async for chunk in search_engine.stream_search(query=query):
            full_response += chunk
        logger.info(f"Local search completed: {len(full_response)} chars")
    else:
        logger.info("Skipping LLM generation, returning context only")
    
    return full_response, context_data


async def global_search(
    config: GraphRagConfig,
    entities: pd.DataFrame,
    communities: pd.DataFrame,
    community_reports: pd.DataFrame,
    community_level: int | None,
    dynamic_community_selection: bool,
    response_type: str,
    query: str,
    generate_answer: bool = False,  # ⭐ 새로운 파라미터
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Perform a global search and return the response and context data.
    
    Args:
        config: GraphRAG configuration
        entities: Entities DataFrame
        communities: Communities DataFrame
        community_reports: Community reports DataFrame
        community_level: Community level to search at
        dynamic_community_selection: Enable dynamic community selection
        response_type: Response type
        query: Search query
        generate_answer: If True, generate LLM answer; if False, return only context (default: False)
        
    Returns:
        Tuple of (response_text or None, context_data)
    """
    logger.info(f"Executing global search: {query} (generate_answer={generate_answer})")
    
    # Read indexer data
    communities_ = read_indexer_communities(communities, community_reports)
    reports = read_indexer_reports(
        community_reports,
        communities,
        community_level=community_level,
        dynamic_community_selection=dynamic_community_selection,
    )
    entities_ = read_indexer_entities(
        entities, communities, community_level=community_level
    )
    
    # Load prompts
    map_prompt = load_search_prompt(config.root_dir, config.global_search.map_prompt)
    reduce_prompt = load_search_prompt(
        config.root_dir, config.global_search.reduce_prompt
    )
    knowledge_prompt = load_search_prompt(
        config.root_dir, config.global_search.knowledge_prompt
    )
    
    # Create search engine
    search_engine = get_global_search_engine(
        config,
        reports=reports,
        entities=entities_,
        communities=communities_,
        response_type=response_type,
        dynamic_community_selection=dynamic_community_selection,
        map_system_prompt=map_prompt,
        reduce_system_prompt=reduce_prompt,
        general_knowledge_inclusion_prompt=knowledge_prompt,
    )
    
    # Build context first (fast - just selects relevant communities)
    context_data = {
        "communities": communities_,
        "reports": reports,
        "entities": entities_,
    }
    
    # Optionally generate LLM answer
    full_response = None
    if generate_answer:
        logger.info("Generating LLM answer...")
        full_response = ""
        async for chunk in search_engine.stream_search(query=query):
            full_response += chunk
        logger.info(f"Global search completed: {len(full_response)} chars")
    else:
        logger.info("Skipping LLM generation, returning context only")
    
    return full_response, context_data