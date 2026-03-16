"""Stage 2: RAG Document Retrieval with Multi-Query Expansion."""

import json
from typing import List

from backend.config import settings, PipelineConfig
from backend.database import vector_store
from backend.models import Document
from backend.observability import instrument_stage
from backend.pipeline.embeddings import embed_question
from backend.pipeline.query_expansion import expand_query, should_expand_query
import logfire


# Pricing keywords that trigger explicit pricing table injection
PRICING_KEYWORDS = [
    'pricing', 'price', 'cost', 'costs', 'plan', 'plans', 'tier', 'tiers',
    'instance type', 'instance types', '$', 'dollar', 'monthly', 'per month',
    'how much', 'what does it cost'
]

PRODUCT_KEYWORDS = {
    'postgres': ['Render Postgres Pricing'],
    'postgresql': ['Render Postgres Pricing'],
    'database': ['Render Postgres Pricing', 'Render Key Value Pricing'],
    'datastore': ['Render Postgres Pricing', 'Render Key Value Pricing'],
    'key value': ['Render Key Value Pricing'],
    'keyvalue': ['Render Key Value Pricing'],
    'redis': ['Render Key Value Pricing'],
    'valkey': ['Render Key Value Pricing'],
    'web service': ['Render Web Services Pricing'],
    'private service': ['Render Web Services Pricing'],
    'background worker': ['Render Web Services Pricing'],
    'cron': ['Render Cron Jobs Pricing'],
    'cron job': ['Render Cron Jobs Pricing'],
}


def detect_pricing_query(question: str) -> List[str]:
    """
    Detect if question is asking about pricing/plans and which products.
    
    Returns list of pricing table titles to explicitly inject.
    """
    question_lower = question.lower()
    
    # IMPORTANT: Don't trigger on "tier" if it's part of "free tier" (that's about instance behavior, not pricing)
    if "free tier" in question_lower or "free instance" in question_lower:
        # This is a question about free tier behavior, not pricing
        return []
    
    # Check if pricing-related
    is_pricing_query = any(keyword in question_lower for keyword in PRICING_KEYWORDS)
    
    if not is_pricing_query:
        return []
    
    # Determine which product pricing tables to inject
    tables_to_inject = set()
    
    for product_keyword, table_titles in PRODUCT_KEYWORDS.items():
        if product_keyword in question_lower:
            tables_to_inject.update(table_titles)
    
    # If no specific product mentioned but pricing query, use smart defaults
    if not tables_to_inject:
        # If asking about "instance types" specifically, include ALL pricing tables
        # since instance types exist for web services, databases, and cron jobs
        if 'instance type' in question_lower:
            tables_to_inject = {
                'Render Web Services Pricing',
                'Render Postgres Pricing',
                'Render Key Value Pricing',
                'Render Cron Jobs Pricing'
            }
        else:
            # For other generic pricing questions, default to databases
            tables_to_inject = {'Render Postgres Pricing', 'Render Key Value Pricing'}
    
    return list(tables_to_inject)


async def inject_pricing_tables(question: str, existing_docs: List[Document]) -> List[Document]:
    """
    Explicitly fetch and inject pricing tables when pricing keywords detected.
    
    This ensures pricing queries ALWAYS get pricing tables, regardless of semantic search.
    """
    pricing_tables_needed = detect_pricing_query(question)
    
    if not pricing_tables_needed:
        return existing_docs
    
    logfire.info(f"Pricing query detected, injecting tables: {pricing_tables_needed}")
    
    # Fetch pricing documents from database
    injected_docs = []
    
    async with vector_store.pool.acquire() as conn:
        for table_title in pricing_tables_needed:
            row = await conn.fetchrow("""
                SELECT content, source, title, section, metadata, embedding
                FROM documents
                WHERE title = $1 AND source = 'https://render.com/pricing'
                LIMIT 1
            """, table_title)
            
            if row:
                # Create Document object with high similarity score to ensure it ranks highly
                # Parse metadata if it's a string (JSONB might come back as string)
                metadata = row['metadata']
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                elif metadata is None:
                    metadata = {}
                
                doc = Document(
                    content=row['content'],
                    source=row['source'],
                    metadata={
                        'title': row['title'],
                        'section': row['section'] or row['title'],
                        **metadata
                    },
                    similarity_score=0.95  # High score to ensure it ranks at top
                )
                injected_docs.append(doc)
    
    if injected_docs:
        logfire.info(f"Injected {len(injected_docs)} pricing tables")
        
        # Add injected docs at the beginning (highest priority)
        return injected_docs + existing_docs
    
    return existing_docs


@instrument_stage(PipelineConfig.STAGE_RETRIEVAL)
async def retrieve_documents(embedding: List[float], original_question: str = None) -> dict:
    """
    Find relevant documentation chunks via vector similarity.
    
    Uses multi-query retrieval for broad questions to ensure diverse coverage
    across multiple products/aspects.
    
    Args:
        embedding: Query embedding vector (used for fallback)
        original_question: Original question text (for query expansion)
        
    Returns:
        dict with 'documents', 'avg_similarity', 'cost_usd'
    """
    
    total_cost = 0.0001  # Base database query cost
    
    # Check if we should use multi-query retrieval
    if original_question and await should_expand_query(original_question):
        logfire.info(
            "Using multi-query retrieval for broad question",
            question_length=len(original_question),
            rag_top_k=settings.rag_top_k
        )
        
        # Expand query
        query_variations, expansion_cost = await expand_query(original_question)
        total_cost += expansion_cost
        
        logfire.info(
            "Expanded query to multiple variations",
            num_queries=len(query_variations),
            queries=query_variations,
            expansion_cost_usd=expansion_cost
        )
        
        # Retrieve documents for each query variation
        all_docs = {}  # Dict for deduplication by content hash
        
        # Calculate how many docs to retrieve per query
        # Target: ~30-40 total docs before dedup, then take top 20
        docs_per_query = max(10, settings.rag_top_k // len(query_variations) + 5)
        
        for i, query in enumerate(query_variations):
            logfire.debug(f"Retrieving for query {i+1}/{len(query_variations)}: {query}")
            
            # Embed the variation
            embed_result = await embed_question(query)
            total_cost += embed_result["cost_usd"]
            
            # Retrieve documents using hybrid search for better accuracy
            docs = await vector_store.hybrid_search(
                query_text=query,
                query_embedding=embed_result["embedding"],
                k=docs_per_query,
                threshold=settings.similarity_threshold,
                bm25_weight=0.4  # 60% semantic, 40% BM25
            )
            
            # Deduplicate: Keep doc with highest similarity if duplicate content
            for doc in docs:
                # Use first 200 chars as content hash
                content_hash = hash(doc.content[:200])
                
                # CRITICAL: Boost similarity for original query (first query)
                # Similarity scores across different queries aren't directly comparable
                # We prioritize results from the original query
                if i == 1:  # First query (original question)
                    doc.similarity_score = doc.similarity_score * 1.15  # 15% boost for original query
                
                if content_hash not in all_docs or doc.similarity_score > all_docs[content_hash].similarity_score:
                    all_docs[content_hash] = doc
        
        # Sort by similarity and take top k
        documents = sorted(all_docs.values(), key=lambda d: d.similarity_score, reverse=True)[:settings.rag_top_k]
        
        logfire.info(
            "Multi-query retrieval completed",
            num_queries=len(query_variations),
            total_docs_before_dedup=sum(1 for _ in all_docs.values()),
            final_docs=len(documents)
        )
    else:
        # Single query retrieval with hybrid search (semantic + BM25)
        logfire.info("Using hybrid search (semantic + BM25)")
        
        documents = await vector_store.hybrid_search(
            query_text=original_question or "",
            query_embedding=embedding,
            k=settings.rag_top_k,
            threshold=settings.similarity_threshold,
            bm25_weight=0.4  # 60% semantic, 40% BM25 - favors semantic but includes keyword matches
        )
    
    # PRICING TABLE INJECTION: If pricing query detected, explicitly inject pricing tables
    if original_question:
        pre_injection_count = len(documents)
        documents = await inject_pricing_tables(original_question, documents)
        if len(documents) > pre_injection_count:
            logfire.info(
                "Injected pricing tables",
                tables_injected=len(documents) - pre_injection_count,
                total_docs=len(documents)
            )
    
    # Calculate average similarity
    avg_similarity = 0.0
    if documents:
        avg_similarity = sum(doc.similarity_score for doc in documents) / len(documents)
    
    logfire.info(
        "Documents retrieved",
        count=len(documents),
        avg_similarity=avg_similarity,
        top_score=documents[0].similarity_score if documents else 0.0
    )
    
    return {
        "documents": documents,
        "avg_similarity": avg_similarity,
        "cost_usd": total_cost
    }

