"""Stage 2: RAG Document Retrieval with Multi-Query Expansion."""

import asyncio
import json
import re
from typing import Callable, List

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

PRICING_SOURCE = "https://render.com/pricing"

# AI/agent keywords that trigger the Render Workflows agents tutorial injection
AI_AGENT_KEYWORDS = [
    'ai agent', 'ai agents', 'llm agent', 'llm', 'language model',
    'artificial intelligence', 'machine learning', 'deploy ai', 'deploy agent',
    'long-running', 'long running', 'self-orchestrating', 'render workflows',
    'agent workflow', 'agent deployment', 'agentic',
]

# Single-word AI keywords matched with word boundaries to avoid false positives.
# 'ai' is matched with word boundaries so it triggers on "ai" but not "email"/"detail".
AI_AGENT_SINGLE_WORD_KEYWORDS = ['agent', 'agents', 'ai']

# For "how do I deploy/run an AI agent on Render?" — and any question mentioning
# "ai" or "agents" — we inject two authoritative sources, both at top priority:
#   1. the Workflows agents tutorial (brings home the canonical answer), and
#   2. the official Workflows docs (gives the verification + accuracy stages
#      authoritative material to check the generated answer against).
AI_AGENT_WORKFLOWS_SOURCE = "https://render.com/tutorials/agents-on-render-workflows/what-youll-build"
AI_AGENT_WORKFLOWS_DOCS_SOURCE = "https://render.com/docs/workflows"

# Autoscaling keywords
AUTOSCALING_KEYWORDS = [
    'autoscaling', 'autoscale', 'auto-scaling', 'auto scaling',
    'horizontal scaling', 'scale automatically', 'automatically scale',
    'scale up', 'scale down', 'min instances', 'max instances',
    'scaling policy', 'scale based on',
]
AUTOSCALING_SINGLE_WORD_KEYWORDS = ['scaling']
AUTOSCALING_DOC_SOURCE = "https://render.com/docs/scaling"

# Node.js deployment keywords
NODEJS_KEYWORDS = [
    'node.js', 'nodejs', 'node js', 'express', 'deploy node',
    'npm start', 'npm install', 'next.js', 'nextjs', 'deploy next',
    'vite', 'javascript app', 'js app', 'deploy javascript',
]
NODEJS_SINGLE_WORD_KEYWORDS = ['node']
NODEJS_DOC_SOURCE = "https://render.com/docs/deploy-node-express-app"

# Tutorials keywords that trigger the render.com/tutorials index recommendation
TUTORIALS_KEYWORDS = ['tutorial', 'tutorials']
TUTORIALS_INDEX_SOURCE = "https://render.com/tutorials"

# Curated docs are injected at the very top of the cosine scale. 1.0 is unreachable
# by a real search hit, so it cleanly marks a doc the curation layer added — and
# keeps it above retrieved docs without any score-boosting math.
INJECTED_DOC_SCORE = 1.0


def detect_ai_agent_query(question: str) -> bool:
    """Detect if the question is asking about AI agents or long-running agent processes."""
    question_lower = question.lower()

    # Check multi-word and phrase keywords first
    if any(keyword in question_lower for keyword in AI_AGENT_KEYWORDS):
        return True

    # Check single-word keywords with word boundaries to avoid false positives
    # (e.g. "agent" should match but "email" should not match "ai")
    for keyword in AI_AGENT_SINGLE_WORD_KEYWORDS:
        if re.search(r'\b' + re.escape(keyword) + r'\b', question_lower):
            return True

    return False


def detect_autoscaling_query(question: str) -> bool:
    """Detect if the question is asking about autoscaling or scaling configuration."""
    question_lower = question.lower()

    if any(keyword in question_lower for keyword in AUTOSCALING_KEYWORDS):
        return True

    for keyword in AUTOSCALING_SINGLE_WORD_KEYWORDS:
        if re.search(r'\b' + re.escape(keyword) + r'\b', question_lower):
            return True

    return False


def detect_nodejs_query(question: str) -> bool:
    """Detect if the question is asking about deploying Node.js or JavaScript apps."""
    question_lower = question.lower()

    if any(keyword in question_lower for keyword in NODEJS_KEYWORDS):
        return True

    for keyword in NODEJS_SINGLE_WORD_KEYWORDS:
        if re.search(r'\b' + re.escape(keyword) + r'\b', question_lower):
            return True

    return False


def detect_tutorials_query(question: str) -> bool:
    """Detect if the question mentions tutorials."""
    question_lower = question.lower()
    return any(
        re.search(r'\b' + re.escape(keyword) + r'\b', question_lower)
        for keyword in TUTORIALS_KEYWORDS
    )


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


# ---------------------------------------------------------------------------
# Data-driven curated-doc injection
# ---------------------------------------------------------------------------
#
# Each curated rule resolves a question to zero or more "fetch specs" — the
# canonical docs that should be present for that topic. A spec is matched either
# by exact source URL or by (title, source) for the pricing tables (which share a
# source but differ by title). One generic injector then fetches, gates, and
# applies a replace-weakest policy, so adding a topic is a single list entry
# rather than another ~50-line copy-pasted function.


def _source_spec(url: str, hint: str) -> dict:
    return {"by": "source", "value": url, "hint": hint}


def _title_spec(title: str, hint: str) -> dict:
    return {"by": "title", "value": (title, PRICING_SOURCE), "hint": hint}


# A rule is (name, resolve) where resolve(question) -> list[spec].
CURATED_RULES: List[tuple[str, Callable[[str], List[dict]]]] = [
    (
        "pricing",
        lambda q: [
            _title_spec(title, "data/scripts/add_pricing_page.py")
            for title in detect_pricing_query(q)
        ],
    ),
    (
        "ai_agent",
        lambda q: (
            [
                _source_spec(AI_AGENT_WORKFLOWS_SOURCE, "data/scripts/add_workflows_tutorial_page.py"),
                _source_spec(AI_AGENT_WORKFLOWS_DOCS_SOURCE, "data/scripts/add_workflows_docs_page.py"),
            ]
            if detect_ai_agent_query(q)
            else []
        ),
    ),
    (
        "autoscaling",
        lambda q: (
            [_source_spec(AUTOSCALING_DOC_SOURCE, "data/scripts/add_autoscaling_page.py")]
            if detect_autoscaling_query(q)
            else []
        ),
    ),
    (
        "nodejs",
        lambda q: (
            [_source_spec(NODEJS_DOC_SOURCE, "data/scripts/add_nodejs_page.py")]
            if detect_nodejs_query(q)
            else []
        ),
    ),
    (
        "tutorials",
        lambda q: (
            [_source_spec(TUTORIALS_INDEX_SOURCE, "data/scripts/add_tutorials_index_page.py")]
            if detect_tutorials_query(q)
            else []
        ),
    ),
]


def _resolve_curated_specs(question: str) -> List[dict]:
    """Collect curated fetch specs from every matching rule, de-duplicated."""
    specs: List[dict] = []
    seen = set()
    for name, resolve in CURATED_RULES:
        for spec in resolve(question):
            key = (spec["by"], spec["value"])
            if key not in seen:
                seen.add(key)
                spec["rule"] = name
                specs.append(spec)
    return specs


def _row_to_curated_doc(row) -> Document:
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    elif metadata is None:
        metadata = {}

    return Document(
        content=row["content"],
        source=row["source"],
        metadata={
            "title": row["title"],
            "section": row["section"] or row["title"],
            **metadata,
        },
        similarity_score=INJECTED_DOC_SCORE,
    )


async def _fetch_curated_docs(conn, spec: dict) -> List[Document]:
    """Fetch curated docs for a spec.

    A by-source spec returns *all* chunks for that page — curated docs are now
    section-chunked (one embedding per heading), so the whole canonical page is
    injected rather than a single arbitrary section. A by-title spec (pricing
    tables) returns the one matching table.
    """
    if spec["by"] == "source":
        rows = await conn.fetch(
            """
            SELECT content, source, title, section, metadata
            FROM documents
            WHERE source = $1
            ORDER BY id
            """,
            spec["value"],
        )
    else:  # by title
        title, source = spec["value"]
        rows = await conn.fetch(
            """
            SELECT content, source, title, section, metadata
            FROM documents
            WHERE title = $1 AND source = $2
            LIMIT 1
            """,
            title,
            source,
        )

    return [_row_to_curated_doc(row) for row in rows]


async def inject_curated_docs(question: str, existing_docs: List[Document]) -> List[Document]:
    """
    Ensure the canonical doc for each matched topic is present — without padding
    the count.

    Policy (relevance-gated, replace-weakest, never grow past rag_top_k):
      - If a topic's curated doc was already retrieved, leave it (don't duplicate).
      - Otherwise insert it at the top; if that pushes the set over the rag_top_k
        ceiling, drop the weakest retrieved docs from the tail.
    """
    specs = _resolve_curated_specs(question)
    if not specs:
        return existing_docs

    existing_sources = {doc.source for doc in existing_docs}
    existing_titles = {(doc.metadata or {}).get("title") for doc in existing_docs}

    injected: List[Document] = []
    async with vector_store.pool.acquire() as conn:
        for spec in specs:
            # Skip curated docs that semantic/BM25 search already surfaced.
            if spec["by"] == "source" and spec["value"] in existing_sources:
                continue
            if spec["by"] == "title" and spec["value"][0] in existing_titles:
                continue

            docs = await _fetch_curated_docs(conn, spec)
            if not docs:
                logfire.warning(
                    "Curated doc not found in DB",
                    rule=spec.get("rule"),
                    target=str(spec["value"]),
                    hint=f"run {spec['hint']}",
                )
                continue
            injected.extend(docs)

    if not injected:
        return existing_docs

    # Replace-weakest: curated docs lead the context; truncating from the tail
    # drops the lowest-ranked retrieved docs so the set never grows past the cap.
    combined = injected + existing_docs
    dropped = max(0, len(combined) - settings.rag_top_k)
    if dropped:
        combined = combined[: settings.rag_top_k]

    logfire.info(
        "Injected curated docs (replace-weakest)",
        injected=len(injected),
        dropped_to_fit=dropped,
        total_docs=len(combined),
    )
    return combined


def _apply_relative_cutoff(documents: List[Document]) -> List[Document]:
    """
    Adaptive relevance gate anchored to the best match.

    Keep only docs whose cosine >= max(similarity_threshold, top * fraction), so
    strong topics filter aggressively while weaker-but-valid topics keep their
    cluster. Applied BEFORE curated injection so the injected 1.0 scores don't
    skew the anchor.
    """
    if not documents:
        return documents

    top = max(doc.similarity_score for doc in documents)
    cutoff = max(settings.similarity_threshold, top * settings.relevance_cutoff_fraction)
    survivors = [doc for doc in documents if doc.similarity_score >= cutoff]
    survivors.sort(key=lambda doc: doc.similarity_score, reverse=True)

    logfire.info(
        "Applied relative relevance cutoff",
        top_score=top,
        cutoff=cutoff,
        kept=len(survivors),
        dropped=len(documents) - len(survivors),
    )
    return survivors


def collapse_sources(documents: List[Document]) -> List[Document]:
    """
    Collapse retrieved chunks into one entry per (source, title) for display.

    Keeps the highest-scoring chunk as the representative and records how many
    sections matched in metadata['matched_sections']. The full chunk list still
    feeds answer generation; only the UI-facing "Sources" view is collapsed, so
    the same page no longer appears five times. Pricing tables share a source but
    differ by title, so grouping on the (source, title) pair keeps them separate.
    """
    representatives: dict = {}
    counts: dict = {}
    order: List[tuple] = []

    for doc in documents:
        title = (doc.metadata or {}).get("title")
        key = (doc.source, title)
        if key not in representatives:
            representatives[key] = doc
            counts[key] = 1
            order.append(key)
        else:
            counts[key] += 1
            if doc.similarity_score > representatives[key].similarity_score:
                representatives[key] = doc

    collapsed: List[Document] = []
    for key in order:
        rep = representatives[key]
        collapsed.append(
            rep.model_copy(
                update={"metadata": {**(rep.metadata or {}), "matched_sections": counts[key]}}
            )
        )
    return collapsed


@instrument_stage(PipelineConfig.STAGE_RETRIEVAL)
async def retrieve_documents(embedding: List[float], original_question: str = None) -> dict:
    """
    Find relevant documentation chunks via vector similarity.

    Uses multi-query retrieval for broad questions to ensure diverse coverage
    across multiple products/aspects, then applies an adaptive relevance cutoff
    and relevance-gated curated-doc injection.

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
        all_docs = {}            # content hash -> Document (highest cosine kept)
        original_hashes = set()  # content hashes that matched the original question

        # Calculate how many docs to retrieve per query
        # Target: ~30-40 total docs before dedup, then take top k
        docs_per_query = max(10, settings.rag_top_k // len(query_variations) + 5)

        async def _embed_and_search(i: int, query: str):
            embed_result = await embed_question(query)
            docs = await vector_store.hybrid_search(
                query_text=query,
                query_embedding=embed_result["embedding"],
                k=docs_per_query,
                threshold=settings.similarity_threshold,
                bm25_weight=0.4  # 60% semantic, 40% BM25
            )
            return i, embed_result["cost_usd"], docs

        query_results = await asyncio.gather(*[
            _embed_and_search(i, query) for i, query in enumerate(query_variations)
        ])

        for i, cost, docs in query_results:
            logfire.debug(f"Retrieved {len(docs)} docs for query {i+1}/{len(query_variations)}")
            total_cost += cost

            for doc in docs:
                # Use first 200 chars as content hash for dedup
                content_hash = hash(doc.content[:200])

                # The original question is at index 0 — track its hits so they win
                # ties without mutating any scores (similarity across different
                # queries isn't directly comparable, so we never boost).
                if i == 0:
                    original_hashes.add(content_hash)

                # Keep the highest cosine seen for each piece of content
                if content_hash not in all_docs or doc.similarity_score > all_docs[content_hash].similarity_score:
                    all_docs[content_hash] = doc

        # Sort by cosine; on ties prefer docs that matched the original question.
        ranked = sorted(
            all_docs.items(),
            key=lambda kv: (kv[1].similarity_score, kv[0] in original_hashes),
            reverse=True,
        )
        documents = [doc for _, doc in ranked][:settings.rag_top_k]

        logfire.info(
            "Multi-query retrieval completed",
            num_queries=len(query_variations),
            total_docs_before_dedup=len(all_docs),
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

    # Adaptive relevance cutoff BEFORE injection so curated 1.0 scores don't skew
    # the best-match anchor.
    documents = _apply_relative_cutoff(documents)

    # Relevance-gated curated-doc injection (pricing, AI agents, autoscaling,
    # Node.js, tutorials) — only adds a canonical doc if it wasn't already
    # retrieved, and never grows the set past rag_top_k.
    if original_question:
        documents = await inject_curated_docs(original_question, documents)

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
