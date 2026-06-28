"""Shared section-aware ingestion for curated single-page docs.

The curated `add_*_page.py` scripts used to embed each page as a single vector,
which diluted specific facts and hurt claim-verification recall. This helper
splits the curated markdown into per-heading chunks (see
`chunk_markdown_by_heading`), embeds each chunk, and inserts them — so each
fact-cluster gets its own focused embedding while still sharing one `source`.

Because the chunks share a source, the retrieval layer's curated injection
(`inject_curated_docs`) still finds and injects the whole page by source, and
`collapse_sources` regroups the chunks for display.
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.database import vector_store  # noqa: E402
from backend.pipeline.embeddings import embed_question  # noqa: E402

from chunking import chunk_document, chunk_markdown_by_heading  # noqa: E402


async def ingest_curated_markdown(
    content: str,
    source: str,
    title: str,
    section: Optional[str] = None,
    metadata: Optional[dict] = None,
    also_delete: Optional[list] = None,
) -> int:
    """Replace any prior copy of `source`, then section-chunk, embed, and insert.

    `also_delete` is an optional list of additional source URLs to purge first
    (e.g. legacy/superseded docs that should no longer surface). Returns the
    number of chunks inserted.
    """
    await vector_store.initialize()

    async with vector_store.pool.acquire() as conn:
        for legacy_source in (also_delete or []):
            result = await conn.execute("DELETE FROM documents WHERE source = $1", legacy_source)
            print(f"   Deleted {int(result.split()[-1])} documents from {legacy_source}")
        result = await conn.execute("DELETE FROM documents WHERE source = $1", source)
        print(f"   Deleted {int(result.split()[-1])} existing documents for {source}")

    chunks = chunk_markdown_by_heading(title, source, content)
    if not chunks:
        # No headings (or content too short) — fall back to size-based chunking.
        chunks = chunk_document(title, section, source, content)

    batch = []
    for chunk in chunks:
        embed_result = await embed_question(chunk["content"])
        chunk_metadata = {
            **(metadata or {}),
            "title": chunk["title"],
            "section": chunk["section"],
        }
        batch.append((
            chunk["content"],
            source,
            chunk["title"],
            embed_result["embedding"],
            chunk["section"],
            chunk_metadata,
        ))

    if batch:
        await vector_store.insert_documents_batch(batch)

    await vector_store.close()
    print(f"Successfully added {len(batch)} chunk(s) for {source}")
    return len(batch)
