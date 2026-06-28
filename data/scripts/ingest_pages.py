"""In-process ingestion entrypoint — no Render Workflows.

Used by the deploy pre-deploy hook and the refresh cron (see ``render.yaml``),
and for local dev. Same registry and shared build → embed → replace-by-source
helpers the old ``ingest_all`` Workflow used.

Run with NO arguments for a full refresh: it loads the pre-embedded core corpus
(``data/embeddings/render_docs.json``, additive sync — no embedding cost), then
ingests every live source. Pass source names to refresh just those (no core load).

Usage:

    python data/scripts/ingest_pages.py            # core corpus + all live sources
    python data/scripts/ingest_pages.py pricing    # one live source only
    python data/scripts/ingest_pages.py pricing nodejs   # several
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

from backend.database import vector_store
from backend.ingestion import embed_documents, replace_source
from data.sources import SOURCES

load_dotenv()


async def _ingest_one(name: str) -> None:
    src = SOURCES[name]
    docs = await embed_documents(await src.build())
    inserted = await replace_source(src.source_url, docs, legacy=src.legacy_sources)
    print(f"  {name}: inserted {inserted} document(s) for {src.source_url}")


async def main(names: list[str], include_core: bool = False) -> None:
    # Core corpus first (establishes base rows). ingest_docs.main manages its own
    # pool lifecycle, so re-initialize before the live-source pass below.
    if include_core:
        from data.scripts import ingest_docs

        print("Loading core corpus (additive sync)...")
        await ingest_docs.main(sync=True)

    if names:
        print(f"Ingesting live sources: {', '.join(names)}")
        await vector_store.initialize()
        try:
            for name in names:
                await _ingest_one(name)
        finally:
            await vector_store.close()
    print("Done.")


if __name__ == "__main__":
    requested = sys.argv[1:]
    # No args => full refresh: core corpus + every live source.
    include_core = not requested
    names = requested or list(SOURCES)
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        sys.exit(f"Unknown source(s): {unknown}. Available: {list(SOURCES)}")
    asyncio.run(main(names, include_core=include_core))
