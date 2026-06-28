"""Shared chunking helper for documentation ingestion.

Splits a single page of content into ~2000-character chunks at paragraph
boundaries, producing the doc dicts ({title, section, source, content}) consumed
by generate_embeddings.py and crawl_tutorials.py. Keeping this in one place means
the bulk-docs corpus and the tutorials crawl chunk content identically.
"""

import re
from typing import Dict, List, Optional

MAX_CHUNK_CHARS = 2000
MIN_CHUNK_CHARS = 100

# Sections shorter than this are merged into the previous chunk rather than
# becoming their own (a heading with almost no body isn't worth its own embedding).
MERGE_FLOOR_CHARS = 60

# Matches level-2 and level-3 ATX markdown headings (## / ###).
_HEADING_RE = re.compile(r'^(#{2,3})\s+(.+?)\s*$', re.MULTILINE)


def chunk_document(
    title: str,
    section: Optional[str],
    source: str,
    content: str,
    max_chars: int = MAX_CHUNK_CHARS,
) -> List[Dict]:
    """Split content into chunks of ~max_chars, splitting at paragraph boundaries.

    Returns a list of doc dicts. Content shorter than MIN_CHUNK_CHARS yields no
    chunks. The display title combines the page title and section when a section
    is present, matching the existing corpus convention.
    """
    clean_content = content.strip()
    if len(clean_content) < MIN_CHUNK_CHARS:
        return []

    display_title = title if not section else f"{title} - {section}"
    display_section = section or title

    def _doc(chunk_content: str) -> Dict:
        return {
            "title": display_title,
            "section": display_section,
            "source": source,
            "content": chunk_content,
        }

    if len(clean_content) <= max_chars:
        return [_doc(clean_content)]

    # Prefer paragraph boundaries; fall back to single newlines for content that
    # arrives as one block (e.g. text scraped from HTML, which has no blank lines).
    separator = "\n\n"
    segments = clean_content.split(separator)
    if len(segments) == 1:
        separator = "\n"
        segments = clean_content.split(separator)

    # Hard-split any segment still larger than max_chars so no chunk is oversized.
    bounded_segments: List[str] = []
    for segment in segments:
        if len(segment) <= max_chars:
            bounded_segments.append(segment)
        else:
            for start in range(0, len(segment), max_chars):
                bounded_segments.append(segment[start:start + max_chars])

    docs: List[Dict] = []
    current_chunk: List[str] = []
    current_length = 0
    sep_len = len(separator)

    for segment in bounded_segments:
        segment_length = len(segment)
        if current_length + segment_length > max_chars and current_chunk:
            docs.append(_doc(separator.join(current_chunk)))
            current_chunk = [segment]
            current_length = segment_length
        else:
            current_chunk.append(segment)
            current_length += segment_length + sep_len

    if current_chunk:
        docs.append(_doc(separator.join(current_chunk)))

    return docs


def chunk_markdown_by_heading(
    title: str,
    source: str,
    content: str,
    max_chars: int = MAX_CHUNK_CHARS,
) -> List[Dict]:
    """Split structured markdown into one chunk per ``##``/``###`` heading section.

    Each heading section becomes its own focused embedding, so a specific fact
    (e.g. "task runs can execute for up to 24 hours") isn't diluted inside a
    page-sized chunk. This noticeably improves retrieval and claim-verification
    recall for curated docs versus embedding the whole page as one vector.

    Behavior:
      - The heading text is kept in the chunk so its vocabulary stays searchable.
      - Content before the first heading (typically an H1 title + a "Source:"
        line) is dropped as preamble.
      - Sections larger than ``max_chars`` fall back to size-based splitting via
        ``chunk_document`` (heading preserved as the section label).
      - Sections shorter than ``MERGE_FLOOR_CHARS`` are merged into the previous
        chunk.
      - Falls back entirely to ``chunk_document`` when the content has no
        ``##``/``###`` headings.

    Returns the same doc-dict shape as ``chunk_document``.
    """
    clean_content = content.strip()
    if len(clean_content) < MIN_CHUNK_CHARS:
        return []

    matches = list(_HEADING_RE.finditer(clean_content))
    if not matches:
        return chunk_document(title, None, source, content, max_chars=max_chars)

    docs: List[Dict] = []
    for idx, match in enumerate(matches):
        heading = match.group(2).strip()
        body_start = match.end()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(clean_content)
        body = clean_content[body_start:body_end].strip()

        section_text = f"## {heading}\n\n{body}".strip() if body else f"## {heading}"

        # Merge tiny sections into the previous chunk instead of embedding them alone.
        if len(section_text) < MERGE_FLOOR_CHARS and docs:
            docs[-1]["content"] = f"{docs[-1]['content']}\n\n{section_text}"
            continue

        if len(section_text) <= max_chars:
            docs.append({
                "title": f"{title} - {heading}",
                "section": heading,
                "source": source,
                "content": section_text,
            })
        else:
            # Oversized section: size-split it while keeping the heading as the label.
            docs.extend(chunk_document(title, heading, source, section_text, max_chars=max_chars))

    return docs
