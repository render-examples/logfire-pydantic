"""Query expansion for improved retrieval diversity and coverage."""

import json
from typing import List
from openai import AsyncOpenAI

from backend.config import settings
from backend.observability import calculate_openai_cost
import logfire


# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=settings.openai_api_key)


QUERY_EXPANSION_PROMPT = """You are a search query expert for Render's cloud platform documentation.

Given a user's question, generate 2-3 alternative phrasings that would help retrieve comprehensive documentation from different angles.

Original question: {question}

Guidelines:
1. **Product Coverage**: If the question is about a general category (e.g., "databases"), explicitly mention ALL relevant products:
   - Databases → Mention both "Postgres" AND "Key Value"
   - Services → Mention "web services", "workers", "cron jobs"
   - Storage → Mention "Postgres storage", "disk storage", "persistent volumes"

2. **Terminology Variations**: Use synonyms and alternative terms:
   - database/datastore
   - plan/tier/instance type
   - pricing/cost/billing
   - backup/recovery/restore

3. **Different Aspects**: Approach from different angles:
   - Features/capabilities
   - Configuration/setup
   - Pricing/plans
   - Limitations/restrictions

4. **Specificity Balance**: Mix general and specific queries:
   - One broad query (covers category)
   - One or two specific queries (target specific products)

Return ONLY a JSON array of 2-3 questions. The first should be the original question (possibly slightly rephrased), followed by 1-2 variations.

Format: ["original or slightly rephrased", "variation 1", "variation 2"]

Example:
Input: "What database plans does Render offer?"
Output: [
  "What database plans and tiers are available on Render?",
  "What are the Postgres instance types and pricing?",
  "What Key Value datastore plans does Render provide?"
]
"""


async def expand_query(question: str) -> tuple[List[str], float]:
    """
    Use LLM to generate query variations for better retrieval coverage.
    
    Args:
        question: Original user question
        
    Returns:
        Tuple of (list of query variations, cost in USD)
    """
    
    logfire.info("Expanding query with LLM", original_question=question)
    
    prompt = QUERY_EXPANSION_PROMPT.format(question=question)
    
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",  # Fast and cheap for query expansion
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,  # Some creativity, but not too much
        max_tokens=300
    )
    
    content = response.choices[0].message.content.strip()
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost_usd = calculate_openai_cost(input_tokens, output_tokens, "gpt-4o-mini")
    
    # Handle markdown code blocks
    if content.startswith("```"):
        lines = content.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = '\n'.join(lines)
    
    try:
        variations = json.loads(content)
        
        # Validate and limit to 3 queries
        if not isinstance(variations, list):
            logfire.warning("Query expansion didn't return a list, using original question")
            variations = [question]
        else:
            variations = variations[:2]  # Limit to 2 expanded queries
            
            # ALWAYS prepend the original question to ensure it's included
            # This prevents LLM rephrasing from losing the best-matching query
            variations = [question] + variations
        
        logfire.info(
            "Query expanded successfully",
            num_variations=len(variations),
            variations=variations,
            cost_usd=cost_usd
        )
        
        return variations, cost_usd
        
    except json.JSONDecodeError as e:
        logfire.error(f"Failed to parse query expansion JSON: {e}, using original question")
        return [question], cost_usd


async def should_expand_query(question: str) -> bool:
    """
    Determine if a question would benefit from query expansion.
    
    Some questions are specific enough that expansion isn't needed.
    Others are broad/ambiguous and benefit from multiple perspectives.
    
    Args:
        question: User's question
        
    Returns:
        True if query expansion would help
    """
    
    question_lower = question.lower()
    
    # Broad category questions that need expansion
    broad_terms = [
        "database", "plan", "tier", "option", "service",
        "storage", "backup", "monitoring", "scaling",
        "pricing", "cost", "feature", "capability"
    ]
    
    # Very specific questions that don't need expansion
    specific_indicators = [
        "how do i", "how to", "error", "troubleshoot",
        "specific", "exactly", "step by step"
    ]
    
    # Check for broad terms
    has_broad_term = any(term in question_lower for term in broad_terms)
    
    # Check for specific indicators
    is_specific = any(indicator in question_lower for indicator in specific_indicators)
    
    # Check length (very long questions are usually specific)
    is_detailed = len(question.split()) > 15
    
    # Expand if broad AND not specific
    should_expand = has_broad_term and not (is_specific or is_detailed)
    
    logfire.debug(
        "Query expansion decision",
        question=question,
        should_expand=should_expand,
        has_broad_term=has_broad_term,
        is_specific=is_specific,
        is_detailed=is_detailed
    )
    
    return should_expand

