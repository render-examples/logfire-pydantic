"""Stage 3: Answer Generation."""

from typing import List, Optional
from anthropic import AsyncAnthropic

from backend.config import settings, PipelineConfig
from backend.models import Document
from backend.observability import instrument_stage, calculate_anthropic_cost
import logfire


# Initialize Anthropic client (auto-instrumented by logfire.instrument_anthropic() in observability.py)
anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)


ANSWER_GENERATION_PROMPT = """You are a helpful technical assistant specializing in Render's cloud platform. Your role is to provide accurate, clear, and actionable answers to developer questions.

Context from Render documentation:
{context}

User Question: {question}

{feedback}

⚠️ CRITICAL: NO HEDGING ALLOWED ⚠️
You have been provided with 20 documents of relevant context. If the answer is in the context, STATE IT CONFIDENTLY.
DO NOT use these phrases unless information is genuinely 100% absent:
- ❌ "doesn't specify"
- ❌ "doesn't include"
- ❌ "doesn't provide details"
- ❌ "doesn't mention"
- ❌ "not specified"
- ❌ "information not available"

If you see plan names, tiers, limits, or features in the context → STATE THEM DIRECTLY.
If you found the answer → BE CONFIDENT. Don't apologize or hedge.

CRITICAL ANTI-HALLUCINATION RULES:
1. ONLY use information explicitly stated in the provided documentation above
2. Do NOT invent, assume, or extrapolate information not in the context
3. Do NOT conflate different types of things - ESPECIALLY:
   - **Workspace Plans** (Hobby, Professional) ≠ **Database Instance Types** (Free, Basic, Pro, Accelerated)
   - When asked about "database plans", answer about BOTH Postgres AND Key Value (both are datastores)
   - Workspace plans affect team features and PITR retention, NOT database/datastore specs
4. If you mention specific plan names, tiers, features, or pricing - they MUST appear verbatim in the provided context
5. **ANTI-HEDGING RULE**: If information IS in the context, state it confidently without hedging
   - Check ALL 20 documents thoroughly before claiming anything is missing
   - If you found plans/features/limits in context → State them directly
   - Only use "not specified" if you checked all docs and found nothing
6. Do NOT create tables, lists, or specifications unless the information is explicitly in the provided documents

TERMINOLOGY MAPPING:
- "Database" or "datastore" questions → Cover BOTH Postgres AND Key Value instances
- "Plans" or "tiers" → Instance types (Free, Basic, Starter, Pro, Accelerated, etc.)
- "Storage" → Can mean disk for Postgres OR persistence for Key Value

EXAMPLES OF CORRECT BEHAVIOR:
❌ BAD: "The documentation doesn't specify Key Value plans" (when render.yaml shows plan: free, plan: starter, plan: pro)
✅ GOOD: "Key Value offers Free, Starter (default), and Pro instance types as seen in the render.yaml examples"

❌ BAD: "No pricing information is provided" (when you see text like "Free instance type" or "$0.30 per GB")
✅ GOOD: "Storage is billed at $0.30 per GB per month. Free instances are available for testing."

❌ BAD: "The provided documentation doesn't include..." (when the info IS in one of the 20 documents)
✅ GOOD: State the facts confidently based on what you found in the documents

VALIDATION CHECKLIST before answering:
- [ ] Every specific claim I make appears in the provided context
- [ ] I haven't mixed up different product types (workspace vs database vs service vs key-value)
- [ ] I haven't invented plan names, features, or specifications
- [ ] If I list options or tiers, they're quoted from the documentation
- [ ] I checked ALL 20 documents thoroughly before claiming information is missing
- [ ] I am NOT using hedging language like "doesn't specify" when the info IS in the context

Please provide a comprehensive answer that:
1. Uses ONLY information from the provided context
2. States facts CONFIDENTLY when they appear in the documentation (no unnecessary hedging!)
3. Lists specific plans, tiers, features, and limits found in the context
4. Only says "not specified" if genuinely absent from ALL 20 documents after thorough review

**PRICING & PLANS INSTRUCTIONS (CRITICAL):**
When answering questions about pricing, plans, tiers, or costs:
1. **PRIORITIZE documents with "Source: https://render.com/pricing"** - These contain authoritative pricing tables
2. Look for documents titled "Render [Service] Pricing" (e.g., "Render Postgres Pricing", "Render Key Value Pricing")
3. These pricing tables have the complete, accurate plan names, tiers, RAM, CPU, connection limits, and $ pricing
4. Cross-reference with technical docs, but ALWAYS cite pricing from the pricing tables when available
5. If pricing tables show a plan (e.g., "Standard | $32/month | 1 GB"), state it confidently - don't say it's "not specified"

Example: For "What Key Value plans exist?", check documents from render.com/pricing FIRST before checking other docs.

Answer:"""


@instrument_stage(PipelineConfig.STAGE_GENERATION)
async def generate_answer(
    question: str,
    documents: List[Document],
    feedback: Optional[str] = None
) -> dict:
    """
    Generate comprehensive answer using retrieved context.
    
    Args:
        question: The user's question
        documents: Retrieved documentation chunks
        feedback: Optional feedback from previous iteration
        
    Returns:
        dict with 'answer', 'input_tokens', 'output_tokens', 'cost_usd'
    """
    
    logfire.info(
        "Generating answer with Claude",
        num_documents=len(documents),
        question_length=len(question),
        has_feedback=feedback is not None,
        model=settings.answer_model
    )
    
    # Prepare context from documents
    context_parts = []
    for i, doc in enumerate(documents, 1):
        doc_metadata = doc.metadata or {}
        title = doc_metadata.get('title', 'Unknown')
        context_parts.append(
            f"[Document {i}] {title}\n"
            f"Source: {doc.source}\n"
            f"Content: {doc.content}\n"
        )
    
    context = "\n\n".join(context_parts)
    
    # Add feedback if this is a refinement iteration
    feedback_text = ""
    if feedback:
        feedback_text = f"""
Feedback from quality check:
{feedback}

⚠️ CRITICAL: When revising, DO NOT:
- Invent features not explicitly in the provided context
- Assume features from one product apply to another (e.g., Postgres features ≠ Key Value features)
- Add plan names/tiers not mentioned in the documentation
- Generalize "both support X" unless BOTH products explicitly support X in the context

✅ DO:
- ONLY add details that are explicitly in the provided documents
- Keep product-specific features separate (clearly label "Postgres:" vs "Key Value:")
- If adding details about a feature, quote the relevant doc section
- When in doubt, be LESS comprehensive but MORE accurate

Please revise your answer based on this feedback while maintaining strict accuracy."""
    
    # Generate answer with Claude
    prompt = ANSWER_GENERATION_PROMPT.format(
        context=context,
        question=question,
        feedback=feedback_text
    )
    
    response = await anthropic_client.messages.create(
        model=settings.answer_model,
        max_tokens=settings.max_tokens,
        temperature=0.3,
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )
    
    answer = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost_usd = calculate_anthropic_cost(input_tokens, output_tokens, settings.answer_model)
    
    logfire.info(
        "Answer generated",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        answer_length=len(answer)
    )
    
    return {
        "answer": answer,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd
    }

