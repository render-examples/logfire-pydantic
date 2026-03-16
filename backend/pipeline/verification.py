"""Stage 5: Claims Verification."""

from typing import List
from openai import AsyncOpenAI

from backend.config import settings, PipelineConfig
from backend.database import vector_store
from backend.models import Claim
from backend.observability import instrument_stage, calculate_embedding_cost
import logfire


# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=settings.openai_api_key)


@instrument_stage(PipelineConfig.STAGE_VERIFICATION)
async def verify_claims(claims: List[str]) -> dict:
    """
    Verify each claim against documentation using RAG.
    
    Args:
        claims: List of claim strings to verify
        
    Returns:
        dict with 'verified_claims', 'verification_rate', 'cost_usd'
    """
    
    logfire.info(f"Verifying {len(claims)} claims")
    
    verified_claims: List[Claim] = []
    total_cost = 0.0
    total_tokens = 0
    
    # Verify each claim
    for claim_text in claims:
        # Embed the claim
        response = await openai_client.embeddings.create(
            model=settings.embedding_model,
            input=claim_text,
            dimensions=settings.embedding_dimensions
        )
        
        embedding = response.data[0].embedding
        claim_tokens = len(claim_text.split()) * 1.3  # Rough estimate
        total_tokens += claim_tokens
        
        # Search for supporting documents - use a lower threshold for search to get candidates
        docs = await vector_store.similarity_search(
            query_embedding=embedding,
            k=5,  # Top 5 docs for verification (more candidates)
            threshold=0.3  # Lower threshold to get candidates, then we filter by verification_threshold
        )
        
        # Determine if claim is verified
        verified = False
        verification_score = 0.0
        supporting_docs = []
        
        if docs:
            # Consider verified if at least one doc has similarity >= verification_threshold
            verification_score = docs[0].similarity_score
            
            # BOOST: Prioritize pricing table sources for pricing-related claims
            # If claim mentions pricing/plans/costs and doc is from render.com/pricing, boost confidence
            is_pricing_claim = any(term in claim_text.lower() for term in ['$', 'pricing', 'price', 'cost', 'plan', 'tier', 'gb', 'ram', 'cpu'])
            is_pricing_source = docs[0].source == "https://render.com/pricing"
            
            if is_pricing_claim and is_pricing_source:
                # Boost verification score by 10% for pricing claims with pricing table sources
                verification_score = min(1.0, verification_score * 1.1)
                logfire.debug(f"Boosted pricing claim verification: {verification_score:.3f}")
            
            verified = verification_score >= settings.verification_threshold
            
            # Collect docs that meet the threshold
            verified_docs = [doc for doc in docs if doc.similarity_score >= settings.verification_threshold]
            supporting_docs = [doc.source for doc in verified_docs[:2]]
            
            # If we have verified docs, use the highest score
            if verified_docs:
                verification_score = max(verification_score, verified_docs[0].similarity_score)
        
        logfire.debug(f"Claim verification: '{claim_text[:50]}...' - verified={verified}, score={verification_score:.3f}, docs_found={len(docs)}, verified_docs={len(supporting_docs)}, threshold={settings.verification_threshold}")
        
        verified_claims.append(Claim(
            claim=claim_text,
            verified=verified,
            verification_score=verification_score,
            supporting_docs=supporting_docs
        ))
    
    # Calculate costs
    cost_usd = calculate_embedding_cost(int(total_tokens)) + (len(claims) * 0.0001)
    total_cost += cost_usd
    
    # Calculate verification rate
    verified_count = sum(1 for c in verified_claims if c.verified)
    verification_rate = verified_count / len(verified_claims) if verified_claims else 0.0
    
    logfire.info(
        "Claims verified",
        total_claims=len(verified_claims),
        verified_count=verified_count,
        verification_rate=verification_rate,
        cost_usd=cost_usd
    )
    
    return {
        "verified_claims": verified_claims,
        "verification_rate": verification_rate,
        "cost_usd": cost_usd
    }

