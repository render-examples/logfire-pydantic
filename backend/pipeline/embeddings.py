"""Stage 1: Question Embedding."""

from openai import AsyncOpenAI
import tiktoken

from backend.config import settings, PipelineConfig
from backend.observability import instrument_stage, calculate_embedding_cost
import logfire


# Initialize OpenAI client (auto-instrumented by logfire.instrument_openai() in observability.py)
openai_client = AsyncOpenAI(api_key=settings.openai_api_key)


@instrument_stage(PipelineConfig.STAGE_EMBEDDING)
async def embed_question(question: str) -> dict:
    """
    Convert natural language question to vector representation.
    
    Args:
        question: The user's question
        
    Returns:
        dict with 'embedding', 'tokens', 'cost_usd'
    """
    
    logfire.info(f"Embedding question: {question[:100]}...")
    
    # Calculate tokens for cost estimation
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = len(encoding.encode(question))
    
    # Create embedding
    response = await openai_client.embeddings.create(
        model=settings.embedding_model,
        input=question,
        dimensions=settings.embedding_dimensions
    )
    
    embedding = response.data[0].embedding
    cost_usd = calculate_embedding_cost(tokens)
    
    logfire.info(
        "Question embedded",
        tokens=tokens,
        cost_usd=cost_usd,
        embedding_length=len(embedding)
    )
    
    return {
        "embedding": embedding,
        "tokens": tokens,
        "cost_usd": cost_usd
    }

