"""Stage 4: Claims Extraction."""

from typing import List
import json
from openai import AsyncOpenAI

from backend.config import settings, PipelineConfig
from backend.observability import instrument_stage, calculate_openai_cost
import logfire


# Initialize OpenAI client (auto-instrumented by logfire.instrument_openai() in observability.py)
openai_client = AsyncOpenAI(api_key=settings.openai_api_key)


CLAIMS_EXTRACTION_PROMPT = """Extract all factual claims from the following answer. A factual claim is a specific, verifiable statement about Render's platform, features, pricing, or capabilities.

Answer:
{answer}

Extract claims as a JSON array of strings. Each claim should be:
- A single, specific fact (one sentence)
- Kept on a single line (no line breaks within the claim text)
- Independently verifiable
- Technical or product-related

IMPORTANT: Each claim must be a complete sentence on ONE line. Do NOT break claims across multiple lines.

Return a JSON object with a "claims" array:
{{
  "claims": [
    "Render supports Node.js versions 14, 16, 18, and 20",
    "PostgreSQL databases include automated daily backups"
  ]
}}"""


@instrument_stage(PipelineConfig.STAGE_CLAIMS)
async def extract_claims(answer: str) -> dict:
    """
    Extract verifiable factual claims from generated answer.
    
    Args:
        answer: The generated answer text
        
    Returns:
        dict with 'claims', 'input_tokens', 'output_tokens', 'cost_usd'
    """
    
    logfire.info(
        "Extracting claims from answer",
        answer_length=len(answer),
        model=settings.claims_model
    )
    
    prompt = CLAIMS_EXTRACTION_PROMPT.format(answer=answer)
    
    response = await openai_client.chat.completions.create(
        model=settings.claims_model,
        messages=[{
            "role": "user",
            "content": prompt
        }],
        temperature=0.1,
        max_tokens=4000,  # Increased for comprehensive pricing answers with many claims
        response_format={"type": "json_object"}  # Force valid JSON output
    )
    
    claims_text = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost_usd = calculate_openai_cost(input_tokens, output_tokens, settings.claims_model)
    finish_reason = response.choices[0].finish_reason
    
    # Check if we hit the token limit (response was truncated)
    if finish_reason == "length":
        logfire.warn(
            "Claims extraction hit max_tokens limit - response truncated",
            max_tokens=4000,
            output_tokens=output_tokens,
            answer_length=len(answer),
            response_preview=claims_text[-200:] if claims_text else ""  # Show end to see truncation
        )
    
    logfire.debug(
        "Claims extraction API response received",
        response_length=len(claims_text) if claims_text else 0,
        response_preview=claims_text[:200] if claims_text else "",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        finish_reason=finish_reason
    )
    
    # Parse JSON claims - json_object mode ensures valid JSON
    claims = []
    try:
        parsed = json.loads(claims_text)
        
        # Extract claims array from object
        if isinstance(parsed, dict):
            
            # CRITICAL FIX: JSON keys may have whitespace - normalize them  
            # LLM may generate: {\n  "claims": [...]} where key literally has \n in it
            claims_found = False
            try:
                # Normalize keys by stripping whitespace
                normalized_dict = {}
                for k, v in parsed.items():
                    clean_key = k.strip() if isinstance(k, str) else k
                    normalized_dict[clean_key] = v
                
                # Try direct dictionary get on normalized dict
                if "claims" in normalized_dict:
                    claims = normalized_dict.get("claims", [])
                    claims_found = True
                else:
                    logfire.warn(
                        "'claims' key not found in response",
                        available_keys=list(normalized_dict.keys())
                    )
                    
            except Exception as norm_err:
                logfire.error(
                    "Error normalizing dict keys",
                    error_type=type(norm_err).__name__,
                    error_message=str(norm_err)
                )
            
            if not claims_found:
                # Try to get first list value from parsed dict as fallback
                logfire.warn("Attempting fallback: extracting first list value")
                try:
                    for value in parsed.values():
                        if isinstance(value, list):
                            claims = value
                            claims_found = True
                            logfire.info(
                                "Found claims via fallback",
                                claim_count=len(value)
                            )
                            break
                except Exception as val_err:
                    logfire.error(
                        "Error in fallback extraction",
                        error=str(val_err)
                    )
            
            # Validate claims is a list
            if not isinstance(claims, list):
                logfire.warn(
                    "Claims is not a list",
                    actual_type=type(claims).__name__
                )
                claims = []
            elif len(claims) == 0:
                logfire.warn(
                    "Empty claims list extracted",
                    answer_length=len(answer)
                )
                
        elif isinstance(parsed, list):
            # Fallback: handle flat array format
            claims = parsed
            logfire.debug(
                "Received flat array format",
                claim_count=len(parsed)
            )
        else:
            logfire.warn(
                "Unexpected JSON type from LLM",
                actual_type=type(parsed).__name__
            )
            
    except json.JSONDecodeError as je:
        logfire.error(
            "JSON decode error in claims extraction",
            error=str(je),
            response_preview=claims_text[:500] if claims_text else ""
        )
        claims = []
    except Exception as e:
        logfire.error(
            "Unexpected error in claims parsing",
            error_type=type(e).__name__,
            error_message=str(e),
            response_preview=claims_text[:300] if claims_text else ""
        )
        claims = []
    
    # Log the result - warn if we got 0 claims from a non-empty answer
    if len(claims) == 0 and len(answer) > 100:
        logfire.warn(
            "Zero claims extracted from substantial answer",
            claim_count=0,
            answer_length=len(answer),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            finish_reason=finish_reason
        )
    else:
        logfire.info(
            "Claims extracted successfully",
            claim_count=len(claims),
            answer_length=len(answer),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd
        )
    
    return {
        "claims": claims,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd
    }

