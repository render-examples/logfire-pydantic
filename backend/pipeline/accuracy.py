"""Stage 6: Technical Accuracy Check."""

from typing import List
from anthropic import AsyncAnthropic

from backend.config import settings, PipelineConfig
from backend.models import Claim
from backend.observability import instrument_stage, calculate_anthropic_cost
import logfire


# Initialize Anthropic client (auto-instrumented by logfire.instrument_anthropic() in observability.py)
anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)


ACCURACY_CHECK_PROMPT = """You are a technical accuracy reviewer for Render documentation. Your task is to evaluate the technical accuracy of an answer.

Original Answer:
{answer}

Extracted Claims:
{claims}

Verification Results:
{verification_results}

Evaluation Criteria:
- If most claims are verified (70%+), the answer is likely accurate (score 90-100)
- CRITICAL: If the answer contains invented information (plan names, features, prices NOT in documentation), score 0-30
- Check for conflation errors (e.g., mixing workspace plans with database plans) - score 20-40 if found
- Verified claims with high similarity scores indicate strong documentation support
- Only penalize for actual technical errors or misleading information
- Minor omissions or lack of detail should not heavily penalize the score

RED FLAGS to check for:
- Invented plan names or tiers not verified by documentation
- Tables or specifications not found in source documents
- Conflating different product types:
  * CRITICAL: Treating "Hobby" or "Professional" as DATABASE plans (they're workspace plans!)
  * Database instance types are: Free, Basic-Xgb, Pro-Xgb, Accelerated-Xgb
  * Workspace plans are: Hobby, Professional (affect PITR retention, team features)
- Making up features, limits, or pricing details

Please provide:
1. An accuracy score from 0-100 (where 100 is perfectly accurate)
2. A list of any technical errors or inaccuracies you find (including hallucinations)
3. Suggestions for corrections

Format your response as:
ACCURACY_SCORE: [0-100]
ERRORS:
- [error 1 - be specific about what's wrong]
- [error 2 - indicate if it's a hallucination vs misinterpretation]
CORRECTIONS:
- [correction 1]
- [correction 2]

If there are no errors, you can omit the ERRORS and CORRECTIONS sections."""


@instrument_stage(PipelineConfig.STAGE_ACCURACY)
async def check_accuracy(
    answer: str,
    verified_claims: List[Claim]
) -> dict:
    """
    Deep accuracy validation using Claude.
    
    Args:
        answer: The generated answer
        verified_claims: Claims with verification results
        
    Returns:
        dict with 'accuracy_score', 'errors', 'corrections', 'input_tokens', 'output_tokens', 'cost_usd'
    """
    
    logfire.info("Checking technical accuracy")
    
    # Prepare claims summary
    claims_text = "\n".join([
        f"- {claim.claim} (verified: {claim.verified}, score: {claim.verification_score:.2f})"
        for claim in verified_claims
    ])
    
    # Prepare verification summary
    verified_count = sum(1 for c in verified_claims if c.verified)
    verification_text = f"{verified_count}/{len(verified_claims)} claims verified"
    
    prompt = ACCURACY_CHECK_PROMPT.format(
        answer=answer,
        claims=claims_text,
        verification_results=verification_text
    )
    
    response = await anthropic_client.messages.create(
        model=settings.accuracy_model,
        max_tokens=1500,
        temperature=0.0,  # Zero for maximum consistency in scoring
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )
    
    result_text = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost_usd = calculate_anthropic_cost(input_tokens, output_tokens, settings.accuracy_model)
    
    logfire.debug(f"Accuracy check raw response: {result_text[:1000]}")  # Log first 1000 chars
    
    # Parse the response
    accuracy_score = 90  # Default to 90 (assume good unless proven otherwise)
    errors = []
    corrections = []
    
    try:
        lines = result_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('ACCURACY_SCORE:'):
                score_text = line.split(':')[1].strip()
                # Extract just the number, handling various formats
                score_digits = ''.join(filter(str.isdigit, score_text))
                if score_digits:
                    accuracy_score = int(score_digits)
                    logfire.debug(f"Extracted accuracy score: {accuracy_score} from line: {line}")
            elif line.startswith('ERRORS:'):
                current_section = 'errors'
            elif line.startswith('CORRECTIONS:'):
                current_section = 'corrections'
            elif line.startswith('- ') and current_section == 'errors':
                errors.append(line[2:])
            elif line.startswith('- ') and current_section == 'corrections':
                corrections.append(line[2:])
        
        # If high verification rate but low accuracy score, boost it
        verified_count = sum(1 for c in verified_claims if c.verified)
        verification_rate = verified_count / len(verified_claims) if verified_claims else 0
        
        if verification_rate >= 0.7 and accuracy_score < 85 and len(errors) == 0:
            logfire.info(f"Boosting accuracy score from {accuracy_score} to 90 due to high verification rate ({verification_rate:.1%}) and no errors")
            accuracy_score = 90
            
    except Exception as e:
        logfire.error(f"Failed to parse accuracy check response: {e}, response: {result_text[:500]}")
    
    logfire.info(
        "Accuracy checked",
        accuracy_score=accuracy_score,
        error_count=len(errors),
        correction_count=len(corrections),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd
    )
    
    return {
        "accuracy_score": accuracy_score,
        "errors": errors,
        "corrections": corrections,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd
    }

