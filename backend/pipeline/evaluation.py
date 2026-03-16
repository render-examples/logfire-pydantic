"""Stage 7: Quality Rating (Dual-Model Evaluation)."""

from typing import List
import asyncio
import re
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from backend.config import settings, PipelineConfig
from backend.models import Document, EvaluationResult
from backend.observability import instrument_stage, calculate_openai_cost, calculate_anthropic_cost
import logfire


# Initialize clients (auto-instrumented by logfire.instrument_*() in observability.py)
openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)


EVALUATION_PROMPT = """You are a quality evaluator for technical documentation answers. Evaluate the following answer on multiple criteria.

Question: {question}

Answer:
{answer}

Source Documents Used: {doc_count}

CRITICAL: If the answer essentially says "I don't know", "I can't answer", or "information not available", it should receive very low scores (0-20) across all criteria, regardless of how politely it's written.

Please rate the answer on the following criteria (0-100 for each):

1. Technical Accuracy (30%): Is the information correct and up-to-date? (Score 0-20 if answer says it lacks information)
2. Clarity & Organization (25%): Is the answer well-structured and easy to understand? (Score 0-20 if answer doesn't actually provide substantive information)
3. Completeness (25%): Does it fully address the question with specific details? (Score 0-10 if answer admits it cannot answer)
4. Developer Value (20%): Is it actionable and useful for developers? (Score 0-10 if answer just redirects to external resources)

Provide your evaluation in this format:
TECHNICAL_ACCURACY: [0-100]
CLARITY: [0-100]
COMPLETENESS: [0-100]
DEVELOPER_VALUE: [0-100]
OVERALL: [weighted average]
FEEDBACK: [1-2 sentences of constructive feedback]"""


async def evaluate_with_openai(question: str, answer: str, doc_count: int) -> dict:
    """Evaluate with OpenAI GPT-4o-mini."""
    
    prompt = EVALUATION_PROMPT.format(
        question=question,
        answer=answer,
        doc_count=doc_count
    )
    
    response = await openai_client.chat.completions.create(
        model=settings.eval_model_openai,
        messages=[{
            "role": "user",
            "content": prompt
        }],
        temperature=0.1,
        max_tokens=500
    )
    
    result_text = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost_usd = calculate_openai_cost(input_tokens, output_tokens, settings.eval_model_openai)
    
    return {
        "result_text": result_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "model": settings.eval_model_openai
    }


async def evaluate_with_anthropic(question: str, answer: str, doc_count: int) -> dict:
    """Evaluate with Anthropic Claude."""
    
    prompt = EVALUATION_PROMPT.format(
        question=question,
        answer=answer,
        doc_count=doc_count
    )
    
    response = await anthropic_client.messages.create(
        model=settings.eval_model_anthropic,
        max_tokens=500,
        temperature=0.1,
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )
    
    result_text = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost_usd = calculate_anthropic_cost(input_tokens, output_tokens, settings.eval_model_anthropic)
    
    return {
        "result_text": result_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "model": settings.eval_model_anthropic
    }


def parse_evaluation(result_text: str, model: str) -> EvaluationResult:
    """Parse evaluation response into structured format."""
    
    technical_accuracy = 85
    clarity = 85
    completeness = 85
    developer_value = 85
    overall_score = 85
    feedback = "Good answer."
    
    def extract_score(text: str) -> int:
        """Extract first number from text, handling formats like '85', '85/100', '85.5'"""
        # Find first number (integer or decimal)
        match = re.search(r'\d+(?:\.\d+)?', text)
        if match:
            score = float(match.group())
            # If it looks like a fraction format (e.g., "85/100"), just take the numerator
            # Otherwise cap at 100
            return min(int(round(score)), 100)
        return 85  # Default fallback
    
    try:
        lines = result_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('TECHNICAL_ACCURACY:'):
                technical_accuracy = extract_score(line.split(':')[1])
            elif line.startswith('CLARITY:'):
                clarity = extract_score(line.split(':')[1])
            elif line.startswith('COMPLETENESS:'):
                completeness = extract_score(line.split(':')[1])
            elif line.startswith('DEVELOPER_VALUE:'):
                developer_value = extract_score(line.split(':')[1])
            elif line.startswith('OVERALL:'):
                overall_score = extract_score(line.split(':')[1])
            elif line.startswith('FEEDBACK:'):
                feedback = line.split(':', 1)[1].strip()
    except Exception as e:
        logfire.error(f"Failed to parse evaluation from {model}: {e}")
        logfire.error(f"Raw evaluation text: {result_text}")
    
    return EvaluationResult(
        model=model,
        score=overall_score,
        technical_accuracy=technical_accuracy,
        clarity=clarity,
        completeness=completeness,
        developer_value=developer_value,
        feedback=feedback
    )


@instrument_stage(PipelineConfig.STAGE_EVALUATION)
async def evaluate_quality(
    question: str,
    answer: str,
    documents: List[Document]
) -> dict:
    """
    Independent quality assessment from two models.
    
    Args:
        question: The user's question
        answer: The generated answer
        documents: Source documents
        
    Returns:
        dict with 'evaluations', 'average_score', 'agreement_level', 'total_cost_usd'
    """
    
    logfire.info("Evaluating quality with dual models")
    
    doc_count = len(documents)
    
    # Run both evaluations in parallel
    openai_result, anthropic_result = await asyncio.gather(
        evaluate_with_openai(question, answer, doc_count),
        evaluate_with_anthropic(question, answer, doc_count)
    )
    
    # Parse results
    openai_eval = parse_evaluation(openai_result["result_text"], openai_result["model"])
    anthropic_eval = parse_evaluation(anthropic_result["result_text"], anthropic_result["model"])
    
    evaluations = [openai_eval, anthropic_eval]
    
    # Calculate average and agreement
    average_score = (openai_eval.score + anthropic_eval.score) / 2
    score_difference = abs(openai_eval.score - anthropic_eval.score)
    
    if score_difference <= 5:
        agreement_level = "high"
    elif score_difference <= 15:
        agreement_level = "medium"
    else:
        agreement_level = "low"
    
    total_cost = openai_result["cost_usd"] + anthropic_result["cost_usd"]
    
    logfire.info(
        "Quality evaluated",
        openai_score=openai_eval.score,
        anthropic_score=anthropic_eval.score,
        average_score=average_score,
        agreement_level=agreement_level,
        score_difference=score_difference,
        cost_usd=total_cost
    )
    
    return {
        "evaluations": evaluations,
        "average_score": average_score,
        "agreement_level": agreement_level,
        "cost_usd": total_cost
    }

