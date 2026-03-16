"""FastAPI application for Render Q&A Assistant."""

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import logfire
from opentelemetry import trace

from backend.config import settings
from backend.models import (
    QuestionRequest,
    AnswerResponse,
    HealthCheck,
    ProgressUpdate,
    PipelineStageResult,
)
from backend.database import vector_store
from backend.observability import pipeline_trace, track_pipeline_metrics
from backend.pipeline import (
    embed_question,
    retrieve_documents,
    generate_answer,
    extract_claims,
    verify_claims,
    check_accuracy,
    evaluate_quality,
    quality_gate_decision,
)
from backend.api.logs import fetch_logfire_logs


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    
    # Startup
    logfire.info("Starting Render Q&A Assistant")
    await vector_store.initialize()
    logfire.info("Application started successfully")
    
    yield
    
    # Shutdown
    logfire.info("Shutting down Render Q&A Assistant")
    await vector_store.close()
    logfire.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Render Q&A Assistant",
    description="Production-grade AI pipeline with observable AI using LangChain, Logfire, and Render",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument FastAPI with Logfire for automatic HTTP tracing
logfire.instrument_fastapi(app)


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "name": "Render Q&A Assistant",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    
    db_healthy = await vector_store.health_check()
    
    return HealthCheck(
        status="healthy" if db_healthy else "degraded",
        database_connected=db_healthy,
        logfire_enabled=True
    )


async def execute_pipeline(question: str, session_id: str = None) -> AnswerResponse:
    """Execute the full 8-stage pipeline."""
    
    async with pipeline_trace(question) as context:
        stages = []
        total_cost = 0.0
        pipeline_start = time.time()
        
        # Stage 1: Question Embedding
        embed_result = await embed_question(question)
        stage_result = PipelineStageResult(
            stage="question_embedding",
            success=True,
            duration_ms=0,  # Tracked by decorator
            cost_usd=embed_result["cost_usd"],
            tokens_used=embed_result["tokens"],
            metadata={"embedding_dimensions": len(embed_result["embedding"])}
        )
        stages.append(stage_result)
        total_cost += embed_result["cost_usd"]
        
        # Stage 2: RAG Retrieval (with multi-query expansion for broad questions)
        retrieval_result = await retrieve_documents(embed_result["embedding"], original_question=question)
        stage_result = PipelineStageResult(
            stage="rag_retrieval",
            success=True,
            duration_ms=0,
            cost_usd=retrieval_result["cost_usd"],
            metadata={
                "documents_retrieved": len(retrieval_result["documents"]),
                "queries_expanded": retrieval_result.get("queries_count", 1)
            }
        )
        stages.append(stage_result)
        total_cost += retrieval_result["cost_usd"]
        
        documents = retrieval_result["documents"]
        
        # Iterative quality refinement loop
        current_iteration = 1
        feedback = None
        answer_text = ""
        verified_claims = []
        accuracy_score = 0
        evaluations = []
        average_score = 0
        
        while current_iteration <= settings.max_iterations:
            logfire.info(f"Starting iteration {current_iteration}")
            
            # Stage 3: Answer Generation
            gen_result = await generate_answer(question, documents, feedback)
            stage_result = PipelineStageResult(
                stage=f"answer_generation_iter_{current_iteration}",
                success=True,
                duration_ms=0,
                cost_usd=gen_result["cost_usd"],
                tokens_used=gen_result["input_tokens"] + gen_result["output_tokens"],
                metadata={
                    "answer_length": len(gen_result["answer"]),
                    "iteration": current_iteration
                }
            )
            stages.append(stage_result)
            total_cost += gen_result["cost_usd"]
            answer_text = gen_result["answer"]
            
            # Stage 4: Claims Extraction
            claims_result = await extract_claims(answer_text)
            claims_count = len(claims_result["claims"])
            stage_result = PipelineStageResult(
                stage=f"claims_extraction_iter_{current_iteration}",
                success=True,
                duration_ms=0,
                cost_usd=claims_result["cost_usd"],
                tokens_used=claims_result["input_tokens"] + claims_result["output_tokens"],
                metadata={
                    "claims_extracted": claims_count,
                    "iteration": current_iteration
                }
            )
            stages.append(stage_result)
            total_cost += claims_result["cost_usd"]
            
            # Stage 5: Claims Verification
            verification_result = await verify_claims(claims_result["claims"])
            verified_claims = verification_result["verified_claims"]
            verification_rate = verification_result["verification_rate"] * 100
            verified_count = len([c for c in verified_claims if c.verified])
            stage_result = PipelineStageResult(
                stage=f"claims_verification_iter_{current_iteration}",
                success=True,
                duration_ms=0,
                cost_usd=verification_result["cost_usd"],
                metadata={
                    "claims_verified": verified_count,
                    "total_claims": len(verified_claims),
                    "verification_rate": f"{verification_rate:.0f}%",
                    "iteration": current_iteration
                }
            )
            stages.append(stage_result)
            total_cost += verification_result["cost_usd"]
            
            # Stage 6: Technical Accuracy
            accuracy_result = await check_accuracy(answer_text, verified_claims)
            accuracy_score = accuracy_result["accuracy_score"]
            stage_result = PipelineStageResult(
                stage=f"technical_accuracy_iter_{current_iteration}",
                success=True,
                duration_ms=0,
                cost_usd=accuracy_result["cost_usd"],
                tokens_used=accuracy_result["input_tokens"] + accuracy_result["output_tokens"],
                metadata={
                    "accuracy_score": accuracy_score,
                    "iteration": current_iteration
                }
            )
            stages.append(stage_result)
            total_cost += accuracy_result["cost_usd"]
            
            # Stage 7: Quality Evaluation
            eval_result = await evaluate_quality(question, answer_text, documents)
            evaluations = eval_result["evaluations"]
            average_score = eval_result["average_score"]
            stage_result = PipelineStageResult(
                stage=f"quality_evaluation_iter_{current_iteration}",
                success=True,
                duration_ms=0,
                cost_usd=eval_result["cost_usd"],
                metadata={
                    "quality_score": f"{average_score:.1f}",
                    "openai_score": evaluations[0].score if len(evaluations) > 0 else None,
                    "claude_score": evaluations[1].score if len(evaluations) > 1 else None,
                    "agreement": eval_result.get("agreement_level", "unknown"),
                    "iteration": current_iteration
                }
            )
            stages.append(stage_result)
            total_cost += eval_result["cost_usd"]
            
            # Stage 8: Quality Gate
            gate_result = await quality_gate_decision(
                average_score=average_score,
                evaluations=evaluations,
                accuracy_score=accuracy_score,
                current_iteration=current_iteration,
                errors=accuracy_result["errors"],
                corrections=accuracy_result["corrections"]
            )
            stage_result = PipelineStageResult(
                stage=f"quality_gate_iter_{current_iteration}",
                success=True,
                duration_ms=0,
                cost_usd=0.0,
                metadata={
                    "should_iterate": gate_result["should_iterate"],
                    "reason": gate_result["reason"],
                    "iteration": current_iteration
                }
            )
            stages.append(stage_result)
            
            # Check if we should iterate
            if not gate_result["should_iterate"]:
                logfire.info(f"Quality gate passed: {gate_result['reason']}")
                break
            
            logfire.info(f"Quality gate requires iteration: {gate_result['reason']}")
            feedback = gate_result["feedback"]
            current_iteration += 1
        
        # Calculate total duration
        total_duration_ms = (time.time() - pipeline_start) * 1000
        
        # Create response
        response = AnswerResponse(
            question=question,
            answer=answer_text,
            sources=documents,
            claims=verified_claims,
            quality_score=average_score,
            accuracy_score=accuracy_score,
            evaluations=evaluations,
            iterations=current_iteration,
            total_cost=total_cost,
            total_duration_ms=total_duration_ms,
            stages=stages,
            session_id=session_id
        )
        
        # Track custom metrics for observability
        track_pipeline_metrics(
            question=question,
            total_cost=total_cost,
            total_duration_ms=total_duration_ms,
            quality_score=average_score,
            accuracy_score=accuracy_score,
            iterations=current_iteration,
            session_id=session_id
        )
        
        return response


@app.post("/ask", response_model=AnswerResponse, tags=["Q&A"])
async def ask_question(request: QuestionRequest):
    """
    Ask a question and get a high-quality answer.
    
    This endpoint executes the full 8-stage pipeline including:
    - Question embedding
    - RAG document retrieval
    - Answer generation
    - Claims extraction & verification
    - Technical accuracy check
    - Dual-model quality evaluation
    - Quality gate decision (with iterative refinement)
    """
    
    # Create session-level span for end-to-end user journey tracking
    with logfire.span(
        "user_session.qa_request",
        session_id=request.session_id or "anonymous",
        question=request.question[:100],  # First 100 chars for preview
        question_length=len(request.question)
    ):
        try:
            logfire.info(
                "Received question",
                session_id=request.session_id or "anonymous",
                question_length=len(request.question)
            )
            
            response = await execute_pipeline(request.question, request.session_id)
            
            logfire.info(
                "Question answered successfully",
                session_id=request.session_id or "anonymous",
                duration_ms=response.total_duration_ms,
                cost_usd=response.total_cost,
                quality_score=response.quality_score,
                iterations=response.iterations
            )
            
            return response
        
        except Exception as e:
            logfire.error(
                "Error processing question",
                session_id=request.session_id or "anonymous",
                error=str(e),
                exc_info=True
            )
            raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


async def pipeline_generator(question: str, session_id: str = None) -> AsyncGenerator[str, None]:
    """Generator for Server-Sent Events during pipeline execution."""
    
    try:
        async with pipeline_trace(question) as context:
            pipeline_start = time.time()  # Track total duration
            total_cost = 0.0
            progress = 0.0
            stages = []  # Track stages with costs
            
            # Stage 1: Embedding
            yield f"data: {json.dumps(ProgressUpdate(stage='question_embedding', status='started', message='Embedding your question...', progress=5, cost_so_far=total_cost).model_dump())}\n\n"
            
            stage_start = time.time()
            embed_result = await embed_question(question)
            stage_duration = (time.time() - stage_start) * 1000
            embed_cost = embed_result["cost_usd"]
            total_cost += embed_cost
            progress = 12.5
            stages.append(PipelineStageResult(
                stage="question_embedding",
                success=True,
                duration_ms=stage_duration,
                cost_usd=embed_cost,
                metadata={"embedding_dimensions": len(embed_result["embedding"])}
            ))
            
            yield f"data: {json.dumps(ProgressUpdate(stage='question_embedding', status='completed', message='Question embedded', progress=progress, cost_so_far=total_cost).model_dump())}\n\n"
            
            # Stage 2: Retrieval (with multi-query expansion for broad questions)
            yield f"data: {json.dumps(ProgressUpdate(stage='rag_retrieval', status='started', message='Searching documentation...', progress=15, cost_so_far=total_cost).model_dump())}\n\n"
            
            stage_start = time.time()
            retrieval_result = await retrieve_documents(embed_result["embedding"], original_question=question)
            stage_duration = (time.time() - stage_start) * 1000
            retrieval_cost = retrieval_result["cost_usd"]
            total_cost += retrieval_cost
            progress = 25
            documents = retrieval_result["documents"]
            stages.append(PipelineStageResult(
                stage="rag_retrieval",
                success=True,
                duration_ms=stage_duration,
                cost_usd=retrieval_cost,
                metadata={
                    "documents_retrieved": len(documents),
                    "queries_expanded": retrieval_result.get("queries_count", 1)
                }
            ))
            
            yield f"data: {json.dumps(ProgressUpdate(stage='rag_retrieval', status='completed', message=f'Found {len(documents)} relevant documents', progress=progress, cost_so_far=total_cost).model_dump())}\n\n"
            
            # Iterative loop
            current_iteration = 1
            feedback = None
            
            while current_iteration <= settings.max_iterations:
                iter_progress_start = 25 + ((current_iteration - 1) * 60 / settings.max_iterations)
                # Each iteration gets an equal share of the 60% progress (from 25% to 85%)
                iter_span = 60 / settings.max_iterations
                
                # Stage 3: Generation
                yield f"data: {json.dumps(ProgressUpdate(stage=f'generation_iter_{current_iteration}', status='started', message=f'Generating answer (iteration {current_iteration})...', progress=min(iter_progress_start + (0.05 * iter_span), 95), cost_so_far=total_cost).model_dump())}\n\n"
                
                stage_start = time.time()
                gen_result = await generate_answer(question, documents, feedback)
                stage_duration = (time.time() - stage_start) * 1000
                gen_cost = gen_result["cost_usd"]
                total_cost += gen_cost
                answer_text = gen_result["answer"]
                stages.append(PipelineStageResult(
                    stage=f"answer_generation_iter_{current_iteration}",
                    success=True,
                    duration_ms=stage_duration,
                    cost_usd=gen_cost,
                    metadata={
                        "answer_length": len(answer_text),
                        "iteration": current_iteration
                    }
                ))
                
                yield f"data: {json.dumps(ProgressUpdate(stage=f'generation_iter_{current_iteration}', status='completed', message='Answer generated', progress=min(iter_progress_start + (0.20 * iter_span), 95), cost_so_far=total_cost).model_dump())}\n\n"
                
                # Stage 4: Claims
                yield f"data: {json.dumps(ProgressUpdate(stage=f'claims_iter_{current_iteration}', status='started', message='Extracting factual claims...', progress=min(iter_progress_start + (0.30 * iter_span), 95), cost_so_far=total_cost).model_dump())}\n\n"
                
                stage_start = time.time()
                claims_result = await extract_claims(answer_text)
                stage_duration = (time.time() - stage_start) * 1000
                claims_cost = claims_result["cost_usd"]
                total_cost += claims_cost
                claims_count = len(claims_result["claims"])
                stages.append(PipelineStageResult(
                    stage=f"claims_extraction_iter_{current_iteration}",
                    success=True,
                    duration_ms=stage_duration,
                    cost_usd=claims_cost,
                    metadata={
                        "claims_extracted": claims_count,
                        "iteration": current_iteration
                    }
                ))
                
                yield f"data: {json.dumps(ProgressUpdate(stage=f'claims_iter_{current_iteration}', status='completed', message=f'Extracted {claims_count} claims', progress=min(iter_progress_start + (0.40 * iter_span), 95), cost_so_far=total_cost).model_dump())}\n\n"
                
                # Stage 5: Verification
                yield f"data: {json.dumps(ProgressUpdate(stage=f'verification_iter_{current_iteration}', status='started', message='Verifying claims...', progress=min(iter_progress_start + (0.50 * iter_span), 95), cost_so_far=total_cost).model_dump())}\n\n"
                
                stage_start = time.time()
                verification_result = await verify_claims(claims_result["claims"])
                stage_duration = (time.time() - stage_start) * 1000
                verification_cost = verification_result["cost_usd"]
                total_cost += verification_cost
                verified_claims = verification_result["verified_claims"]
                verification_rate = verification_result["verification_rate"] * 100
                verified_count = len([c for c in verified_claims if c.verified])
                stages.append(PipelineStageResult(
                    stage=f"claims_verification_iter_{current_iteration}",
                    success=True,
                    duration_ms=stage_duration,
                    cost_usd=verification_cost,
                    metadata={
                        "claims_verified": verified_count,
                        "total_claims": len(verified_claims),
                        "verification_rate": f"{verification_rate:.0f}%",
                        "iteration": current_iteration
                    }
                ))
                
                yield f"data: {json.dumps(ProgressUpdate(stage=f'verification_iter_{current_iteration}', status='completed', message=f'{verification_rate:.0f}% claims verified', progress=min(iter_progress_start + (0.60 * iter_span), 95), cost_so_far=total_cost).model_dump())}\n\n"
                
                # Stage 6: Accuracy
                yield f"data: {json.dumps(ProgressUpdate(stage=f'accuracy_iter_{current_iteration}', status='started', message='Checking technical accuracy...', progress=min(iter_progress_start + (0.70 * iter_span), 95), cost_so_far=total_cost).model_dump())}\n\n"
                
                stage_start = time.time()
                accuracy_result = await check_accuracy(answer_text, verified_claims)
                stage_duration = (time.time() - stage_start) * 1000
                accuracy_cost = accuracy_result["cost_usd"]
                total_cost += accuracy_cost
                accuracy_score = accuracy_result["accuracy_score"]
                stages.append(PipelineStageResult(
                    stage=f"accuracy_check_iter_{current_iteration}",
                    success=True,
                    duration_ms=stage_duration,
                    cost_usd=accuracy_cost,
                    metadata={
                        "accuracy_score": accuracy_score,
                        "iteration": current_iteration
                    }
                ))
                
                yield f"data: {json.dumps(ProgressUpdate(stage=f'accuracy_iter_{current_iteration}', status='completed', message=f'Accuracy score: {accuracy_score}/100', progress=min(iter_progress_start + (0.80 * iter_span), 95), cost_so_far=total_cost).model_dump())}\n\n"
                
                # Stage 7: Evaluation
                yield f"data: {json.dumps(ProgressUpdate(stage=f'evaluation_iter_{current_iteration}', status='started', message='Evaluating quality...', progress=min(iter_progress_start + (0.85 * iter_span), 95), cost_so_far=total_cost).model_dump())}\n\n"
                
                stage_start = time.time()
                eval_result = await evaluate_quality(question, answer_text, documents)
                stage_duration = (time.time() - stage_start) * 1000
                eval_cost = eval_result["cost_usd"]
                total_cost += eval_cost
                evaluations = eval_result["evaluations"]
                average_score = eval_result["average_score"]
                stages.append(PipelineStageResult(
                    stage=f"quality_evaluation_iter_{current_iteration}",
                    success=True,
                    duration_ms=stage_duration,
                    cost_usd=eval_cost,
                    metadata={
                        "quality_score": f"{average_score:.1f}",
                        "openai_score": evaluations[0].score if len(evaluations) > 0 else None,
                        "claude_score": evaluations[1].score if len(evaluations) > 1 else None,
                        "agreement": eval_result.get("agreement_level", "unknown"),
                        "iteration": current_iteration
                    }
                ))
                
                yield f"data: {json.dumps(ProgressUpdate(stage=f'evaluation_iter_{current_iteration}', status='completed', message=f'Quality score: {average_score:.1f}/100', progress=min(iter_progress_start + (0.90 * iter_span), 95), cost_so_far=total_cost).model_dump())}\n\n"
                
                # Stage 8: Quality Gate
                yield f"data: {json.dumps(ProgressUpdate(stage=f'quality_gate_iter_{current_iteration}', status='started', message='Checking quality gate...', progress=min(iter_progress_start + (0.95 * iter_span), 95), cost_so_far=total_cost).model_dump())}\n\n"
                
                gate_result = await quality_gate_decision(
                    average_score=average_score,
                    evaluations=evaluations,
                    accuracy_score=accuracy_score,
                    current_iteration=current_iteration,
                    errors=accuracy_result["errors"],
                    corrections=accuracy_result["corrections"]
                )
                
                if not gate_result["should_iterate"]:
                    yield f"data: {json.dumps(ProgressUpdate(stage=f'quality_gate_iter_{current_iteration}', status='completed', message='Quality gate passed!', progress=95, cost_so_far=total_cost).model_dump())}\n\n"
                    
                    # Calculate total duration
                    total_duration_ms = (time.time() - pipeline_start) * 1000
                    
                    # Send final result
                    response = AnswerResponse(
                        question=question,
                        answer=answer_text,
                        sources=documents,
                        claims=verified_claims,
                        quality_score=average_score,
                        accuracy_score=accuracy_score,
                        evaluations=evaluations,
                        iterations=current_iteration,
                        total_cost=total_cost,
                        total_duration_ms=total_duration_ms,
                        stages=stages,
                        session_id=session_id
                    )
                    
                    # Save session to database
                    try:
                        # Capture current trace ID
                        current_span = trace.get_current_span()
                        trace_id = None
                        if current_span and current_span.get_span_context().is_valid:
                            trace_id = format(current_span.get_span_context().trace_id, '032x')
                        
                        saved_session_id = await vector_store.save_session(
                            question=question,
                            answer=answer_text,
                            sources=[doc.model_dump() for doc in documents],
                            claims=[claim.model_dump() for claim in verified_claims],
                            evaluations=[ev.model_dump() for ev in evaluations],
                            quality_score=average_score,
                            iterations=current_iteration,
                            total_cost=total_cost,
                            total_duration_ms=total_duration_ms,
                            trace_id=trace_id,
                            stages=[s.model_dump() for s in stages]
                        )
                        logfire.info(f"Saved session to database: {saved_session_id}", trace_id=trace_id)
                        
                        # Update response with saved session ID
                        response.session_id = saved_session_id
                    except Exception as e:
                        logfire.error(f"Failed to save session: {e}")
                    
                    yield f"data: {json.dumps({'type': 'complete', 'result': response.model_dump(mode='json')})}\n\n"
                    break
                
                gate_reason = gate_result["reason"]
                yield f"data: {json.dumps(ProgressUpdate(stage=f'quality_gate_iter_{current_iteration}', status='completed', message=f'Refining answer... ({gate_reason})', progress=min(iter_progress_start + iter_span, 85), cost_so_far=total_cost).model_dump())}\n\n"
                
                feedback = gate_result["feedback"]
                current_iteration += 1
    
    except Exception as e:
        logfire.error(f"Error in pipeline generator: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


@app.post("/ask/stream", tags=["Q&A"])
async def ask_question_stream(request: QuestionRequest):
    """
    Ask a question with real-time progress updates via Server-Sent Events.
    
    This endpoint provides the same functionality as /ask but streams
    progress updates as the pipeline executes.
    """
    
    logfire.info(f"Received streaming question: {request.question}")
    
    return StreamingResponse(
        pipeline_generator(request.question, request.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/stats", tags=["Admin"])
async def get_stats():
    """Get database statistics."""
    
    doc_count = await vector_store.get_document_count()
    
    return {
        "document_count": doc_count,
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "rag_top_k": settings.rag_top_k,
        "quality_threshold": settings.quality_threshold,
        "max_iterations": settings.max_iterations
    }


@app.get("/history", tags=["Q&A"])
async def get_history(limit: int = 20):
    """
    Get recent Q&A sessions.
    
    Args:
        limit: Maximum number of sessions to return (default: 20, max: 100)
    """
    
    if limit > 100:
        raise HTTPException(status_code=400, detail="Limit cannot exceed 100")
    
    sessions = await vector_store.get_recent_sessions(limit=limit)
    
    return {
        "sessions": sessions,
        "count": len(sessions)
    }


@app.get("/history/{session_id}", tags=["Q&A"])
async def get_session(session_id: str):
    """
    Get a specific Q&A session by ID.
    
    Args:
        session_id: The UUID of the session
    """
    
    session = await vector_store.get_session_by_id(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session


@app.delete("/history/{session_id}", tags=["Q&A"])
async def delete_session(session_id: str):
    """
    Delete a specific Q&A session by ID.
    
    Args:
        session_id: The UUID of the session to delete
    """
    
    deleted = await vector_store.delete_session(session_id)
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "success": True,
        "message": "Session deleted successfully",
        "session_id": session_id
    }


@app.delete("/history", tags=["Q&A"])
async def clear_all_history():
    """
    Delete all Q&A sessions from history.
    
    This action cannot be undone.
    """
    
    deleted_count = await vector_store.delete_all_sessions()
    
    return {
        "success": True,
        "message": f"Deleted {deleted_count} sessions",
        "count": deleted_count
    }


@app.get("/sessions/{session_id}/logs", tags=["Observability"])
async def get_session_logs(session_id: str):
    """
    Fetch Logfire logs for a specific Q&A session.
    
    Returns detailed observability logs from Logfire for the given session,
    including all spans, traces, and metrics captured during execution.
    """
    # Get session from database to retrieve trace_id
    session = await vector_store.get_session_by_id(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    trace_id = session.get('trace_id')
    if not trace_id:
        raise HTTPException(
            status_code=404, 
            detail="Trace ID not available for this session (may be from before trace logging was enabled)"
        )
    
    # Fetch logs from Logfire API
    try:
        logs_data = await fetch_logfire_logs(trace_id)
        return logs_data
    except HTTPException:
        raise
    except Exception as e:
        logfire.error(f"Unexpected error fetching logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch logs: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower()
    )

