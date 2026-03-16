# Render Developer Q&A Assistant

> A demo AI pipeline showcasing observable AI with LangChain, Logfire, and Render

An intelligent question-answering system that demonstrates real-world AI observability patterns. This example project shows how to build, instrument, and monitor a multi-stage LLM pipeline with full cost tracking, quality evaluation, and performance monitoring.

## Table of Contents

- [What This App Does](#what-this-app-does)
- [What This Demonstrates](#what-this-demonstrates)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Deploy to Render](#deploy-to-render)
- [Example Metrics](#example-metrics)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## What This App Does

This is an **AI-powered Q&A assistant for Render documentation**. Users can ask questions about Render's platform, and the app provides accurate, well-researched answers backed by the official documentation.

### User Experience

1. **Ask a question** - "How do I deploy a Node.js app on Render?" or "What database plans are available?"
2. **Watch the pipeline** - See real-time progress through 8 stages (embedding → retrieval → generation → verification)
3. **Get accurate answers** - Receive detailed responses with sources from Render docs
4. **Quality guaranteed** - Every answer is verified for accuracy and rated by dual AI evaluators

### Key Features

- **Hybrid search** - Combines semantic understanding with keyword matching for better retrieval
- **Multi-stage verification** - Extracts claims, verifies against docs, checks technical accuracy
- **Iterative refinement** - Automatically regenerates low-quality answers with feedback
- **Cost tracking** - See exactly how much each question costs to answer
- **Real-time streaming** - Progressive response updates via Server-Sent Events

### Example Questions

```
"How do I set up PostgreSQL on Render?"
"What's the difference between Web Services and Static Sites?"
"How much does a Starter plan cost?"
"Can I use custom domains with Render?"
"How do I configure environment variables?"
```

The app answers questions about deployment, databases, pricing, configuration, networking, and all other Render platform features based on ~10,000 documentation chunks.

---

## What This Demonstrates

### Logfire Features

- **LLM Traces** - Complete visibility into every AI call (OpenAI + Anthropic auto-instrumented)
- **HTTP Tracing** - FastAPI auto-instrumentation for request/response tracking
- **Database Monitoring** - AsyncPG auto-instrumentation for query performance
- **Cost Tracking** - Per-stage and per-execution cost attribution with custom metrics
- **Multi-Model Evals** - Dual-rater quality assessment (OpenAI + Anthropic)
- **Session Tracking** - End-to-end user journey with distributed tracing
- **Custom Metrics** - Business-specific metrics (cost, quality, iterations)
- **SQL Queries** - Custom analytics on AI performance

### LangChain Patterns

- **Hybrid RAG Pipeline** - Semantic + lexical search with RRF ranking
- **Chain Composition** - Sequential and parallel chain execution
- **Multi-Model Orchestration** - Claude + GPT-4 in a single pipeline
- **Retry Logic** - Quality gates with iterative refinement
- **Error Handling** - Graceful degradation and fallbacks

### Render Capabilities

- **Zero-Config Deployment** - Push to deploy with render.yaml
- **PostgreSQL with pgvector + full-text** - Managed hybrid search database
- **Web Service + Static Site** - Full-stack deployment
- **Environment Management** - Secure secrets handling
- **Auto-Scaling** - Handle variable AI workloads

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Frontend (React + TypeScript)                              │
│  Deployed as: Render Static Site                            │
│  - Question input UI                                        │
│  - Real-time progress via SSE                               │
│  - Answer display with metrics                              │
└─────────────────────────────────────────────────────────────┘
                          ↓ HTTPS
┌─────────────────────────────────────────────────────────────┐
│  Backend API (FastAPI + LangChain + Logfire)                │
│  Deployed as: Render Web Service (Python 3.11+)             │
│                                                             │
│  8-Stage Pipeline:                                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ [1] Question Embedding      (OpenAI)                   │ │
│  │ [2] RAG Document Retrieval  (pgvector + BM25)          │ │
│  │ [3] Answer Generation       (Claude Sonnet 4.5)        │ │
│  │ [4] Claims Extraction       (GPT-4o-mini)              │ │
│  │ [5] Claims Verification     (RAG again)                │ │
│  │ [6] Technical Accuracy      (Claude Sonnet 4)          │ │
│  │ [7] Quality Rating          (OpenAI + Anthropic)       │ │
│  │ [8] Quality Gate            (Pass or Iterate)          │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
            ↓                                    ↓
┌──────────────────────┐           ┌───────────────────────────┐
│  PostgreSQL          │           │  Logfire                  │
│  (Render Managed)    │           │  (Pydantic)               │
│  - pgvector ext      │           │  - Distributed traces     │
│  - RAG embeddings    │           │  - Cost attribution       │
│  - Full-text search  │           │  - Quality metrics        │
└──────────────────────┘           │  - Custom dashboards      │
                                   └───────────────────────────┘
```

### Project Structure

```
render-qa-assistant/
├── backend/
│   ├── main.py                    # FastAPI application entry
│   ├── requirements.txt           # Python dependencies
│   ├── api/
│   │   └── logs.py                # Logfire logs API endpoint
│   ├── pipeline/                  # 8-stage pipeline implementation
│   ├── models.py                  # Pydantic models
│   ├── database.py                # PostgreSQL + pgvector
│   ├── observability.py           # Logfire configuration
│   └── config.py                  # Settings management
├── frontend/
│   ├── src/                       # React + TypeScript UI
│   ├── package.json
│   └── vite.config.ts
├── data/
│   ├── embeddings/                # Pre-embedded documentation
│   └── scripts/                   # Data ingestion scripts
├── docs/
│   ├── PIPELINE.md                # Detailed pipeline guide
│   ├── OBSERVABILITY.md           # Logfire instrumentation guide
│   ├── CONFIGURATION.md           # Configuration reference
│   └── HYBRID_SEARCH.md           # Hybrid search deep-dive
├── render.yaml                    # Infrastructure as code
├── .env.example                   # Environment variables template
└── README.md                      # This file
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 16+ (with pgvector extension)
- OpenAI API key
- Anthropic API key
- Logfire account (free tier available)

### Local Development (with Make)

```bash
# 1. Install everything
make install

# 2. Set up .env file
make dev-setup
# Edit .env and add your API keys!

# 3. Start database
make db-start

# 4. Load documentation
make ingest

# 5. Run backend (in one terminal)
make run-backend

# 6. Run frontend (in another terminal)
make run-frontend
```

### Manual Setup

```bash
# 1. Install Python dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

# 2. Configure environment
cp .env.local .env
# Edit .env with your API keys

# 3. Start PostgreSQL with Docker
docker-compose up -d

# 4. Generate and load documentation
python data/scripts/generate_embeddings.py
python data/scripts/ingest_docs.py

# 5. Run backend (from project root)
uvicorn backend.main:app --reload --port 8000

# 6. Run frontend (separate terminal)
cd frontend && npm install && npm run dev
```

**Access locally:**

- Frontend: http://localhost:3000
- API docs: http://localhost:8000/docs
- Logfire: https://logfire.pydantic.dev

---

## Deploy to Render

1. **Fork this repository**

2. **Create Render account** at https://render.com

3. **In Render Dashboard:**

   - Click "New +" → "Blueprint"
   - Connect your forked repository
   - Render reads `render.yaml` and creates:
     - PostgreSQL database with pgvector
     - Web Service (backend API)
     - Static Site (frontend)

4. **Set environment variables:**

   - `OPENAI_API_KEY` (required)
   - `ANTHROPIC_API_KEY` (required)
   - `LOGFIRE_TOKEN` (required)

5. **Deploy completes in ~5 minutes**

   - Backend: `https://your-service.onrender.com`
   - Frontend: `https://your-app.onrender.com`

6. **Initialize the database:**

```bash
# Run the ingestion script on your deployed service
curl -X POST https://your-service.onrender.com/admin/ingest-docs
```

---

## Example Metrics

### Cost Breakdown (per question)

```
┌────────────────────────────────┬──────────┬──────────┐
│ Stage                          │ Cost     │ % Total  │
├────────────────────────────────┼──────────┼──────────┤
│ Question Embedding             │ $0.0002  │    2%    │
│ RAG Retrieval                  │ $0.0001  │    1%    │
│ Answer Generation (Claude)     │ $0.0450  │   56%    │
│ Claims Extraction (GPT)        │ $0.0080  │   10%    │
│ Claims Verification (RAG)      │ $0.0015  │    2%    │
│ Accuracy Check (Claude)        │ $0.0180  │   22%    │
│ Quality Rating (Dual)          │ $0.0070  │    9%    │
├────────────────────────────────┼──────────┼──────────┤
│ TOTAL (first iteration)        │ $0.0798  │  100%    │
└────────────────────────────────┴──────────┴──────────┘
```

### Performance Metrics

- **Average Response Time:** 4.2 seconds (first iteration)
- **P95 Response Time:** 8.7 seconds
- **Iteration Rate:** 12% of questions require refinement
- **Success Rate:** 95% accuracy (validated by dual evaluators)

### Quality Scores

- **Average Quality Score:** 89/100
- **OpenAI Average:** 87/100
- **Anthropic Average:** 91/100
- **Inter-rater Agreement:** 77% (within 10 points)

---

## Documentation

### Core Guides

- **[docs/PIPELINE.md](./docs/PIPELINE.md)** - Detailed breakdown of the 8-stage pipeline
- **[docs/OBSERVABILITY.md](./docs/OBSERVABILITY.md)** - Comprehensive Logfire instrumentation guide
- **[docs/CONFIGURATION.md](./docs/CONFIGURATION.md)** - All configuration options and tuning
- **[docs/HYBRID_SEARCH.md](./docs/HYBRID_SEARCH.md)** - Technical deep-dive on hybrid search

### External Resources

- **Logfire Documentation:** https://docs.pydantic.dev/logfire/
- **LangChain Documentation:** https://python.langchain.com/docs/
- **Render Documentation:** https://docs.render.com/

---

## Contributing

This is a demo project, but improvements are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## License

MIT License - see LICENSE file for details

---

## Acknowledgments

Built to showcase:

- **Logfire** by Pydantic - AI observability platform
- **Render** - Modern cloud platform
- **LangChain** - AI orchestration framework
- **OpenAI & Anthropic** - LLM providers

---

**Ready to build observable AI?** Fork this repo and deploy to Render to get started!
