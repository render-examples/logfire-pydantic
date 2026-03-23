# pydantic-ai Migration Summary

## Overview

Replaced direct OpenAI and Anthropic SDK calls with [pydantic-ai](https://github.com/pydantic/pydantic-ai) across the LLM pipeline stages. Also removed 4 unused LangChain packages that were listed in `requirements.txt` but never imported anywhere in the codebase.

---

## Files Changed

### `backend/requirements.txt`
- Removed 4 dead LangChain packages: `langchain`, `langchain-openai`, `langchain-anthropic`, `langchain-postgres`
- Added `pydantic-ai>=0.0.14`
- Kept as a reference; superseded by `pyproject.toml` (see UV migration below)

### `backend/config.py`
- Updated model defaults to latest versions
- Added GPT-4o cost constants (`GPT4O_INPUT_COST_PER_M`, `GPT4O_OUTPUT_COST_PER_M`)
- Added Claude Sonnet 4.6 cost constants (`CLAUDE_SONNET_46_INPUT_COST_PER_M`, `CLAUDE_SONNET_46_OUTPUT_COST_PER_M`)
- Added `query_expansion_model` setting (previously hardcoded in `query_expansion.py`)

### `backend/observability.py`
- Replaced `logfire.instrument_anthropic()` with `logfire.instrument_pydantic_ai()`
- Kept `logfire.instrument_openai()` for `embeddings.py` and `verification.py` which still use the raw OpenAI SDK for embeddings
- Updated `calculate_openai_cost()` to handle `gpt-5.4-mini` pricing
- Updated `calculate_anthropic_cost()` to handle `claude-sonnet-4-6` pricing

### `backend/models.py`
Added 4 intermediate structured output models used as `output_type` in pydantic-ai agents:
- `QueryExpansionOutput` — list of query variations
- `ClaimsOutput` — list of extracted factual claims
- `AccuracyOutput` — accuracy score, errors, corrections
- `EvaluationOutput` — per-criterion scores and feedback

### `backend/pipeline/query_expansion.py`
- Replaced `AsyncOpenAI` + manual JSON/markdown parsing with a pydantic-ai `Agent`
- Eliminated ~25 lines of `json.loads()`, markdown code block stripping, and list-validation fallback logic

### `backend/pipeline/generation.py`
- Replaced `AsyncAnthropic` with a pydantic-ai `Agent`
- Moved static system instructions to the `instructions=` parameter

### `backend/pipeline/claims.py`
- Replaced `AsyncOpenAI` + ~80 lines of fragile JSON parsing with a pydantic-ai `Agent` using `output_type=ClaimsOutput`
- pydantic-ai handles `response_format=json_object` automatically when `output_type` is specified
- Replaced the `finish_reason == "length"` truncation check with a token-count proximity check (`output_tokens >= 3900`)

### `backend/pipeline/accuracy.py`
- Replaced `AsyncAnthropic` + ~30 lines of line-by-line text parsing with a pydantic-ai `Agent` using `output_type=AccuracyOutput`
- Prompt updated to elicit structured JSON instead of labelled text blocks (`ACCURACY_SCORE: [0-100]` etc.)

### `backend/pipeline/evaluation.py`
- Replaced two direct SDK clients + `parse_evaluation()` + `extract_score()` (~50 lines) with two pydantic-ai `Agent`s using `output_type=EvaluationOutput`
- `asyncio.gather()` parallelism between OpenAI and Anthropic evaluators is preserved

### `env.example`
- Updated model names to match new defaults
- Added `QUERY_EXPANSION_MODEL` entry

---

## UV Migration

Migrated from pip + venv to [uv](https://docs.astral.sh/uv/) to fix Python 3.14 install failures (`pydantic-core` and `tiktoken` have no pre-built wheels for 3.14 and their Rust build toolchain does not support it yet).

### Files Changed

- **`.python-version`** (new) — pins Python to `3.13`
- **`pyproject.toml`** (new) — all deps migrated from `backend/requirements.txt`; `logfire` extra `openai` removed (dropped in 4.29.0, instrumentation still works via `httpx`)
- **`uv.lock`** (new) — fully-resolved lockfile (132 packages)
- **`Makefile`** — replaced `python3 -m venv` / `pip install` with `uv sync`; all `venv/bin/activate &&` prefixes replaced with `uv run`; `clean` target removes `.venv` instead of `venv`
- **`.gitignore`** — added `.venv/`
- **`README.md`** — updated prereqs (uv instead of Python 3.11+), manual setup commands, project structure diagram

### Key Commands After Migration

```bash
make install          # uv sync --group dev (creates .venv with Python 3.13)
make run-backend      # uv run uvicorn backend.main:app --reload --port 8000
make test             # uv run pytest backend/tests/ -v
uv add <package>      # add a new dependency
uv sync               # install from lockfile after pulling changes
```

---

## Model Upgrades

| Stage | Before | After |
|-------|--------|-------|
| Answer generation | `claude-sonnet-4-5-20250929` | `claude-sonnet-4-6` |
| Accuracy check | `claude-sonnet-4-20250514` | `claude-sonnet-4-6` |
| Evaluation (Anthropic) | `claude-sonnet-4-20250514` | `claude-sonnet-4-6` |
| Evaluation (OpenAI) | `gpt-4o-mini` | `gpt-5.4-mini` |
| Claims extraction | `gpt-4o-mini` | `gpt-5.4-mini` |
| Query expansion | `gpt-4o-mini` (hardcoded) | configurable via `QUERY_EXPANSION_MODEL` env var |

---

### `data/scripts/generate_embeddings.py`
- Added exponential backoff retry (up to 5 attempts: 1s, 2s, 4s, 8s) on `RateLimitError` in `generate_embedding()`
- Increased per-request delay from 0.2s → 0.65s to stay under the 100 RPM limit for `text-embedding-3-small`

---

## Files NOT Changed

| File | Reason |
|------|--------|
| `backend/pipeline/embeddings.py` | Uses OpenAI Embeddings API — not covered by pydantic-ai |
| `backend/pipeline/verification.py` | Uses OpenAI Embeddings API for claim re-search — not covered by pydantic-ai |
| `backend/pipeline/retrieval.py` | Pure database queries, no LLM calls |
| `backend/pipeline/quality_gate.py` | Pure decision logic, no LLM calls |
| `backend/database.py` | Database layer |
| `backend/main.py` | Application entry point |

---

## Key Benefits

- **Eliminated ~185 lines** of fragile manual JSON/text parsing across `claims.py`, `accuracy.py`, and `evaluation.py`
- **Type-safe structured outputs** — pydantic-ai validates LLM responses against Pydantic models automatically, with built-in retry on validation failure
- **Unified observability** — `logfire.instrument_pydantic_ai()` captures all agent runs with token usage, latency, and errors in a single instrumentation call
- **Cleaner prompts** — System instructions are separated from user prompts via the `instructions=` parameter
