# Makefile for Render Q&A Assistant
# Simplifies common development tasks

.PHONY: help install dev-setup db-start db-stop db-reset ingest run-backend run-frontend test clean

help:
	@echo "Render Q&A Assistant - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install       - Install all dependencies (Python + Node)"
	@echo "  make dev-setup     - Complete development setup"
	@echo ""
	@echo "Database:"
	@echo "  make db-start      - Start PostgreSQL with Docker"
	@echo "  make db-stop       - Stop PostgreSQL"
	@echo "  make db-reset      - Reset database (delete all data)"
	@echo "  make ingest        - Generate embeddings and load into database"
	@echo ""
	@echo "Development:"
	@echo "  make run-backend   - Run backend API (port 8000)"
	@echo "  make run-frontend  - Run frontend dev server (port 3000)"
	@echo "  make test          - Run tests"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         - Clean up build artifacts"

install:
	@echo "📦 Installing Python dependencies..."
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	@echo ""
	@echo "📦 Installing Node.js dependencies..."
	cd frontend && npm install
	@echo ""
	@echo "✅ All dependencies installed!"

dev-setup: install
	@echo ""
	@echo "⚙️  Setting up development environment..."
	@if [ ! -f .env ]; then \
		cp .env.local .env; \
		echo "📝 Created .env file from .env.local"; \
		echo "⚠️  Please edit .env and add your API keys!"; \
	else \
		echo "✅ .env file already exists"; \
	fi
	@echo ""
	@echo "🚀 Next steps:"
	@echo "  1. Edit .env and add your API keys"
	@echo "  2. Run: make db-start"
	@echo "  3. Run: make ingest"
	@echo "  4. Run: make run-backend (in one terminal)"
	@echo "  5. Run: make run-frontend (in another terminal)"

db-start:
	@echo "🐘 Starting PostgreSQL with pgvector..."
	docker-compose up -d
	@echo "⏳ Waiting for database to be ready..."
	@sleep 5
	@docker-compose ps
	@echo "✅ Database is running on localhost:5432"

db-stop:
	@echo "🛑 Stopping PostgreSQL..."
	docker-compose down
	@echo "✅ Database stopped"

db-reset:
	@echo "⚠️  This will delete ALL data in the database!"
	@read -p "Are you sure? (y/N): " confirm; \
	if [ "$$confirm" = "y" ]; then \
		docker-compose down -v; \
		docker-compose up -d; \
		echo "✅ Database reset complete"; \
	else \
		echo "❌ Cancelled"; \
	fi

ingest:
	@echo "🔄 Generating embeddings for Render documentation..."
	. venv/bin/activate && python data/scripts/generate_embeddings.py
	@echo ""
	@echo "📊 Loading embeddings into database..."
	. venv/bin/activate && python data/scripts/ingest_docs.py
	@echo "✅ Documentation ingested!"

add-pricing:
	@echo "🏷️  Adding Render pricing page to vector database..."
	@echo "This adds accurate pricing tables for all Render services"
	@echo ""
	. venv/bin/activate && python data/scripts/add_pricing_page.py
	@echo "✅ Pricing data added!"

run-backend:
	@echo "🚀 Starting backend API on http://localhost:8000"
	@echo "📖 API docs: http://localhost:8000/docs"
	@echo ""
	. venv/bin/activate && uvicorn backend.main:app --reload --port 8000

run-frontend:
	@echo "🎨 Starting frontend on http://localhost:3000"
	@echo ""
	cd frontend && npm run dev

test:
	@echo "🧪 Running tests..."
	. venv/bin/activate && cd backend && pytest tests/ -v

clean:
	@echo "🧹 Cleaning up..."
	rm -rf venv
	rm -rf frontend/node_modules
	rm -rf frontend/.next
	rm -rf frontend/out
	rm -rf backend/__pycache__
	rm -rf backend/**/__pycache__
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup complete"

