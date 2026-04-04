# Research Paper Replication Agent

Production-oriented scaffold for a system that ingests ML papers, analyzes reproducibility, supports semantic search and Q&A, and generates experiment code bundles.

## Structure

- `apps/api`: FastAPI entrypoint and HTTP routes
- `apps/worker`: background worker and LangGraph execution entrypoint
- `packages/research_agent`: core Python package
- `frontend`: minimal Next.js frontend scaffold
- `migrations`: Alembic configuration and migration versions

## Quick Start

```bash
uvicorn apps.api.main:app --reload
```
