import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

from apps.api.routers import analysis, experiments, ingestion, papers, qa, reproducibility, search
from research_agent.domain import models  # noqa: F401
from research_agent.infrastructure.db.session import Base, engine


load_dotenv()
logging.basicConfig(level=logging.INFO)

STORAGE_DIR = Path("storage/papers")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Research Paper Replication Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(search.router, tags=["papers"])
app.include_router(qa.router, tags=["papers"])
app.include_router(analysis.router, tags=["papers"])
app.include_router(ingestion.router, tags=["papers"])
app.include_router(papers.router, prefix="/papers", tags=["papers"])
app.include_router(reproducibility.router, prefix="/reproducibility", tags=["reproducibility"])
app.include_router(experiments.router, prefix="/experiments", tags=["experiments"])
