from sqlalchemy.orm import Session

from research_agent.infrastructure.db.session import SessionLocal


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
