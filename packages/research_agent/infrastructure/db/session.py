import os
import socket
from urllib.parse import urlparse

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/research_agent"
)
DATABASE_URL_IPV4 = os.getenv("DATABASE_URL_IPV4")


def _host_has_ipv4(url: str) -> bool:
    hostname = urlparse(url).hostname
    if not hostname:
        return True
    try:
        addresses = socket.getaddrinfo(hostname, None, socket.AF_INET)
    except socket.gaierror:
        return False
    return len(addresses) > 0


if DATABASE_URL_IPV4 and not _host_has_ipv4(DATABASE_URL):
    DATABASE_URL = DATABASE_URL_IPV4

engine = create_engine(
    DATABASE_URL,
    future=True,
    pool_pre_ping=True,
    pool_recycle=300,
    pool_timeout=30,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()
