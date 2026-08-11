"""
SQLAlchemy Database Connection & Session Factory for ECIP.
Supports PostgreSQL with local SQLite fallback during development.
"""

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from config.settings import Settings
from utils.logger import setup_logger

logger = setup_logger("ECIP.Database")

settings = Settings()
db_dir = settings.get_path("paths.output_dir")
db_dir.mkdir(parents=True, exist_ok=True)
sqlite_path = db_dir / "ecip.db"

# Database Connection URL (SQLite fallback)
SQLALCHEMY_DATABASE_URL = f"sqlite:///{sqlite_path}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency injection helper for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
