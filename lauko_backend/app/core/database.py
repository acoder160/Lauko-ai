import logging
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import event

logger = logging.getLogger("lauko-Database")

# Path to our local SQLite database file
# We use aiosqlite driver for non-blocking async operations
SQLALCHEMY_DATABASE_URL = "sqlite+aiosqlite:///./lauko_data.db"

# Create the asynchronous engine
engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False}
)

# Enable WAL (Write-Ahead Logging) mode for extreme SQLite performance
@event.listens_for(engine.sync_engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()

# Create an async session factory
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy ORM models.
    """
    pass

async def get_db():
    """
    Dependency generator for FastAPI endpoints.
    Ensures safe creation and closure of database sessions.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()