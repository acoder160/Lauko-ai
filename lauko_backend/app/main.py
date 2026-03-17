from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router as chat_router
from app.core.database import engine, Base
# Import models so SQLAlchemy registers them before table creation
from app.models.chat_history import Message

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events: Code executed before the server starts accepting requests.
    We use this to initialize our database tables asynchronously.
    """
    # Create all tables defined in our models
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield  # Server is running and accepting requests
    
    # Cleanup code goes here when the server stops
    await engine.dispose()

# Initialize the FastAPI application with the lifespan manager
app = FastAPI(
    title="lauko Backend API",
    description="High-performance backend for the lauko proactive AI companion.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS (Cross-Origin Resource Sharing)
# Essential for allowing React Native / Web clients to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register the API routers
# Prefixing with /api/v1 helps with future API versioning
app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])

@app.get("/")
async def health_check():
    """
    Simple health check endpoint for deployment monitoring (e.g., Docker, Azure).
    """
    return {"status": "healthy", "service": "lauko API"}