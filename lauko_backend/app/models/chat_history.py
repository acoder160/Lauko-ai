import uuid
from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.sql import func
from app.core.database import Base

def generate_uuid():
    """Generates a unique string ID for new records."""
    return str(uuid.uuid4())

class UserProfile(Base):
    """
    SQLAlchemy model for storing long-term user state (Level 2 & 3 Memory).
    """
    __tablename__ = "user_profiles"

    # We use the same user_id from the client
    user_id = Column(String, primary_key=True, index=True)
    
    # LEVEL 2 MEMORY: A running summary of older, archived conversations
    conversation_summary = Column(Text, default="", nullable=False)
    
    # LEVEL 3 MEMORY: A JSON-formatted string containing core facts (Dossier)
    dossier = Column(Text, default="{}", nullable=False)
    

class Message(Base):
    """
    SQLAlchemy model for storing short-term conversation history (Level 1 Memory).
    """
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    user_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)