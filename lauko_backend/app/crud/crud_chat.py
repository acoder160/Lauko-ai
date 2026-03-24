from typing import List, Sequence, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete, func, text

from app.models.chat_history import Message, UserProfile

async def get_user_profile(db: AsyncSession, user_id: str) -> Optional[UserProfile]:
    """Fetches the user profile from the database."""
    stmt = select(UserProfile).where(UserProfile.user_id == user_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()

async def get_or_create_user_profile(db: AsyncSession, user_id: str) -> UserProfile:
    """Fetches the user profile or creates a new one if it doesn't exist."""
    profile = await get_user_profile(db, user_id)
    if not profile:
        profile = UserProfile(user_id=user_id, dossier="{}")
        db.add(profile)
        await db.commit()
        await db.refresh(profile)
    return profile

async def add_message(db: AsyncSession, user_id: str, role: str, content: str) -> Message:
    """Saves a new chat message to the database."""
    new_msg = Message(user_id=user_id, role=role, content=content)
    db.add(new_msg)
    await db.commit()
    return new_msg

async def get_recent_messages(db: AsyncSession, user_id: str, limit: int = 20) -> Sequence[Message]:
    """Fetches recent messages for context window, returned chronologically."""
    stmt = select(Message).where(Message.user_id == user_id).order_by(Message.created_at.desc()).limit(limit)
    result = await db.execute(stmt)
    # [::-1] reverses the list so the LLM reads top-to-bottom
    return result.scalars().all()[::-1]

async def get_total_message_count(db: AsyncSession, user_id: str) -> int:
    """Counts the total number of messages for a user."""
    stmt = select(func.count(Message.id)).where(Message.user_id == user_id)
    result = await db.execute(stmt)
    return result.scalar() or 0

async def get_oldest_messages(db: AsyncSession, user_id: str, limit: int = 10) -> Sequence[Message]:
    """Fetches the oldest messages for summarization."""
    stmt = select(Message).where(Message.user_id == user_id).order_by(Message.created_at.asc()).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()

async def delete_messages_by_ids(db: AsyncSession, msg_ids: List[int]) -> None:
    """Deletes old summarized messages to save space."""
    if not msg_ids:
        return
    stmt = delete(Message).where(Message.id.in_(msg_ids))
    await db.execute(stmt)
    await db.commit()

async def schedule_task(db: AsyncSession, user_id: str, scheduled_at: str, context: str, task_type: str = "reminder"):
    """Inserts a new scheduled task directly using raw SQL."""
    insert_query = text("""
        INSERT INTO scheduled_tasks (user_id, scheduled_at, message_context, task_type)
        VALUES (:user_id, :scheduled_at, :message_context, :task_type)
    """)
    await db.execute(insert_query, {
        "user_id": user_id,
        "scheduled_at": scheduled_at,
        "message_context": context,
        "task_type": task_type
    })
    await db.commit()