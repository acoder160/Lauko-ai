import json
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete, func, text  # Added 'text' for raw SQL execution
import fitz  # PyMuPDF

from app.schemas.chat import ChatRequest, ChatResponse
from app.core.llm_manager import llm_manager
from app.core.database import get_db
from app.models.chat_history import Message, UserProfile

router = APIRouter()

# --- Dynamic Prompt Generator ---
def build_system_prompt(dossier_content: str, location: str = "Unknown Location") -> str:
    """
    Builds the foundational persona and awareness for Lauko.
    Injects real-time data so the LLM is aware of the current date and location.
    """
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    current_year = now.year
    
    system_prompt = f"""
You are Lauko, a proactive, intelligent, and highly capable AI companion.

[REAL-WORLD CONTEXT]
- Current Date and Time: {current_time}
- Current Year: {current_year}
- User Location: {location}
CRITICAL INSTRUCTION: Always use this real-world context for planning, answering questions about time, or organizing the user's schedule. Do not rely on your training data for the current date or year.

[USER DOSSIER (Level 3 Memory)]
Here is everything you know about this specific user:
{dossier_content}

[YOUR PERSONA & GUIDELINES]
- You are concise, sharp, and helpful. Do not be overly verbose.
- Never use robotic phrases like "As an AI language model..." or "I don't have real-time access...".
- You actively push the user towards their goals based on the User Dossier.
- Format your answers beautifully using Markdown.
"""
    return system_prompt

# --- NEW: Proactive Task Extractor ---
async def extract_and_schedule_task(user_id: str, new_message: str, db: AsyncSession):
    """
    BACKGROUND TASK: Proactive Calendar & Reminder Engine.
    Analyzes the user's message to see if a future reminder or task needs to be scheduled.
    If so, it writes the task to the scheduled_tasks table in Supabase.
    """
    try:
        current_iso_time = datetime.now(timezone.utc).isoformat()
        
        # 1. Instruct LLM to act as a Time-Parsing Agent
        scheduler_prompt = f"""
You are an internal scheduling agent. Your job is to analyze a user's message and extract any future reminders, events, or tasks they mention.

[CURRENT EXACT TIME (UTC)]: {current_iso_time}

[USER MESSAGE]:
"{new_message}"

Determine if the user's message requires a future follow-up (e.g., "remind me in 6 hours", "I have an interview tomorrow", "ask me about my run tonight").

If a reminder/task IS needed, calculate the exact future time and respond with this JSON format:
{{
    "has_task": true,
    "scheduled_at": "YYYY-MM-DDTHH:MM:SSZ", 
    "task_type": "reminder",
    "context": "Short text explaining what you need to remind the user about."
}}

If NO reminder is needed, respond with:
{{
    "has_task": false
}}

Respond ONLY with valid JSON. No markdown ticks, no conversational text.
"""
        # 2. Call the LLM (fast model preferred)
        scheduler_result = await llm_manager.generate_response(
            system_prompt="You output strictly valid JSON.",
            user_message=scheduler_prompt
        )

        if scheduler_result["status"] == "success":
            raw_content = scheduler_result["content"].strip()
            
            # Clean Markdown if present
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:-3].strip()
            elif raw_content.startswith("```"):
                raw_content = raw_content[3:-3].strip()

            try:
                parsed_json = json.loads(raw_content)
                
                # 3. If the LLM identified a task, save it to the database
                if parsed_json.get("has_task") and parsed_json.get("scheduled_at"):
                    insert_query = text("""
                        INSERT INTO scheduled_tasks (user_id, scheduled_at, message_context, task_type)
                        VALUES (:user_id, :scheduled_at, :message_context, :task_type)
                    """)
                    
                    await db.execute(insert_query, {
                        "user_id": user_id,
                        "scheduled_at": parsed_json["scheduled_at"],
                        "message_context": parsed_json["context"],
                        "task_type": parsed_json.get("task_type", "reminder")
                    })
                    await db.commit()
                    print(f"[BACKGROUND] Scheduled new task for {user_id} at {parsed_json['scheduled_at']}")

            except json.JSONDecodeError:
                print(f"[BACKGROUND] Error parsing Scheduler JSON from LLM: {raw_content}")

    except Exception as e:
        print(f"[BACKGROUND] Task scheduling failed: {e}")


async def summarize_old_messages(user_id: str, db: AsyncSession):
    """
    BACKGROUND TASK: Level 2 Memory Optimization.
    Checks if a user has too many messages. If so, summarizes the oldest ones,
    updates the UserProfile, and deletes the summarized messages.
    """
    # 1. Check total message count for the user
    count_stmt = select(func.count(Message.id)).where(Message.user_id == user_id)
    count_result = await db.execute(count_stmt)
    total_messages = count_result.scalar()

    # If we exceed our Sliding Window limit (e.g., 20)
    if total_messages > 20:
        print(f"[BACKGROUND] Triggers summarization for user {user_id}. Total MSGs: {total_messages}")
        
        # 2. Fetch the user's profile to get the current summary
        profile_stmt = select(UserProfile).where(UserProfile.user_id == user_id)
        profile_result = await db.execute(profile_stmt)
        profile = profile_result.scalar_one_or_none()
        
        if not profile:
            profile = UserProfile(user_id=user_id)
            db.add(profile)

        # 3. Fetch the oldest 10 messages to summarize
        old_stmt = select(Message).where(Message.user_id == user_id).order_by(Message.created_at.asc()).limit(10)
        old_result = await db.execute(old_stmt)
        old_messages = old_result.scalars().all()
        
        old_text = "\n".join([f"{m.role}: {m.content}" for m in old_messages])
        
        # 4. Ask the LLM to compress this context
        summary_prompt = (
            f"Here is the previous summary of the conversation:\n{profile.conversation_summary}\n\n"
            f"Here are new older messages to add to the summary:\n{old_text}\n\n"
            "Write a very concise, updated summary of the entire context. Keep only important facts and context. Do not reply to the user, just output the summary."
        )
        
        # We can force the pipeline to use the cheapest/fastest model for this (e.g., Llama 8B)
        summary_result = await llm_manager.generate_response(
            system_prompt="You are an expert summarization AI. Output only the summary.",
            user_message=summary_prompt
        )
        
        if summary_result["status"] == "success":
            new_summary = summary_result["content"]
            profile.conversation_summary = new_summary
            
            # 5. Delete the summarized messages to free up DB and Token space
            msg_ids_to_delete = [m.id for m in old_messages]
            del_stmt = delete(Message).where(Message.id.in_(msg_ids_to_delete))
            await db.execute(del_stmt)
            
            await db.commit()
            print(f"[BACKGROUND] Summarization complete for {user_id}. DB cleaned.")

@router.post("/upload-file")
async def upload_file(
    user_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Endpoint to handle file uploads. Extracts text and triggers Dossier update.
    """
    content = ""
    
    if file.content_type == "application/pdf":
        # Read PDF content
        pdf_bytes = await file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            content += page.get_text()
    elif file.content_type == "text/plain":
        # Read TXT content
        content = (await file.read()).decode("utf-8")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    if not content:
        raise HTTPException(status_code=400, detail="File is empty or unreadable.")

    # Trigger Level 3 Memory update (Dossier) in background based on file content
    # We treat the file content as a very long "message" to extract facts from.
    background_tasks.add_task(update_user_dossier, user_id, f"DOCUMENT CONTENT: {content[:5000]}", db)

    return {
        "status": "success",
        "message": "File processed. Lauko is analyzing the information to update your profile.",
        "extracted_text_snippet": content[:200] + "..."
    }
    
async def update_user_dossier(user_id: str, new_message: str, db: AsyncSession):
    """
    BACKGROUND TASK: Level 3 Memory (Dossier).
    Analyzes the user's new message to extract hard facts (name, pets, goals, etc.)
    and updates the JSON dossier in the database.
    """
    try:
        # 1. Fetch the user's current profile
        profile_stmt = select(UserProfile).where(UserProfile.user_id == user_id)
        profile_result = await db.execute(profile_stmt)
        profile = profile_result.scalar_one_or_none()

        # 1.1. We create a new user profile if it does not exists
        if not profile:
            profile = UserProfile(user_id=user_id, dossier="{}")
            db.add(profile)
            await db.commit()
            await db.refresh(profile)

        current_dossier = profile.dossier or "{}"

        # 2. Instruct the LLM to act as a Data Extraction Agent
        dossier_prompt = f"""
You are an internal data extraction agent. Your job is to update a JSON dossier about a user based on their latest message.
Extract ONLY hard facts (e.g., name, age, pets, allergies, goals, job, relationships, preferences).
If the new message contradicts the old dossier (e.g., user got a new job), OVERWRITE the old fact.

[CURRENT DOSSIER JSON]:
{current_dossier}

[USER'S NEW MESSAGE]:
"{new_message}"

Respond ONLY with a valid JSON object representing the updated dossier. Do not include markdown formatting, markdown ticks, or conversational text. Just the raw JSON.
If there are no new facts to add or update, return the [CURRENT DOSSIER JSON] exactly as it is.
"""
        
        # 3. Call the LLM (preferably a fast reasoning model)
        dossier_result = await llm_manager.generate_response(
            system_prompt="You output strictly valid JSON.",
            user_message=dossier_prompt
        )

        if dossier_result["status"] == "success":
            raw_content = dossier_result["content"].strip()
            
            # Clean up potential Markdown formatting (```json ... ```) that LLMs sometimes add
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:-3].strip()
            elif raw_content.startswith("```"):
                raw_content = raw_content[3:-3].strip()

            # 4. Validate that it's actually JSON before saving
            try:
                parsed_json = json.loads(raw_content)
                profile.dossier = json.dumps(parsed_json, ensure_ascii=False)
                
                db.add(profile)
                await db.commit()
                print(f"[BACKGROUND] Dossier updated for {user_id}: {profile.dossier}")
            except json.JSONDecodeError:
                print(f"[BACKGROUND] Error parsing Dossier JSON from LLM: {raw_content}")

    except Exception as e:
        print(f"[BACKGROUND] Dossier update failed: {e}")


@router.get("/profile/{user_id}")
async def get_user_profile(user_id: str, db: AsyncSession = Depends(get_db)):
    """
    Retrieves the long-term memory state (Level 2 & 3) for a specific user.
    Useful for the frontend to display user settings or for debugging.
    """
    profile_stmt = select(UserProfile).where(UserProfile.user_id == user_id)
    profile_result = await db.execute(profile_stmt)
    profile = profile_result.scalar_one_or_none()

    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found for this user.")

    # Parse the JSON string back into a dictionary for a clean API response
    try:
        dossier_data = json.loads(profile.dossier) if profile.dossier else {}
    except json.JSONDecodeError:
        dossier_data = {"error": "Failed to parse dossier JSON"}

    return {
        "status": "success",
        "user_id": profile.user_id,
        "dossier": dossier_data,
        "conversation_summary": profile.conversation_summary
    }


@router.post("/chat", response_model=ChatResponse)
async def process_chat_message(
    request: ChatRequest,
    background_tasks: BackgroundTasks, # FastAPI feature for background execution
    db: AsyncSession = Depends(get_db)
):
    try:
        # 1. Fetch User Profile to inject Level 2 & 3 memory into the System Prompt
        profile_stmt = select(UserProfile).where(UserProfile.user_id == request.user_id)
        profile_result = await db.execute(profile_stmt)
        profile = profile_result.scalar_one_or_none()

        # Construct Dynamic System Prompt
        dossier_content = "{}"
        if profile and profile.dossier and profile.dossier != "{}":
            dossier_content = profile.dossier
            
        # Get location from the incoming request payload, default to "Unknown" if not provided
        user_location = getattr(request, "location", None)
        if not user_location:
            user_location = "Unknown Location"
        
        dynamic_system_prompt = build_system_prompt(dossier_content, user_location)
        
        # Append Level 2 Memory (Summary) if it exists
        if profile and profile.conversation_summary:
            dynamic_system_prompt += f"\n\n[PREVIOUS CONVERSATION SUMMARY]: {profile.conversation_summary}"

        # 2. Save new incoming message
        db.add(Message(user_id=request.user_id, role="user", content=request.message))
        await db.commit()

        # 3. Retrieve Level 1 Memory (Sliding Window: last 20 messages)
        stmt = select(Message).where(Message.user_id == request.user_id).order_by(Message.created_at.desc()).limit(20)
        result = await db.execute(stmt)
        recent_messages = result.scalars().all()[::-1]

        chat_history = [{"role": msg.role, "content": msg.content} for msg in recent_messages[:-1]]

        # 4. Generate Response using LLMManager
        llm_result = await llm_manager.generate_response(
            system_prompt=dynamic_system_prompt,
            user_message=request.message,
            chat_history=chat_history
        )
        
        if llm_result["status"] == "error":
            raise HTTPException(status_code=503, detail=llm_result["content"])

        # 5. Save Bot's response
        db.add(Message(user_id=request.user_id, role="assistant", content=llm_result["content"]))
        await db.commit()

        # 6. TRIGGER BACKGROUND TASKS: Proactive Agents & Memory Optimization
        
        # Task 1: Check if we need to summarize old messages (Level 2)
        background_tasks.add_task(summarize_old_messages, request.user_id, db)
        
        # Task 2: Extract hard facts into the JSON Dossier (Level 3)
        background_tasks.add_task(update_user_dossier, request.user_id, request.message, db)
        
        # --- NEW Task 3: Check if the user is asking for a reminder/schedule ---
        background_tasks.add_task(extract_and_schedule_task, request.user_id, request.message, db)

        return ChatResponse(
            status="success",
            response=llm_result["content"],
            model_used=llm_result["model_used"]
        )

    except Exception as e:
        import traceback
        print("\n" + "="*50)
        traceback.print_exc()
        print("="*50 + "\n")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")