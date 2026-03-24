import json
import traceback
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
import fitz  

from app.schemas.chat import ChatRequest, ChatResponse
from app.core.llm_manager import llm_manager
from app.core.database import get_db
from app.crud import crud_chat
from app.prompts.system_prompts import build_system_prompt
from app.services import memory_service

router = APIRouter()

@router.post("/upload-file")
async def upload_file(
    user_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """Endpoint to handle file uploads. Extracts text and triggers Dossier update."""
    content = ""
    
    if file.content_type == "application/pdf":
        pdf_bytes = await file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            content += page.get_text()
    elif file.content_type == "text/plain":
        content = (await file.read()).decode("utf-8")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    if not content:
        raise HTTPException(status_code=400, detail="File is empty or unreadable.")

    # Trigger background service
    background_tasks.add_task(memory_service.update_user_dossier, user_id, f"DOCUMENT CONTENT: {content[:5000]}", db)

    return {
        "status": "success",
        "message": "File processed. Lauko is analyzing the information to update your profile.",
        "extracted_text_snippet": content[:200] + "..."
    }

@router.get("/profile/{user_id}")
async def get_user_profile(user_id: str, db: AsyncSession = Depends(get_db)):
    """Retrieves the user profile and long-term memory state."""
    profile = await crud_chat.get_user_profile(db, user_id)

    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found for this user.")

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
    background_tasks: BackgroundTasks, 
    db: AsyncSession = Depends(get_db)
):
    """Main chat endpoint handling the conversation logic and triggering background agents."""
    try:
        # 1. Fetch Profile & Build Prompt via CRUD and Prompt layers
        profile = await crud_chat.get_user_profile(db, request.user_id)
        dossier_content = profile.dossier if (profile and profile.dossier and profile.dossier != "{}") else "{}"
            
        user_location = getattr(request, "location", None) or "Unknown Location"
        dynamic_system_prompt = build_system_prompt(dossier_content, user_location)
        
        if profile and profile.conversation_summary:
            dynamic_system_prompt += f"\n\n[PREVIOUS CONVERSATION SUMMARY]: {profile.conversation_summary}"

        # 2. Save User Message
        await crud_chat.add_message(db, request.user_id, "user", request.message)

        # 3. Retrieve Context Window
        recent_messages = await crud_chat.get_recent_messages(db, request.user_id, limit=20)
        chat_history = [{"role": msg.role, "content": msg.content} for msg in recent_messages[:-1]]

        # 4. Generate LLM Response
        llm_result = await llm_manager.generate_response(
            system_prompt=dynamic_system_prompt,
            user_message=request.message,
            chat_history=chat_history
        )
        
        if llm_result["status"] == "error":
            raise HTTPException(status_code=503, detail=llm_result["content"])

        # 5. Save Bot Message
        await crud_chat.add_message(db, request.user_id, "assistant", llm_result["content"])

        # 6. Trigger Background Tasks dynamically through the Service layer
        background_tasks.add_task(memory_service.summarize_old_messages, request.user_id, db)
        background_tasks.add_task(memory_service.update_user_dossier, request.user_id, request.message, db)
        background_tasks.add_task(memory_service.extract_and_schedule_task, request.user_id, request.message, db)

        return ChatResponse(
            status="success",
            response=llm_result["content"],
            model_used=llm_result["model_used"]
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")