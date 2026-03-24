import json
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.llm_manager import llm_manager
from app.prompts.system_prompts import build_scheduler_prompt, build_summary_prompt, build_dossier_prompt
from app.crud import crud_chat

async def extract_and_schedule_task(user_id: str, new_message: str, db: AsyncSession):
    """BACKGROUND TASK: Proactive Calendar Engine using fast Llama 8B."""
    print("[BACKGROUND] Starting task extraction...")
    try:
        current_iso_time = datetime.now(timezone.utc).isoformat()
        scheduler_prompt = build_scheduler_prompt(new_message, current_iso_time)
        
        scheduler_result = await llm_manager.generate_response(
            system_prompt="You output strictly valid JSON.",
            user_message=scheduler_prompt,
            require_json=True,
            model="llama-3.1-8b-instant" 
        )

        if scheduler_result["status"] == "success":
            raw_content = scheduler_result["content"].strip()
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:-3].strip()
            elif raw_content.startswith("```"):
                raw_content = raw_content[3:-3].strip()

            try:
                parsed_json = json.loads(raw_content)
                if parsed_json.get("has_task") and parsed_json.get("scheduled_at"):
                    await crud_chat.schedule_task(
                        db, 
                        user_id, 
                        parsed_json["scheduled_at"], 
                        parsed_json.get("context", ""), 
                        parsed_json.get("task_type", "reminder")
                    )
                    print(f"[BACKGROUND] Scheduled new task for {user_id} at {parsed_json['scheduled_at']}")
            except json.JSONDecodeError:
                print(f"[BACKGROUND] Error parsing Scheduler JSON from LLM: {raw_content}")

    except Exception as e:
        print(f"[BACKGROUND] Task scheduling failed: {e}")

async def summarize_old_messages(user_id: str, db: AsyncSession):
    """BACKGROUND TASK: Level 2 Memory Optimization."""
    try:
        total_messages = await crud_chat.get_total_message_count(db, user_id)

        if total_messages > 20:
            print(f"[BACKGROUND] Triggers summarization for user {user_id}. Total MSGs: {total_messages}")
            
            profile = await crud_chat.get_or_create_user_profile(db, user_id)
            old_messages = await crud_chat.get_oldest_messages(db, user_id, limit=10)
            
            if not old_messages:
                return

            old_text = "\n".join([f"{m.role}: {m.content}" for m in old_messages])
            summary_prompt = build_summary_prompt(profile.conversation_summary or "", old_text)
            
            summary_result = await llm_manager.generate_response(
                system_prompt="You are an expert summarization AI. Output only the summary.",
                user_message=summary_prompt,
                model="llama-3.1-8b-instant" 
            )
            
            if summary_result["status"] == "success":
                new_summary = summary_result["content"]
                profile.conversation_summary = new_summary
                
                msg_ids_to_delete = [m.id for m in old_messages]
                await crud_chat.delete_messages_by_ids(db, msg_ids_to_delete)
                
                db.add(profile)
                await db.commit()
                print(f"[BACKGROUND] Summarization complete for {user_id}. DB cleaned.")
    except Exception as e:
        print(f"[BACKGROUND] Summarization failed: {e}")

async def update_user_dossier(user_id: str, new_message: str, db: AsyncSession):
    """BACKGROUND TASK: Fast Fact Extractor (Micro-Agent)."""
    print("[BACKGROUND] Starting fast fact extraction (Micro-Agent)...")
    try:
        profile = await crud_chat.get_or_create_user_profile(db, user_id)

        try:
            current_dossier_dict = json.loads(profile.dossier) if profile.dossier else {}
        except json.JSONDecodeError:
            current_dossier_dict = {}

        if "unprocessed_facts" not in current_dossier_dict:
            current_dossier_dict["unprocessed_facts"] = []

        dossier_prompt = build_dossier_prompt(new_message)
        
        dossier_result = await llm_manager.generate_response(
            system_prompt="You output strictly valid JSON.",
            user_message=dossier_prompt,
            require_json=True,
            model="llama-3.1-8b-instant" 
        )

        if dossier_result["status"] == "success":
            raw_content = dossier_result["content"].strip()
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:-3].strip()
            elif raw_content.startswith("```"):
                raw_content = raw_content[3:-3].strip()

            try:
                parsed_json = json.loads(raw_content)
                new_facts = parsed_json.get("new_facts", [])
                
                if new_facts:
                    current_dossier_dict["unprocessed_facts"].extend(new_facts)
                    profile.dossier = json.dumps(current_dossier_dict, ensure_ascii=False)
                    
                    db.add(profile)
                    await db.commit()
                    print(f"[BACKGROUND] New facts appended for {user_id}. Dossier buffer size: {len(current_dossier_dict['unprocessed_facts'])}")
                else:
                    print("[BACKGROUND] No new facts detected. Skipping DB write.")

            except json.JSONDecodeError:
                print(f"[BACKGROUND] Error parsing Dossier JSON from LLM: {raw_content}")

    except Exception as e:
        print(f"[BACKGROUND] Fast fact extraction failed: {e}")