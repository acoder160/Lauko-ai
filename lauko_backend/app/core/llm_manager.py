import logging
from groq import AsyncGroq
from openai import AsyncOpenAI
from app.core.config import settings

# Configure logging to monitor model fallbacks in the terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lauko-LLM-Manager")

class LLMManager:
    """
    Resilient LLM Manager that implements a fallback pipeline.
    Ensures high availability by switching models if one fails.
    """
    def __init__(self):
        # Initialize asynchronous clients
        self.groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.openrouter_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY,
        )

        # The Fallback Pipeline (Juggling Magic v3.0)
        self.models_pipeline = [
            {"client_type": "openrouter", "model": "deepseek/deepseek-r1", "max_chars": 40000},
            {"client_type": "openrouter", "model": "google/gemini-2.0-flash-001", "max_chars": 60000},
            {"client_type": "groq", "model": "llama-3.3-70b-versatile", "max_chars": 28000},
            {"client_type": "openrouter", "model": "qwen/qwen-2.5-72b-instruct", "max_chars": 30000},
            {"client_type": "openrouter", "model": "stepfun/step-3.5-flash", "max_chars": 35000},
            {"client_type": "openrouter", "model": "mistralai/mistral-small-24b-instruct-2501", "max_chars": 25000},
            {"client_type": "groq", "model": "llama-3.1-8b-instant", "max_chars": 12000}
        ]

    async def generate_response(self, system_prompt: str, user_message: str, chat_history: list = None) -> dict:
        """
        Iterates through the model pipeline. Gracefully falls back to the next model
        if a rate limit or server error occurs. Supports conversation history.
        """
        messages = [{"role": "system", "content": system_prompt}]
        
        # Inject short-term memory (history) if provided
        if chat_history:
            messages.extend(chat_history)
            
        messages.append({"role": "user", "content": user_message})

        for config in self.models_pipeline:
            client_type = config["client_type"]
            model_name = config["model"]
            
            logger.info(f"Attempting request to {model_name} via {client_type}...")

            try:
                if client_type == "groq":
                    response = await self.groq_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=1024
                    )
                    content = response.choices[0].message.content
                
                elif client_type == "openrouter":
                    response = await self.openrouter_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=1024
                    )
                    content = response.choices[0].message.content
                else:
                    logger.warning(f"Unknown client type: {client_type}")
                    continue

                logger.info(f"Success! Response received from {model_name}.")
                
                return {
                    "status": "success",
                    "content": content,
                    "model_used": model_name
                }

            except Exception as e:
                logger.error(f"Error calling {model_name}: {e}. Switching to fallback...")
                continue 

        logger.critical("WARNING: All models in the pipeline failed!")
        return {
            "status": "error",
            "content": "Sorry, my neural pathways are currently overloaded. Give me a moment to recover.",
            "model_used": "none"
        }

# Create a singleton instance to be imported by the API endpoints
llm_manager = LLMManager()