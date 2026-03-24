import logging
from groq import AsyncGroq
from openai import AsyncOpenAI
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.core.config import settings

# Configure logging to monitor model fallbacks in the terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lauko-LLM-Manager")

class LLMManager:
    """
    Resilient LLM Manager that implements a fallback pipeline.
    Ensures high availability by switching models if one fails.
    Now equipped with exponential backoff retries and JSON mode.
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def _execute_api_call(self, client_type: str, request_kwargs: dict):
        """
        Internal method to execute the API call with Tenacity retries.
        If a network timeout occurs, it will retry up to 3 times before failing.
        """
        if client_type == "groq":
            response = await self.groq_client.chat.completions.create(**request_kwargs)
            return response.choices[0].message.content
        elif client_type == "openrouter":
            response = await self.openrouter_client.chat.completions.create(**request_kwargs)
            return response.choices[0].message.content
        else:
            raise ValueError(f"Unknown client type: {client_type}")

    async def generate_response(
        self, 
        system_prompt: str, 
        user_message: str, 
        chat_history: list = None, 
        require_json: bool = False, 
        model: str = None
    ) -> dict:
        """
        Iterates through the model pipeline. Gracefully falls back to the next model
        if a rate limit or server error occurs. Supports forced models and JSON enforcement.
        """
        messages = [{"role": "system", "content": system_prompt}]
        
        # Inject short-term memory (history) if provided
        if chat_history:
            messages.extend(chat_history)
            
        messages.append({"role": "user", "content": user_message})

        # If a specific model is requested (e.g., for fast background tasks), override the pipeline
        pipeline_to_use = self.models_pipeline
        if model:
            # Find the requested model in the pipeline to determine its client_type
            specific_config = next((cfg for cfg in self.models_pipeline if cfg["model"] == model), None)
            if specific_config:
                pipeline_to_use = [specific_config]
            else:
                # Default to openrouter if the forced model isn't in the predefined list
                pipeline_to_use = [{"client_type": "openrouter", "model": model, "max_chars": 30000}]

        for config in pipeline_to_use:
            client_type = config["client_type"]
            model_name = config["model"]
            
            logger.info(f"Attempting request to {model_name} via {client_type}...")

            # Prepare standard request arguments
            request_kwargs = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024
            }

            # Enforce JSON mode if the background task requires it
            if require_json:
                request_kwargs["response_format"] = {"type": "json_object"}

            try:
                # Call the internal method that is protected by Tenacity retries
                content = await self._execute_api_call(client_type, request_kwargs)
                
                logger.info(f"Success! Response received from {model_name}.")
                
                return {
                    "status": "success",
                    "content": content,
                    "model_used": model_name
                }

            except Exception as e:
                # If it fails even after 3 internal Tenacity retries, we catch it here and fall back
                logger.error(f"Error calling {model_name} after retries: {e}. Switching to fallback...")
                continue 

        logger.critical("WARNING: All models in the pipeline failed!")
        return {
            "status": "error",
            "content": "Sorry, my neural pathways are currently overloaded. Give me a moment to recover.",
            "model_used": "none"
        }

# Create a singleton instance to be imported by the API endpoints
llm_manager = LLMManager()