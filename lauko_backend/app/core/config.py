from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    GROQ_API_KEY: str
    OPENROUTER_API_KEY: str

    # Instruct Pydantic to read from the .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

# Singleton instance to be used across the app
settings = Settings()