from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """
    Schema for incoming chat requests from the client.
    """
    message: str = Field(..., min_length=1, max_length=2000, description="User's input message")
    user_id: str = Field(..., description="Unique identifier for the user")

class ChatResponse(BaseModel):
    """
    Schema for outgoing chat responses.
    """
    status: str
    response: str
    model_used: str