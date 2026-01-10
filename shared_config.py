"""Shared configuration for LLM initialization."""
import os

from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ASSISTANT_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"


def create_llm():
    """Create ChatOpenAI instance with OpenRouter configuration.
    
    Returns:
        ChatOpenAI instance configured for OpenRouter.
    """
    from langchain_openai import ChatOpenAI
    
    
    return ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        model=ASSISTANT_MODEL
    )
