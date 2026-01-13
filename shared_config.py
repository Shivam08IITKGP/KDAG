"""Shared configuration for LLM initialization."""
import os

from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
ASSISTANT_MODEL = OPENROUTER_MODEL

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

ROW_DELAY_SECONDS = 2
MAX_QUERIES = 7


def create_llm():
    """Create LLM instance. Defaults to Gemini.
    
    Returns:
        LLM instance (ChatGoogleGenerativeAI).
    """
    # Primary: Gemini
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=1.0
    )

    # Optional: OpenRouter (Commented out)
    # from langchain_openai import ChatOpenAI
    # return ChatOpenAI(
    #     api_key=OPENROUTER_API_KEY,
    #     base_url=OPENROUTER_BASE_URL,
    #     model=OPENROUTER_MODEL
    # )
