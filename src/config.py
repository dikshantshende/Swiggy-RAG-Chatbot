import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application configuration settings perfectly decoupled from business logic.
    Reads from environment variables and provides sensible defaults.
    """
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    VECTOR_STORE_DIR: str = os.path.join(BASE_DIR, "faiss_index")
    
    # Expected PDF paths (for default processing)
    DEFAULT_PDF_PATH: str = os.path.join(DATA_DIR, "swiggy_annual_report.pdf")
    
    # RAG Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    RETRIEVER_K: int = 5
    
    # Model preferences
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    LLM_MODEL: str = "gemini-2.5-flash"
    
    # API Keys - Expected to be passed via environment or .env file
    GEMINI_API_KEY: str = ""
    
    class Config:
        env_file = ".env"
        extra = "ignore"

# Global settings instance
settings = Settings()
