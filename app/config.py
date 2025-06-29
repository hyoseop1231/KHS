import os
from dotenv import load_dotenv
from typing import Optional
from pathlib import Path

load_dotenv()
class Settings:
    """Application settings with environment variable support"""
    
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # File handling
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "100")) * 1024 * 1024  # 100MB default
    ALLOWED_EXTENSIONS: list = os.getenv("ALLOWED_EXTENSIONS", ".pdf").split(",")
    
    # Vector DB settings
    CHROMA_DATA_PATH: str = os.getenv("CHROMA_DATA_PATH", "vector_db_data")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "pdf_documents_collection")
    
    # Text processing settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask")
    
    # Ollama LLM settings
    OLLAMA_API_URL: str = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
    OLLAMA_DEFAULT_MODEL: str = os.getenv("OLLAMA_DEFAULT_MODEL", "gemma3n:e4b")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.5"))
    LLM_NUM_PREDICT_TEXT: int = int(os.getenv("LLM_NUM_PREDICT_TEXT", "2048"))
    LLM_NUM_PREDICT_MULTIMODAL: int = int(os.getenv("LLM_NUM_PREDICT_MULTIMODAL", "3072"))
    
    # OCR settings
    TESSERACT_CMD: Optional[str] = os.getenv("TESSERACT_CMD")
    OCR_LANGUAGES: str = os.getenv("OCR_LANGUAGES", "kor+eng")
    OCR_DPI: int = int(os.getenv("OCR_DPI", "300"))
    OCR_CORRECTION_ENABLED: bool = os.getenv("OCR_CORRECTION_ENABLED", "True").lower() == "true"
    OCR_CORRECTION_USE_LLM: bool = os.getenv("OCR_CORRECTION_USE_LLM", "True").lower() == "true"
    
    # Search settings
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "3"))
    
    
    # Streaming settings
    ENABLE_STREAMING: bool = os.getenv("ENABLE_STREAMING", "True").lower() == "true"
    
    # Performance optimization settings
    OCR_MAX_WORKERS: int = int(os.getenv("OCR_MAX_WORKERS", "8"))
    OCR_BATCH_SIZE: int = int(os.getenv("OCR_BATCH_SIZE", "4"))
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    ENABLE_PARALLEL_SEARCH: bool = os.getenv("ENABLE_PARALLEL_SEARCH", "True").lower() == "true"
    ENABLE_ASYNC_LLM: bool = os.getenv("ENABLE_ASYNC_LLM", "True").lower() == "true"
    CONTEXT_COMPRESSION_MAX_TOKENS: int = int(os.getenv("CONTEXT_COMPRESSION_MAX_TOKENS", "2000"))
    
    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "http://localhost:8000").split(",")
    
    # Logging settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "app.log")
    
    def __init__(self):
        # Create necessary directories
        Path(self.UPLOAD_DIR).mkdir(exist_ok=True)
        Path(self.CHROMA_DATA_PATH).mkdir(exist_ok=True)
        
        # Set Tesseract path if provided
        if self.TESSERACT_CMD:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = self.TESSERACT_CMD
        # Warn if SECRET_KEY is default
        import logging
        if self.SECRET_KEY == "your-secret-key-change-in-production":
            if not self.DEBUG:
                # In production, this should be an error, not just a warning
                raise ValueError(
                    "SECRET_KEY must be set to a secure value in production environment. "
                    "Please set SECRET_KEY environment variable."
                )
            else:
                logging.getLogger(__name__).warning(
                    "SECRET_KEY is using default value; please set SECRET_KEY in environment for production"
                )

# Global settings instance
settings = Settings()