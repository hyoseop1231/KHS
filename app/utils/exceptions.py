"""Custom exceptions for the application"""

class BaseAppException(Exception):
    """Base exception for all application errors"""
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class FileProcessingError(BaseAppException):
    """Exception raised when file processing fails"""
    pass

class OCRError(FileProcessingError):
    """Exception raised when OCR processing fails"""
    pass

class EmbeddingError(BaseAppException):
    """Exception raised when embedding generation fails"""
    pass

class TextProcessingError(BaseAppException):
    """Exception raised when text processing (e.g., chunking, PDF processing) fails"""
    pass

class VectorDBError(BaseAppException):
    """Exception raised when vector database operations fail"""
    pass

class LLMError(BaseAppException):
    """Exception raised when LLM operations fail"""
    pass

class ConfigurationError(BaseAppException):
    """Exception raised when configuration is invalid"""
    pass

class ValidationError(BaseAppException):
    """Exception raised when input validation fails"""
    pass