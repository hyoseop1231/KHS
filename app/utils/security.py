"""Security utilities for file validation and safety checks"""

import hashlib
import os
from pathlib import Path
from typing import Optional, List
from app.config import settings
from app.utils.logging_config import get_logger
from app.utils.exceptions import ValidationError

logger = get_logger(__name__)

# Try to import python-magic, fall back if not available
try:
    import magic
    HAS_MAGIC = True
    logger.info("python-magic library available for MIME type detection")
except ImportError:
    HAS_MAGIC = False
    logger.warning("python-magic not available. MIME type validation will be skipped.")

class FileValidator:
    """File upload validation and security checks"""
    
    # Safe MIME types for PDF files
    ALLOWED_MIME_TYPES = [
        'application/pdf',
        'application/x-pdf',
    ]
    
    # Maximum file size (from settings)
    MAX_FILE_SIZE = settings.MAX_FILE_SIZE
    
    @staticmethod
    def validate_file_extension(filename: str) -> bool:
        """Validate file extension"""
        if not filename:
            return False
        
        file_ext = Path(filename).suffix.lower()
        return file_ext in settings.ALLOWED_EXTENSIONS
    
    @staticmethod
    def validate_file_size(file_size: int) -> bool:
        """Validate file size"""
        return file_size <= FileValidator.MAX_FILE_SIZE
    
    @staticmethod
    def validate_mime_type(file_path: str) -> bool:
        """Validate MIME type using python-magic or fallback to file signature"""
        if not HAS_MAGIC:
            logger.warning("python-magic not available. Using file signature fallback.")
            return FileValidator._validate_pdf_signature(file_path)
        
        try:
            mime_type = magic.from_file(file_path, mime=True)
            logger.debug(f"Detected MIME type: {mime_type} for file: {file_path}")
            return mime_type in FileValidator.ALLOWED_MIME_TYPES
        except Exception as e:
            logger.warning(f"Error detecting MIME type for {file_path}: {e}. Using file signature fallback.")
            return FileValidator._validate_pdf_signature(file_path)
    
    @staticmethod
    def _validate_pdf_signature(file_path: str) -> bool:
        """Validate PDF file by checking file signature (magic bytes)"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(8)
                # PDF files start with %PDF- (0x255044462D)
                if header.startswith(b'%PDF-'):
                    logger.debug(f"Valid PDF signature detected for: {file_path}")
                    return True
                else:
                    logger.warning(f"Invalid PDF signature for: {file_path}. Header: {header[:8]}")
                    return False
        except Exception as e:
            logger.error(f"Error reading file signature for {file_path}: {e}")
            return False
    
    @staticmethod
    def generate_safe_filename(original_filename: str, document_id: str) -> str:
        """Generate a safe filename"""
        # Remove only problematic characters for file systems
        forbidden_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\0']
        safe_filename = original_filename
        
        for char in forbidden_chars:
            safe_filename = safe_filename.replace(char, '_')
        
        # Remove control characters
        safe_filename = ''.join(c for c in safe_filename if ord(c) >= 32)
        
        # Ensure filename isn't too long (considering filesystem limits)
        if len(safe_filename) > 150:
            name, ext = os.path.splitext(safe_filename)
            safe_filename = name[:140] + ext
        
        return f"{document_id}_{safe_filename}"
    
    @staticmethod
    def calculate_file_hash(file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    @classmethod
    def validate_uploaded_file(cls, file_path: str, original_filename: str, file_size: int) -> dict:
        """Comprehensive file validation"""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "file_hash": ""
        }
        
        # Check file extension
        if not cls.validate_file_extension(original_filename):
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Invalid file extension. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}")
        
        # Check file size
        if not cls.validate_file_size(file_size):
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"File too large. Maximum size: {cls.MAX_FILE_SIZE / (1024*1024):.1f}MB")
        
        # Check if file exists
        if not os.path.exists(file_path):
            validation_result["is_valid"] = False
            validation_result["errors"].append("File not found")
            return validation_result
        
        # Check MIME type
        if not cls.validate_mime_type(file_path):
            validation_result["is_valid"] = False
            validation_result["errors"].append("Invalid file type. Only PDF files are allowed.")
        
        # Calculate file hash for integrity
        file_hash = cls.calculate_file_hash(file_path)
        validation_result["file_hash"] = file_hash
        
        if validation_result["is_valid"]:
            logger.info(f"File validation passed for: {original_filename}")
        else:
            logger.warning(f"File validation failed for: {original_filename}. Errors: {validation_result['errors']}")
        
        return validation_result

def sanitize_input(text: str, max_length: int = 1000) -> str:
    """Sanitize user input text"""
    if not text:
        return ""
    
    # Remove potentially dangerous characters
    sanitized = text.strip()
    
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
        logger.warning(f"Input truncated to {max_length} characters")
    
    return sanitized

def validate_document_id(document_id: str) -> bool:
    """Validate document ID format"""
    if not document_id:
        return False
    
    # Check length
    if len(document_id) > 200:  # Increased for Korean characters
        return False
    
    # Allow alphanumeric characters, hyphens, underscores, dots, and Korean characters
    # Also allow some common symbols found in document names
    forbidden_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\0']
    
    # Check for forbidden characters
    for char in forbidden_chars:
        if char in document_id:
            return False
    
    # Check for control characters
    if any(ord(char) < 32 for char in document_id):
        return False
    
    return True