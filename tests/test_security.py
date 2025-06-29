"""Tests for security utilities"""

import pytest
import tempfile
import os
from pathlib import Path
from app.utils.security import FileValidator, sanitize_input, validate_document_id

class TestFileValidator:
    
    def test_validate_file_extension_valid(self):
        """Test valid PDF file extension"""
        assert FileValidator.validate_file_extension("document.pdf") == True
        assert FileValidator.validate_file_extension("DOCUMENT.PDF") == True
    
    def test_validate_file_extension_invalid(self):
        """Test invalid file extensions"""
        assert FileValidator.validate_file_extension("document.txt") == False
        assert FileValidator.validate_file_extension("document.docx") == False
        assert FileValidator.validate_file_extension("document") == False
        assert FileValidator.validate_file_extension("") == False
        assert FileValidator.validate_file_extension(None) == False
    
    def test_validate_file_size_valid(self):
        """Test valid file size"""
        assert FileValidator.validate_file_size(1024) == True  # 1KB
        assert FileValidator.validate_file_size(50 * 1024 * 1024) == True  # 50MB
    
    def test_validate_file_size_invalid(self):
        """Test invalid file size"""
        assert FileValidator.validate_file_size(200 * 1024 * 1024) == False  # 200MB (exceeds 100MB limit)
    
    def test_generate_safe_filename(self):
        """Test safe filename generation"""
        original = "test file (1).pdf"
        doc_id = "doc123"
        safe = FileValidator.generate_safe_filename(original, doc_id)
        
        assert doc_id in safe
        assert safe.startswith(doc_id)
        assert "(" in safe  # Parentheses are now allowed
        assert ")" in safe
        
    def test_generate_safe_filename_korean(self):
        """Test safe filename generation with Korean characters"""
        original = "주물기술총서 Vol. 04 - 주조방안.pdf"
        doc_id = "주물기술총서_Vol._04___주조방안_4873719f"
        safe = FileValidator.generate_safe_filename(original, doc_id)
        
        assert doc_id in safe
        assert safe.startswith(doc_id)
        assert "주물기술총서" in safe
        assert ".pdf" in safe
        
    def test_generate_safe_filename_forbidden_chars(self):
        """Test safe filename generation removes forbidden characters"""
        original = 'test<file>with"forbidden*chars?.pdf'
        doc_id = "doc123"
        safe = FileValidator.generate_safe_filename(original, doc_id)
        
        assert doc_id in safe
        # Forbidden characters should be replaced with underscores
        forbidden_chars = ['<', '>', '"', '*', '?']
        for char in forbidden_chars:
            assert char not in safe
    
    def test_calculate_file_hash(self, tmp_path):
        """Test file hash calculation"""
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("test content")
        
        hash1 = FileValidator.calculate_file_hash(str(test_file))
        hash2 = FileValidator.calculate_file_hash(str(test_file))
        
        assert hash1 == hash2  # Same file should produce same hash
        assert len(hash1) == 64  # SHA256 produces 64 character hex string
    
    def test_calculate_file_hash_nonexistent(self):
        """Test hash calculation for non-existent file"""
        hash_result = FileValidator.calculate_file_hash("/nonexistent/file.txt")
        assert hash_result == ""

class TestInputSanitization:
    
    def test_sanitize_input_normal(self):
        """Test normal input sanitization"""
        input_text = "This is a normal query about documents."
        result = sanitize_input(input_text)
        assert result == input_text.strip()
    
    def test_sanitize_input_empty(self):
        """Test empty input"""
        assert sanitize_input("") == ""
        assert sanitize_input(None) == ""
        assert sanitize_input("   ") == ""
    
    def test_sanitize_input_too_long(self):
        """Test input that's too long"""
        long_input = "a" * 2000
        result = sanitize_input(long_input, max_length=100)
        assert len(result) == 100
    
    def test_sanitize_input_with_whitespace(self):
        """Test input with leading/trailing whitespace"""
        input_text = "  test query  "
        result = sanitize_input(input_text)
        assert result == "test query"

class TestDocumentIdValidation:
    
    def test_validate_document_id_valid(self):
        """Test valid document IDs"""
        assert validate_document_id("doc123") == True
        assert validate_document_id("test_document_456") == True
        assert validate_document_id("DOC-789") == True
        assert validate_document_id("file_abc-123_xyz") == True
        # Korean characters and dots should be allowed
        assert validate_document_id("주물기술총서_Vol._01___주물결함_9cb2ca0c") == True
        assert validate_document_id("한글문서.pdf") == True
        assert validate_document_id("document.with.dots") == True
        assert validate_document_id("doc with spaces") == True  # Spaces now allowed
    
    def test_validate_document_id_invalid(self):
        """Test invalid document IDs"""
        assert validate_document_id("") == False
        assert validate_document_id(None) == False
        # Test forbidden characters
        assert validate_document_id("doc/123") == False
        assert validate_document_id("doc\\123") == False
        assert validate_document_id("doc:123") == False
        assert validate_document_id("doc*123") == False
        assert validate_document_id("doc?123") == False
        assert validate_document_id('doc"123') == False
        assert validate_document_id("doc<123") == False
        assert validate_document_id("doc>123") == False
        assert validate_document_id("doc|123") == False
        assert validate_document_id("a" * 201) == False  # Too long (increased limit)