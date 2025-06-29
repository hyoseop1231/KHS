"""Tests for API endpoints"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from app.main import app
from app.api.endpoints import ChatRequest

client = TestClient(app, base_url="http://localhost")

class TestUploadEndpoint:
    
    @patch('app.api.endpoints.FileValidator.validate_uploaded_file')
    @patch('app.api.endpoints.process_pdf_background')
    def test_upload_pdf_success(self, mock_process, mock_validator):
        """Test successful PDF upload"""
        # Mock validation
        mock_validator.return_value = {
            "is_valid": True,
            "errors": [],
            "file_hash": "test_hash"
        }
        
        # Create a test PDF file
        test_pdf_content = b"%PDF-1.4\ntest content\n%%EOF"
        
        response = client.post(
            "/api/upload_pdf/",
            files={
                "files": ("test.pdf", test_pdf_content, "application/pdf")
            },
            data={
                "ocr_correction_enabled": "false",
                "llm_correction_enabled": "false"
            }
        )
        
        assert response.status_code == 202
        data = response.json()
        assert "document_id" in data
        assert "file_hash" in data
        assert data["filename"] == "test.pdf"
    
    def test_upload_invalid_file_type(self):
        """Test upload with invalid file type"""
        test_content = b"not a pdf"
        
        response = client.post(
            "/api/upload_pdf/",
            files={
                "files": ("test.txt", test_content, "text/plain")
            },
            data={
                "ocr_correction_enabled": "false",
                "llm_correction_enabled": "false"
            }
        )
        
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]
    
    def test_upload_empty_file(self):
        """Test upload with empty file"""
        response = client.post(
            "/api/upload_pdf/",
            files={
                "files": ("test.pdf", b"", "application/pdf")
            },
            data={
                "ocr_correction_enabled": "false",
                "llm_correction_enabled": "false"
            }
        )
        
        assert response.status_code == 400
        assert "Empty file uploaded" in response.json()["detail"]
    
    @patch('app.api.endpoints.FileValidator.validate_uploaded_file')
    def test_upload_validation_failure(self, mock_validator):
        """Test upload with validation failure"""
        # Mock validation failure
        mock_validator.return_value = {
            "is_valid": False,
            "errors": ["Invalid MIME type"],
            "file_hash": ""
        }
        
        test_pdf_content = b"fake pdf content"
        
        response = client.post(
            "/api/upload_pdf/",
            files={
                "files": ("test.pdf", test_pdf_content, "application/pdf")
            },
            data={
                "ocr_correction_enabled": "false",
                "llm_correction_enabled": "false"
            }
        )
        
        assert response.status_code == 400
        assert "File validation failed" in response.json()["detail"]

class TestChatEndpoint:
    
    @patch('app.api.endpoints.get_embeddings')
    @patch('app.services.vector_db_service.search_multimodal_content')
    @patch('app.api.endpoints.process_llm_chat_request')
    def test_chat_success(self, mock_llm, mock_search, mock_embeddings):
        """Test successful chat request"""
        # Mock embeddings
        mock_embeddings.return_value = [[0.1, 0.2, 0.3]]
        
        # Mock search results
        mock_search.return_value = [
            {
                "text": "Sample document text",
                "metadata": {"source_document_id": "doc1"}
            }
        ]
        
        # Mock LLM response
        mock_llm.return_value = "This is the AI response."
        
        response = client.post(
            "/api/chat/",
            json={
                "query": "What is this document about?",
                "model_name": "test_model",
                "lang": "ko"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What is this document about?"
        assert data["response"] == "This is the AI response."
    
    def test_chat_empty_query(self):
        """Test chat with empty query"""
        response = client.post(
            "/api/chat/",
            json={"query": ""}
        )
        
        assert response.status_code == 400
        assert "Query not provided" in response.json()["detail"]
    
    def test_chat_invalid_document_id(self):
        """Test chat with invalid document ID"""
        response = client.post(
            "/api/chat/",
            json={
                "query": "Test query",
                "document_ids": ["invalid@id"]
            }
        )
        
        assert response.status_code == 400
        assert "Invalid document ID" in response.json()["detail"]
    
    @patch('app.api.endpoints.sanitize_input')
    def test_chat_input_sanitization(self, mock_sanitize):
        """Test that input is properly sanitized"""
        mock_sanitize.return_value = "cleaned query"
        
        # This test ensures sanitize_input is called
        with patch('app.api.endpoints.get_embeddings') as mock_embeddings:
            mock_embeddings.return_value = [[0.1, 0.2, 0.3]]
            with patch('app.services.vector_db_service.search_multimodal_content') as mock_search:
                mock_search.return_value = []
                with patch('app.api.endpoints.process_llm_chat_request') as mock_llm:
                    mock_llm.return_value = "response"
                    
                    response = client.post(
                        "/api/chat/",
                        json={"query": "  malicious input  "}
                    )
                    
                    mock_sanitize.assert_called_once()

class TestStatusEndpoints:
    
    def test_upload_status_existing(self):
        """Test getting status for existing document"""
        # Mock the status dictionary
        with patch('app.api.endpoints.pdf_processing_status', {"doc123": {"step": "Done", "message": "Complete"}}):
            response = client.get("/api/upload_status/doc123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["step"] == "Done"
            assert data["message"] == "Complete"
    
    def test_upload_status_nonexistent(self):
        """Test getting status for non-existent document"""
        response = client.get("/api/upload_status/nonexistent")
        
        assert response.status_code == 200
        data = response.json()
        assert data["step"] == "Unknown"
    
    @patch('app.api.endpoints.requests.get')
    def test_ollama_status_running(self, mock_get):
        """Test Ollama status when running"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        response = client.get("/api/ollama/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
    
    @patch('app.api.endpoints.requests.get')
    def test_ollama_status_not_running(self, mock_get):
        """Test Ollama status when not running"""
        mock_get.side_effect = Exception("Connection failed")
        
        response = client.get("/api/ollama/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unreachable"