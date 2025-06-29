"""Tests for document management functionality"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.utils.file_manager import DocumentFileManager
from app.services.vector_db_service import delete_document, delete_all_documents

client = TestClient(app)

class TestDocumentManagement:
    
    @patch('app.services.vector_db_service.delete_multimodal_document')
    @patch('app.utils.file_manager.DocumentFileManager.delete_file_by_document_id')
    def test_delete_document_success(self, mock_file_delete, mock_db_delete):
        """Test successful document deletion"""
        # Mock successful deletion from both DB and file system
        mock_db_delete.return_value = True
        mock_file_delete.return_value = True
        
        response = client.delete("/api/documents/test_doc_123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "test_doc_123"
        assert data["deleted_from_db"] == True
        assert data["deleted_from_files"] == True
        assert "deleted successfully" in data["message"]
    
    @patch('app.services.vector_db_service.delete_multimodal_document')
    @patch('app.utils.file_manager.DocumentFileManager.delete_file_by_document_id')
    def test_delete_document_not_found(self, mock_file_delete, mock_db_delete):
        """Test deletion when document doesn't exist"""
        # Mock no deletion from both DB and file system
        mock_db_delete.return_value = False
        mock_file_delete.return_value = False
        
        response = client.delete("/api/documents/nonexistent_doc")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_delete_document_invalid_id(self):
        """Test deletion with invalid document ID"""
        response = client.delete("/api/documents/invalid@id")
        
        assert response.status_code == 400
        assert "Invalid document ID" in response.json()["detail"]
    
    @patch('app.services.vector_db_service.delete_all_multimodal_documents')
    @patch('app.services.vector_db_service.delete_all_documents')
    @patch('app.utils.file_manager.DocumentFileManager.delete_all_files')
    def test_delete_all_documents_success(self, mock_file_delete_all, mock_db_delete_all, mock_multimodal_db_delete_all):
        """Test successful deletion of all documents"""
        # Mock successful deletion
        mock_db_delete_all.return_value = 5  # 5 documents deleted
        mock_multimodal_db_delete_all.return_value = 5 # 5 multimodal documents deleted
        mock_file_delete_all.return_value = 5  # 5 files deleted
        
        response = client.delete("/api/documents")
        
        assert response.status_code == 200
        data = response.json()
        assert data["deleted_documents_count"] == 5
        assert data["deleted_files_count"] == 5
        assert "All documents deleted successfully" in data["message"]
    
    @patch('app.services.vector_db_service.get_multimodal_document_info')
    @patch('app.utils.file_manager.DocumentFileManager.get_file_info')
    def test_get_document_details_success(self, mock_file_info, mock_db_info):
        """Test getting document details"""
        # Mock document info
        mock_db_info.return_value = {
            "document_id": "test_doc",
            "text_chunks": 10,
            "images": 2,
            "tables": 1
        }
        mock_file_info.return_value = {
            "filename": "test_doc_file.pdf",
            "size_mb": 2.5
        }
        
        response = client.get("/api/documents/test_doc")
        
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "test_doc"
        assert data["db_info"]["text_chunks"] == 10
        assert data["file_info"]["size_mb"] == 2.5
    
    @patch('app.services.vector_db_service.get_multimodal_document_info')
    def test_get_document_details_not_found(self, mock_db_info):
        """Test getting details for non-existent document"""
        mock_db_info.return_value = None
        
        response = client.get("/api/documents/nonexistent")
        
        assert response.status_code == 404
        assert "not found in database" in response.json()["detail"]
    
    @patch('app.services.vector_db_service.get_all_documents')
    @patch('app.utils.file_manager.DocumentFileManager.cleanup_orphaned_files')
    def test_cleanup_orphaned_files(self, mock_cleanup, mock_get_docs):
        """Test cleanup of orphaned files"""
        # Mock existing documents
        mock_get_docs.return_value = [
            {"document_id": "doc1"},
            {"document_id": "doc2"}
        ]
        mock_cleanup.return_value = 3  # 3 orphaned files cleaned
        
        response = client.post("/api/documents/cleanup")
        
        assert response.status_code == 200
        data = response.json()
        assert data["cleaned_files_count"] == 3
        assert data["valid_documents_count"] == 2
    
    @patch('app.utils.file_manager.DocumentFileManager.get_storage_stats')
    @patch('app.services.vector_db_service.get_all_documents')
    def test_storage_statistics(self, mock_get_docs, mock_file_stats):
        """Test getting storage statistics"""
        # Mock storage stats
        mock_file_stats.return_value = {
            "total_files": 5,
            "total_size_mb": 25.6,
            "directory_exists": True
        }
        mock_get_docs.return_value = [
            {"document_id": "doc1", "chunk_count": 10},
            {"document_id": "doc2", "chunk_count": 15}
        ]
        
        response = client.get("/api/storage/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["file_storage"]["total_files"] == 5
        assert data["file_storage"]["total_size_mb"] == 25.6
        assert data["vector_db"]["total_documents"] == 2
        assert data["vector_db"]["total_chunks"] == 25

class TestFileManager:
    
    def test_get_uploaded_files_empty_directory(self, temp_dir):
        """Test getting files from empty directory"""
        with patch('app.config.settings.UPLOAD_DIR', temp_dir):
            files = DocumentFileManager.get_uploaded_files()
            assert files == []
    
    def test_delete_file_by_document_id(self, temp_dir):
        """Test deleting file by document ID"""
        # Create a test file
        test_file = os.path.join(tmp_path, "test_doc_123_sample.pdf")
        with open(test_file, "w") as f:
            f.write("test content")
        
        with patch('app.config.settings.UPLOAD_DIR', temp_dir):
            result = DocumentFileManager.delete_file_by_document_id("test_doc_123")
            assert result == True
            assert not os.path.exists(test_file)
    
    def test_delete_file_nonexistent(self, temp_dir):
        """Test deleting non-existent file"""
        with patch('app.config.settings.UPLOAD_DIR', temp_dir):
            result = DocumentFileManager.delete_file_by_document_id("nonexistent")
            assert result == False
    
    def test_delete_all_files(self, temp_dir):
        """Test deleting all files"""
        # Create test files
        for i in range(3):
            test_file = os.path.join(temp_dir, f"doc_{i}_test.pdf")
            with open(test_file, "w") as f:
                f.write(f"test content {i}")
        
        with patch('app.config.settings.UPLOAD_DIR', temp_dir):
            count = DocumentFileManager.delete_all_files()
            assert count == 3
            # Check that no PDF files remain
            remaining_files = [f for f in os.listdir(temp_dir) if f.endswith('.pdf')]
            assert len(remaining_files) == 0
    
    def test_cleanup_orphaned_files(self, temp_dir):
        """Test cleanup of orphaned files"""
        # Create test files
        valid_files = ["doc1_file.pdf", "doc2_file.pdf"]
        orphaned_files = ["orphan1_old.pdf", "orphan2_old.pdf"]
        
        for filename in valid_files + orphaned_files:
            test_file = os.path.join(tmp_path, filename)
            with open(test_file, "w") as f:
                f.write("test content")
        
        with patch('app.config.settings.UPLOAD_DIR', temp_dir):
            # Only doc1 and doc2 are valid
            count = DocumentFileManager.cleanup_orphaned_files(["doc1", "doc2"])
            assert count == 2  # 2 orphaned files cleaned
            
            # Check that valid files remain and orphaned files are gone
            remaining_files = os.listdir(temp_dir)
            assert "doc1_file.pdf" in remaining_files
            assert "doc2_file.pdf" in remaining_files
            assert "orphan1_old.pdf" not in remaining_files
            assert "orphan2_old.pdf" not in remaining_files
    
    def test_get_storage_stats(self, tmp_path):
        """Test getting storage statistics"""
        # Create test files with known sizes
        test_content = "x" * 1024  # 1KB content
        for i in range(3):
            test_file = os.path.join(temp_dir, f"doc_{i}_test.pdf")
            with open(test_file, "w") as f:
                f.write(test_content)
        
        with patch('app.config.settings.UPLOAD_DIR', temp_dir):
            stats = DocumentFileManager.get_storage_stats()
            assert stats["total_files"] == 3
            assert stats["total_size_bytes"] == 3 * 1024  # 3KB total
            assert stats["directory_exists"] == True