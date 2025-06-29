"""Tests for text processing service"""

import pytest
from unittest.mock import Mock, patch
from app.services.text_processing_service import (
    split_text_into_chunks, 
    get_embeddings,
    EmbeddingModelManager
)
from app.services.ocr_service import correct_foundry_terms
from app.utils.exceptions import EmbeddingError

import numpy as np

class TestTextSplitting:
    
    def test_split_text_basic(self):
        """Test basic text splitting"""
        text = "This is a test document. It has multiple sentences. This should be split into chunks."
        chunks = split_text_into_chunks(text, chunk_size=50, chunk_overlap=10)
        
        assert len(chunks) > 0
        assert all(len(chunk) <= 60 for chunk in chunks)  # Allowing some overlap
    
    def test_split_text_empty(self):
        """Test splitting empty text"""
        result = split_text_into_chunks("")
        assert result == []
    
    def test_split_text_short(self):
        """Test splitting text shorter than chunk size"""
        text = "Short text"
        chunks = split_text_into_chunks(text, chunk_size=100, chunk_overlap=20)
        
        assert len(chunks) == 1
        assert chunks[0] == text

class TestFoundryTermCorrection:
    
    def test_correct_foundry_terms(self):
        """Test foundry term correction"""
        text = "주물 공정에서 몰드를 사용합니다."
        corrected = correct_foundry_terms(text)
        
        assert corrected == "주조 공정에서 주형을 사용합니다."
    
    def test_correct_foundry_terms_no_change(self):
        """Test text with no foundry terms"""
        text = "이것은 일반적인 텍스트입니다."
        corrected = correct_foundry_terms(text)
        
        assert corrected == text

class TestEmbeddingGeneration:
    
    @patch('app.services.text_processing_service.model_manager')
    def test_get_embeddings_success(self, mock_manager):
        """Test successful embedding generation"""
        # Mock the model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_manager.get_model.return_value = mock_model
        
        text_chunks = ["첫 번째 텍스트", "두 번째 텍스트"]
        embeddings = get_embeddings(text_chunks)
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 3
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
    
    def test_get_embeddings_empty(self):
        """Test embedding generation with empty input"""
        result = get_embeddings([])
        assert result == []
    
    @patch('app.services.text_processing_service.model_manager')
    def test_get_embeddings_batch_processing(self, mock_manager):
        """Test batch processing of embeddings"""
        # Mock the model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_manager.get_model.return_value = mock_model
        
        # Test with batch_size smaller than input
        text_chunks = ["chunk1", "chunk2", "chunk3", "chunk4"]
        embeddings = get_embeddings(text_chunks, batch_size=2)
        
        # Should call encode twice (2 batches of 2 items each)
        assert mock_model.encode.call_count == 2
        assert len(embeddings) == 4

class TestEmbeddingModelManager:
    
    def test_singleton_pattern(self):
        """Test that EmbeddingModelManager is a singleton"""
        manager1 = EmbeddingModelManager()
        manager2 = EmbeddingModelManager()
        
        assert manager1 is manager2
    
    @patch('app.services.text_processing_service.SentenceTransformer')
    def test_model_loading_success(self, mock_transformer):
        """Test successful model loading"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        
        manager = EmbeddingModelManager()
        # Clear any cached model
        manager._model = None
        
        result = manager.get_model()
        
        assert result is mock_model
        mock_transformer.assert_called_once()
    
    @patch('app.services.text_processing_service.SentenceTransformer')
    def test_model_loading_failure(self, mock_transformer):
        """Test model loading failure"""
        mock_transformer.side_effect = Exception("Model loading failed")
        
        manager = EmbeddingModelManager()
        # Clear any cached model
        manager._model = None
        
        with pytest.raises(EmbeddingError):
            manager.get_model()