"""Test configuration and fixtures"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from app.config import settings

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing"""
    # This is a minimal PDF structure - in real tests you'd use a proper PDF
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n0000000053 00000 n \n0000000111 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n178\n%%EOF"

@pytest.fixture
def mock_embedding_model():
    """Mock embedding model for testing"""
    with patch('app.services.text_processing_service.model_manager') as mock_manager:
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_manager.get_model.return_value = mock_model
        yield mock_model

@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing"""
    with patch('app.services.llm_service.get_llm_response') as mock_llm:
        mock_llm.return_value = "This is a test response from the LLM."
        yield mock_llm

@pytest.fixture
def mock_ocr_service():
    """Mock OCR service for testing"""
    with patch('app.services.ocr_service.extract_text_from_pdf') as mock_ocr:
        mock_ocr.return_value = "This is extracted text from the PDF."
        yield mock_ocr