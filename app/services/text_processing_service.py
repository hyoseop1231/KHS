from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import threading
import re
from typing import List, Dict, Any
from app.config import settings
from app.utils.logging_config import get_logger
from app.utils.exceptions import EmbeddingError, TextProcessingError

from app.services.vector_db_service import store_text_vectors, store_multimodal_content
from app.services.ocr_service import extract_multimodal_content_from_pdf

logger = get_logger(__name__)

# Thread-safe singleton for embedding model
class EmbeddingModelManager:
    _instance = None
    _lock = threading.Lock()
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self):
        if self._model is None:
            # Ensure only one thread loads the model
            with self._lock:
                if self._model is None:
                    try:
                        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
                        self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
                        logger.info("Embedding model loaded successfully")
                    except Exception as e:
                        logger.error(f"Error loading SentenceTransformer model '{settings.EMBEDDING_MODEL}': {e}")
                        raise EmbeddingError(f"Could not load embedding model: {e}", "MODEL_LOAD_ERROR")
        return self._model

# Global model manager instance
model_manager = EmbeddingModelManager()


def split_text_into_chunks(text: str, chunk_size: int = None, chunk_overlap: int = None, apply_correction: bool = False) -> List[str]:
    """
    Splits a long text into smaller, overlapping chunks using Langchain's RecursiveCharacterTextSplitter.
    
    Args:
        text: The input text to be split
        chunk_size: Maximum size of each chunk (default from settings)
        chunk_overlap: Overlap between chunks (default from settings)
        apply_correction: Whether to apply OCR correction to chunks
    """
    if not text:
        return []
    
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        
        # Apply OCR correction if requested
        if apply_correction:
            try:
                from app.services.ocr_correction_service import correct_text_chunks
                chunks = correct_text_chunks(chunks, use_llm=True)
                logger.info(f"Applied OCR correction to {len(chunks)} chunks")
            except Exception as e:
                logger.warning(f"OCR correction failed for chunks: {e}")
        
        logger.info(f"Split text into {len(chunks)} chunks (size: {chunk_size}, overlap: {chunk_overlap})")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text: {e}")
        raise EmbeddingError(f"Text splitting failed: {e}", "TEXT_SPLIT_ERROR")

def get_optimal_batch_size(num_chunks: int, available_memory_gb: float = 4.0) -> int:
    """
    GPU/CPU 메모리와 청크 수에 따른 최적 배치 크기 계산
    """
    # 기본 배치 크기
    base_batch_size = 32
    
    # 메모리에 따른 배치 크기 조정
    if available_memory_gb >= 8:
        max_batch_size = 128
    elif available_memory_gb >= 4:
        max_batch_size = 64
    else:
        max_batch_size = 32
    
    # 청크 수에 따른 배치 크기 조정
    if num_chunks < 50:
        optimal_batch_size = min(16, max_batch_size)
    elif num_chunks < 200:
        optimal_batch_size = min(32, max_batch_size)
    else:
        optimal_batch_size = max_batch_size
    
    return optimal_batch_size

def get_embeddings(text_chunks: List[str], batch_size: int = None) -> List[List[float]]:
    """
    Converts text chunks into vector embeddings with optimized batch processing.
    Returns a list of embeddings, where each embedding is a list of floats.
    """
    if not text_chunks:
        return []
    
    try:
        model = model_manager.get_model()
        
        # 최적 배치 크기 결정
        if batch_size is None:
            batch_size = get_optimal_batch_size(len(text_chunks))
        
        logger.info(f"Generating embeddings for {len(text_chunks)} chunks using '{settings.EMBEDDING_MODEL}' (batch_size: {batch_size})")
        
        # 빈 청크 제거 및 검증
        non_empty_chunks = []
        for i, chunk in enumerate(text_chunks):
            if chunk and chunk.strip() and len(chunk.strip()) >= 2:  # 최소 2자 이상
                non_empty_chunks.append(chunk.strip())
            else:
                logger.debug(f"Skipped empty/short chunk at index {i}")
        
        if len(non_empty_chunks) != len(text_chunks):
            logger.info(f"Filtered out {len(text_chunks) - len(non_empty_chunks)} empty/short chunks")
        
        if not non_empty_chunks:
            logger.warning("No valid chunks found for embedding generation")
            return []
        
        # Process in batches for better memory management
        all_embeddings = []
        total_batches = (len(non_empty_chunks) + batch_size - 1) // batch_size
        
        for i in range(0, len(non_empty_chunks), batch_size):
            batch = non_empty_chunks[i:i + batch_size]
            current_batch_num = i // batch_size + 1
            
            logger.debug(f"Processing batch {current_batch_num}/{total_batches} ({len(batch)} chunks)")
            
            try:
                # Generate embeddings for this batch with optimized settings
                batch_embeddings = model.encode(
                    batch, 
                    convert_to_tensor=False,
                    show_progress_bar=False,
                    batch_size=min(len(batch), 32),  # 메모리 제한을 위해 최대 32개씩
                    normalize_embeddings=True  # 정규화로 성능 향상
                )
                
                # Convert to list format
                if len(batch_embeddings.shape) == 1:
                    # 단일 임베딩인 경우
                    batch_embeddings_list = [batch_embeddings.tolist()]
                else:
                    # 여러 임베딩인 경우
                    batch_embeddings_list = [embedding.tolist() for embedding in batch_embeddings]
                
                all_embeddings.extend(batch_embeddings_list)
                
            except Exception as batch_error:
                logger.error(f"Error processing batch {current_batch_num}: {batch_error}")
                # 실패한 배치에 대해 더미 임베딩 생성
                model_dim = getattr(model, 'get_sentence_embedding_dimension', lambda: 384)()
                dummy_embeddings = [[0.0] * model_dim for _ in batch]
                all_embeddings.extend(dummy_embeddings)
                logger.warning(f"Used dummy embeddings for failed batch {current_batch_num}")
            
            # 메모리 정리
            import gc
            gc.collect()
        
        # 빈 청크에 대한 더미 임베딩 추가 (원래 순서 유지)
        if len(non_empty_chunks) != len(text_chunks):
            final_embeddings = []
            embedding_idx = 0
            for chunk in text_chunks:
                if chunk.strip():
                    final_embeddings.append(all_embeddings[embedding_idx])
                    embedding_idx += 1
                else:
                    # 빈 청크에 대한 제로 벡터
                    model_dim = len(all_embeddings[0]) if all_embeddings else 384
                    final_embeddings.append([0.0] * model_dim)
            all_embeddings = final_embeddings
        
        logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
        return all_embeddings
        
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}")
        raise EmbeddingError(f"Embedding generation failed: {e}", "EMBEDDING_GENERATION_ERROR")

def process_text_only_pdf_and_store(pdf_path: str, document_id: str, filename: str) -> Dict[str, Any]:
    """
    Extracts text from PDF, chunks it, generates embeddings, and stores them in the vector DB.
    This function is for text-only processing.
    """
    logger.info(f"Processing text-only PDF: {pdf_path} (Document ID: {document_id})")
    try:
        # Use extract_multimodal_content_from_pdf but only take text
        extracted_content = extract_multimodal_content_from_pdf(pdf_path, document_id)
        extracted_text = extracted_content.get("text", "")
        
        if not extracted_text.strip():
            logger.warning(f"No text extracted from {pdf_path}. Skipping vector storage.")
            return {"status": "skipped", "reason": "no text extracted"}

        text_chunks = split_text_into_chunks(extracted_text)
        embeddings = get_embeddings(text_chunks)
        
        # Prepare metadata for each chunk
        metadatas = [
            {"source_document_id": document_id, "filename": filename, "chunk_index": i, "content_type": "text"}
            for i in range(len(text_chunks))
        ]
        
        store_text_vectors(document_id, text_chunks, embeddings, metadatas)
        logger.info(f"Successfully processed and stored {len(text_chunks)} text chunks for {pdf_path}")
        return {"status": "success", "chunks_stored": len(text_chunks)}
    except Exception as e:
        logger.error(f"Error processing text-only PDF {pdf_path}: {e}")
        raise TextProcessingError(f"Failed to process text-only PDF: {e}", "PDF_PROCESSING_FAILED")

def process_multimodal_pdf_and_store(pdf_path: str, document_id: str, filename: str, progress_callback: callable = None) -> Dict[str, Any]:
    """
    Extracts multimodal content from PDF, processes it, generates embeddings for text,
    and stores all content in the multimodal vector DB.
    """
    logger.info(f"Processing multimodal PDF: {pdf_path} (Document ID: {document_id})")
    try:
        # OCR 완료 후 시작 (55% 지점부터)
        if progress_callback:
            progress_callback(document_id, 55, "chunking", "텍스트 청크 분할 시작...")
        
        multimodal_content = extract_multimodal_content_from_pdf(pdf_path, document_id)
        
        extracted_text = multimodal_content.get("text", "")
        extracted_images = multimodal_content.get("images", [])
        extracted_tables = multimodal_content.get("tables", [])
        
        text_chunks = []
        text_embeddings = []
        text_metadatas = []
        
        if extracted_text.strip():
            # 청크 분할 (60%)
            if progress_callback:
                progress_callback(document_id, 60, "chunking", "텍스트 청크 분할 중...")
            text_chunks = split_text_into_chunks(extracted_text)
            
            # 임베딩 생성 (70%)
            if progress_callback:
                progress_callback(document_id, 70, "embedding", f"{len(text_chunks)}개 청크 임베딩 생성 중...")
            text_embeddings = get_embeddings(text_chunks)
            
            # 메타데이터 준비 (80%)
            if progress_callback:
                progress_callback(document_id, 80, "metadata", "메타데이터 준비 중...")
            text_metadatas = [
                {"source_document_id": document_id, "filename": filename, "chunk_index": i, "content_type": "text"}
                for i in range(len(text_chunks))
            ]
            logger.info(f"Prepared {len(text_chunks)} text chunks for {pdf_path}")
        else:
            logger.warning(f"No text extracted from {pdf_path}.")
            if progress_callback:
                progress_callback(document_id, 80, "warning", "추출된 텍스트가 없습니다.")

        # 데이터베이스 저장 (90%)
        if progress_callback:
            progress_callback(document_id, 90, "storing", "벡터 데이터베이스에 저장 중...")
        
        store_multimodal_content(
            document_id=document_id,
            content_data={
                "text_chunks": text_chunks,
                "images": extracted_images,
                "tables": extracted_tables
            },
            text_vectors=text_embeddings,
            text_metadatas=text_metadatas
        )
        
        # 완료 (100%)
        if progress_callback:
            progress_callback(document_id, 100, "completed", f"처리 완료! 텍스트: {len(text_chunks)}청크, 이미지: {len(extracted_images)}개, 표: {len(extracted_tables)}개")
        
        logger.info(f"Successfully processed and stored multimodal content for {pdf_path}")
        return {
            "status": "success",
            "text_chunks_stored": len(text_chunks),
            "images_stored": len(extracted_images),
            "tables_stored": len(extracted_tables)
        }
    except Exception as e:
        logger.error(f"Error processing multimodal PDF {pdf_path}: {e}")
        raise TextProcessingError(f"Failed to process multimodal PDF: {e}", "MULTIMODAL_PROCESSING_FAILED")

if __name__ == '__main__':
    # 간단한 테스트용
    print("Text processing service module loaded.")
    print(f"Using embedding model: {settings.EMBEDDING_MODEL}")
    print(f"Chunk size: {settings.CHUNK_SIZE}, Chunk overlap: {settings.CHUNK_OVERLAP}")

    # Example usage (requires a dummy PDF and ChromaDB running)
    # from app.utils.file_manager import save_upload_file
    # import uuid
    # dummy_pdf_path = "path/to/your/dummy.pdf"  # Replace with a real PDF path for testing
    # if os.path.exists(dummy_pdf_path):
    #     print(f"\n--- Testing PDF processing with {dummy_pdf_path} ---")
    #     test_document_id = str(uuid.uuid4())
    #     try:
    #         result = process_multimodal_pdf_and_store(dummy_pdf_path, test_document_id, "dummy.pdf")
    #         print(f"Processing result: {result}")
    #         # info = get_multimodal_document_info(test_document_id)  # This function is now in vector_db_service
    #         # print(f"Document info after processing: {info}")
    #         # Clean up
    #         # delete_multimodal_document(test_document_id)  # This function is now in vector_db_service
    #         # print(f"Cleaned up document {test_document_id}")
    #     except Exception as e:
    #         print(f"Test failed: {e}")
    # else:
    #     print(f"\nSkipping PDF processing test. Please create a dummy PDF at {dummy_pdf_path} to run this test.")