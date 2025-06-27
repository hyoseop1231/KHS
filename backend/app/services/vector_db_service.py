# backend/app/services/vector_db_service.py
# This file is being refactored to remove ChromaDB dependencies.
# Function signatures are kept for compatibility with existing tests.
# These functions will be reimplemented or replaced by FAISS/TinyDB logic elsewhere.

import os
from typing import List, Dict, Any, Optional # Optional 추가
# from app.config import settings # No longer using ChromaDB settings
from app.utils.logging_config import get_logger # 로거는 유지 가능
# from app.utils.exceptions import VectorDBError # 예외 타입은 유지하거나 일반 예외 사용

logger = get_logger(__name__)

logger.info("Vector DB service (vector_db_service.py) is now a placeholder for FAISS/TinyDB transition.")
logger.warning("All functions in this file are deprecated and will be removed or reimplemented.")

def store_vectors(document_id: str, text_chunks: List[str], vectors: List[List[float]], metadatas: List[Dict[str, Any]] = None):
    """
    [DEPRECATED] This function is no longer used. Vector storage is handled by Celery tasks
    using FAISS and TinyDB.
    """
    logger.warning("DEPRECATED: store_vectors called. This should not happen in the new FAISS/TinyDB workflow.")
    return

def search_similar_vectors(query_vector: List[float], top_k: int = 5, filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    [DEPRECATED] This function is no longer used. Vector search will be reimplemented
    using FAISS and TinyDB (likely in app.rag or a new search service).
    """
    logger.warning("DEPRECATED: search_similar_vectors called. Reimplement for FAISS/TinyDB.")
    return []

def get_all_documents() -> List[Dict[str, Any]]:
    """
    [DEPRECATED] Kept for test compatibility. Document listing will be reimplemented
    based on TinyDB metadata.
    """
    logger.warning("DEPRECATED: get_all_documents called. Reimplement based on TinyDB.")
    return []

def delete_document(document_id: str) -> bool:
    """
    [DEPRECATED] Kept for test compatibility. Document deletion will be reimplemented
    for FAISS/TinyDB.
    """
    logger.warning(f"DEPRECATED: delete_document('{document_id}') called. Reimplement for FAISS/TinyDB.")
    return True

def delete_all_documents() -> int:
    """
    [DEPRECATED] Kept for test compatibility. Deletion of all documents will be
    reimplemented for FAISS/TinyDB.
    """
    logger.warning("DEPRECATED: delete_all_documents called. Reimplement for FAISS/TinyDB.")
    return 0

def get_document_info(document_id: str) -> Optional[Dict[str, Any]]:
    """
    [DEPRECATED] Kept for test compatibility. Document info retrieval will be
    reimplemented based on TinyDB.
    """
    logger.warning(f"DEPRECATED: get_document_info('{document_id}') called. Reimplement based on TinyDB.")
    # 테스트에서 특정 구조를 기대할 수 있으므로, 간단한 dict 반환 또는 None
    # return {"document_id": document_id, "chunk_count": 0, "message": "Deprecated function"}
    return None


if __name__ == '__main__':
    logger.info("Vector DB service module (vector_db_service.py) - DEPRECATED for FAISS/TinyDB.")
    logger.info("This module is kept temporarily for test compatibility.")
