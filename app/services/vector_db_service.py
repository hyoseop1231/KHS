import chromadb
import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from app.config import settings
from app.utils.logging_config import get_logger
from app.utils.exceptions import VectorDBError

logger = get_logger(__name__)

# ChromaDB 클라이언트 초기화
# ChromaDB 클라이언트 초기화
try:
    client = chromadb.PersistentClient(path=settings.CHROMA_DATA_PATH)
    logger.info(f"ChromaDB client initialized at: {settings.CHROMA_DATA_PATH}")
except Exception as e:
    error_msg = f"Error initializing ChromaDB PersistentClient at '{settings.CHROMA_DATA_PATH}': {e}"
    logger.error(error_msg)
    raise VectorDBError(error_msg, "DB_INIT_ERROR") # Raise error immediately

# 컬렉션들 가져오기 또는 생성
try:
    # Text collection
    text_collection = client.get_or_create_collection(
        name=settings.COLLECTION_NAME,
    )
    
    # Images collection
    images_collection = client.get_or_create_collection(
        name=f"{settings.COLLECTION_NAME}_images",
    )
    
    # Tables collection
    tables_collection = client.get_or_create_collection(
        name=f"{settings.COLLECTION_NAME}_tables",
    )
    
    logger.info(f"ChromaDB collections loaded/created successfully")
    logger.info(f"  - Text: '{settings.COLLECTION_NAME}'")
    logger.info(f"  - Images: '{settings.COLLECTION_NAME}_images'")
    logger.info(f"  - Tables: '{settings.COLLECTION_NAME}_tables'")
except Exception as e:
    error_msg = f"Error getting or creating ChromaDB collections: {e}"
    logger.error(error_msg)
    raise VectorDBError(error_msg, "COLLECTION_INIT_ERROR") # Raise error immediately

# Backward compatibility (for existing code that uses 'collection')
collection = text_collection

def store_multimodal_content(document_id: str, content_data: Dict[str, Any], text_vectors: Optional[List[List[float]]] = None, text_metadatas: Optional[List[Dict[str, Any]]] = None):
    """
    Stores all types of content (text, images, tables) into their respective collections.
    Args:
        document_id (str): A unique identifier for the source document.
        content_data (Dict[str, Any]): Dictionary containing 'text_chunks', 'images', 'tables'.
        text_vectors (List[List[float]], optional): Vector embeddings for text chunks.
        text_metadatas (List[Dict[str, Any]], optional): Metadata for text chunks.
    """
    try:
        # 1. Store text content
        if text_collection and text_vectors:
            text_chunks = content_data.get('text_chunks', [])
            if text_chunks:
                # Ensure all lists have the same length
                min_len = min(len(text_chunks), len(text_vectors), len(text_metadatas) if text_metadatas else len(text_chunks))
                
                valid_chunks = text_chunks[:min_len]
                valid_vectors = text_vectors[:min_len]
                valid_metadatas = text_metadatas[:min_len] if text_metadatas else None
                
                if valid_chunks and valid_vectors:
                    store_text_vectors(document_id, valid_chunks, valid_vectors, valid_metadatas)
                    logger.info(f"Stored {len(valid_chunks)} text chunks for document: {document_id}")

        # 2. Store images
        if images_collection:
            images_data = content_data.get('images', [])
            if images_data:
                store_images(document_id, images_data)
                logger.info(f"Stored {len(images_data)} images for document: {document_id}")

        # 3. Store tables
        if tables_collection:
            tables_data = content_data.get('tables', [])
            if tables_data:
                store_tables(document_id, tables_data)
                logger.info(f"Stored {len(tables_data)} tables for document: {document_id}")

        logger.info(f"Successfully stored multimodal content for document: {document_id}")

    except Exception as e:
        logger.error(f"Error storing multimodal content for {document_id}: {e}")
        raise VectorDBError(f"Failed to store multimodal content: {e}", "STORE_ERROR")

def store_text_vectors(document_id: str, text_chunks: List[str], vectors: List[List[float]], metadatas: List[Dict[str, Any]] = None):
    """
    Stores text chunks and their vectors in the text collection.
    """
    if not text_collection:
        logger.error("Text collection is not available. Cannot store vectors.")
        raise VectorDBError("Text collection not available", "COLLECTION_UNAVAILABLE")

    if not text_chunks or not vectors:
        logger.error("Text chunks or vectors are empty. Nothing to store.")
        raise VectorDBError("Empty text chunks or vectors", "EMPTY_DATA")

    if len(text_chunks) != len(vectors):
        logger.error(f"Mismatch between text chunks ({len(text_chunks)}) and vectors ({len(vectors)}) count.")
        raise VectorDBError("Text chunks and vectors count mismatch", "SIZE_MISMATCH")

    if metadatas and len(metadatas) != len(text_chunks):
        logger.error(f"Mismatch between text chunks ({len(text_chunks)}) and metadatas ({len(metadatas)}) count.")
        raise VectorDBError("Text chunks and metadatas count mismatch", "METADATA_SIZE_MISMATCH")

    # If no metadatas provided, create basic ones
    if not metadatas:
        metadatas = [
            {
                'source_document_id': document_id,
                'chunk_index': i,
                'content_type': 'text',
                'original_text_preview': chunk[:200]
            }
            for i, chunk in enumerate(text_chunks)
        ]
    else:
        # Ensure content_type is set
        for meta in metadatas:
            meta['content_type'] = 'text'

    # Generate unique IDs for each chunk
    ids = [f"{document_id}_text_chunk_{i}" for i in range(len(text_chunks))]

    try:
        text_collection.add(
            embeddings=vectors,
            documents=text_chunks,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Successfully stored {len(text_chunks)} text chunks for document '{document_id}'")
        
    except Exception as e:
        logger.error(f"Error storing vectors in ChromaDB for document '{document_id}': {e}")
        raise VectorDBError(f"Failed to store vectors: {e}", "CHROMADB_STORE_ERROR")

def store_images(document_id: str, images_data: List[Dict[str, Any]]):
    """
    Stores image metadata and descriptions in the images collection.
    """
    if not images_collection:
        logger.error("Images collection is not available.")
        raise VectorDBError("Images collection not available", "COLLECTION_UNAVAILABLE")
    
    if not images_data:
        logger.info(f"No images to store for document: {document_id}")
        return
    
    try:
        # Prepare data for storage
        ids = []
        documents = []  # Image descriptions
        metadatas = []
        
        for i, img_data in enumerate(images_data):
            img_id = f"{document_id}_image_{i}"
            img_description = img_data.get('description', f"Image from page {img_data.get('page', 'unknown')}")
            
            metadata = {
                'source_document_id': document_id,
                'content_type': 'image',
                'filename': img_data.get('filename', ''),
                'page': img_data.get('page', 0),
                'index': img_data.get('index', i),
                'width': img_data.get('width', 0),
                'height': img_data.get('height', 0),
                'size_bytes': img_data.get('size_bytes', 0),
                'file_path': img_data.get('path', '')
            }
            
            ids.append(img_id)
            documents.append(img_description)
            metadatas.append(metadata)
        
        # Store in ChromaDB (without embeddings for now - could add image embeddings later)
        images_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Successfully stored {len(images_data)} images for document '{document_id}'")
        
    except Exception as e:
        logger.error(f"Error storing images for document '{document_id}': {e}")
        raise VectorDBError(f"Failed to store images: {e}", "IMAGES_STORE_ERROR")

def store_tables(document_id: str, tables_data: List[Dict[str, Any]]):
    """
    Stores table metadata and content in the tables collection.
    """
    if not tables_collection:
        logger.error("Tables collection is not available.")
        raise VectorDBError("Tables collection not available", "COLLECTION_UNAVAILABLE")
    
    if not tables_data:
        logger.info(f"No tables to store for document: {document_id}")
        return
    
    try:
        # Prepare data for storage
        ids = []
        documents = []  # Table content as text
        metadatas = []
        
        for i, table_data in enumerate(tables_data):
            table_id = f"{document_id}_table_{i}"
            
            # Convert table data to searchable text
            table_text = table_data.get('raw_text', '')
            parsed_data = table_data.get('parsed_data', [])
            
            # Create a structured text representation
            if parsed_data:
                structured_text = []
                for row in parsed_data:
                    if isinstance(row, list):
                        structured_text.append(' | '.join(str(cell) for cell in row))
                table_content = '\n'.join(structured_text)
            else:
                table_content = table_text
            
            metadata = {
                'source_document_id': document_id,
                'content_type': 'table',
                'filename': table_data.get('filename', ''),
                'page': table_data.get('page', 0),
                'index': table_data.get('index', i),
                'x': table_data.get('x', 0),
                'y': table_data.get('y', 0),
                'width': table_data.get('width', 0),
                'height': table_data.get('height', 0),
                'size_bytes': table_data.get('size_bytes', 0),
                'file_path': table_data.get('path', ''),
                'raw_text': table_text,
                'parsed_data': json.dumps(parsed_data) if parsed_data else ''
            }
            
            ids.append(table_id)
            documents.append(table_content or f"Table from page {table_data.get('page', 'unknown')}")
            metadatas.append(metadata)
        
        # Store in ChromaDB
        tables_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Successfully stored {len(tables_data)} tables for document '{document_id}'")
        
    except Exception as e:
        logger.error(f"Error storing tables for document '{document_id}': {e}")
        raise VectorDBError(f"Failed to store tables: {e}", "TABLES_STORE_ERROR")

def search_multimodal_content(
    query_vector: List[float],
    top_k: int = 5,
    doc_ids: Optional[List[str]] = None,
    filter_metadata: Optional[Dict[str, Any]] = None,
    include_images: bool = True,
    include_tables: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Searches across content types (text, images, tables) for relevant information.

    Args:
        query_vector (List[float]): The vector representation of the user's query.
        top_k (int, optional): Number of results to return per content type. Defaults to 5.
        doc_ids (Optional[List[str]]): List of document IDs to filter by. If set and filter_metadata is None,
            filters by these IDs. Ignored if filter_metadata is provided.
        filter_metadata (Optional[Dict[str, Any]]): Explicit metadata filter dict. Takes precedence over doc_ids.
        include_images (bool): Whether to include image search results. Defaults to True.
        include_tables (bool): Whether to include table search results. Defaults to True.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Keys 'text', 'images', 'tables' mapping to respective results.
    """
    results = {'text': [], 'images': [], 'tables': []}
    # Determine metadata filter: explicit filter_metadata wins, else doc_ids filter
    meta_filter = filter_metadata if filter_metadata is not None else (
        {"source_document_id": {"$in": doc_ids}} if doc_ids else None
    )

    try:
        # 병렬 검색을 위한 ThreadPoolExecutor 사용
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            
            # Text search (vector similarity) - 비동기 실행
            if text_collection and query_vector:
                futures['text'] = executor.submit(search_text_vectors, query_vector, top_k, meta_filter)
            
            # Image search (metadata-only) - 비동기 실행
            if include_images and images_collection:
                futures['images'] = executor.submit(search_images, meta_filter, top_k)
            
            # Table search (metadata-only) - 비동기 실행
            if include_tables and tables_collection:
                futures['tables'] = executor.submit(search_tables, meta_filter, top_k)
            
            # 결과 수집
            for search_type, future in futures.items():
                try:
                    results[search_type] = future.result(timeout=30)  # 30초 타임아웃
                except Exception as e:
                    logger.warning(f"Error in {search_type} search: {e}")
                    results[search_type] = []
                    
    except Exception as e:
        logger.error(f"Error in multimodal search: {e}")
        raise VectorDBError(f"Multimodal search failed: {e}", "SEARCH_ERROR")
    return results

def search_text_vectors(query_vector: List[float], top_k: int = 5, filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Searches for text chunks with vectors similar to the query_vector.
    """
    if not text_collection:
        logger.error("Text collection is not available. Cannot search vectors.")
        raise VectorDBError("Text collection not available", "COLLECTION_UNAVAILABLE")

    if not query_vector:
        logger.error("Query vector is empty or None.")
        raise VectorDBError("Query vector is empty", "EMPTY_QUERY_VECTOR")

    try:
        # Perform similarity search
        where_clause = filter_metadata if filter_metadata else None
        
        results = text_collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_clause,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        if results and results.get('ids') and results.get('ids')[0]:
            ids = results['ids'][0]
            documents = results['documents'][0] if results.get('documents') else [None] * len(ids)
            metadatas = results['metadatas'][0] if results.get('metadatas') else [None] * len(ids)
            distances = results['distances'][0] if results.get('distances') else [None] * len(ids)

            for i in range(len(ids)):
                formatted_results.append({
                    "id": ids[i],
                    "text": documents[i],
                    "metadata": metadatas[i],
                    "distance": distances[i] # 거리가 짧을수록 유사함
                })
        
        logger.info(f"Found {len(formatted_results)} similar text chunks")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error searching similar vectors: {e}")
        raise VectorDBError(f"Vector search failed: {e}", "SEARCH_ERROR")

def search_images(filter_metadata: Dict[str, Any] = None, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Searches for relevant images based on metadata filters.
    """
    if not images_collection:
        logger.warning("Images collection is not available.")
        return []
    
    try:
        where_clause = filter_metadata if filter_metadata else None
        
        results = images_collection.get(
            where=where_clause,
            limit=top_k,
            include=['documents', 'metadatas']
        )
        
        # Format results
        formatted_results = []
        if results and results.get('ids'):
            for i in range(len(results['ids'])):
                formatted_results.append({
                    'id': results['ids'][i],
                    'description': results['documents'][i],
                    'metadata': results['metadatas'][i]
                })
        
        logger.info(f"Found {len(formatted_results)} relevant images")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error searching images: {e}")
        return []

def search_tables(filter_metadata: Dict[str, Any] = None, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Searches for relevant tables based on metadata filters.
    """
    if not tables_collection:
        logger.warning("Tables collection is not available.")
        return []
    
    try:
        where_clause = filter_metadata if filter_metadata else None
        
        results = tables_collection.get(
            where=where_clause,
            limit=top_k,
            include=['documents', 'metadatas']
        )
        
        # Format results
        formatted_results = []
        if results and results.get('ids'):
            for i in range(len(results['ids'])):
                metadata = results['metadatas'][i]
                parsed_data = json.loads(metadata.get('parsed_data', '[]')) if metadata.get('parsed_data') else []
                
                formatted_results.append({
                    'id': results['ids'][i],
                    'content': results['documents'][i],
                    'metadata': metadata,
                    'parsed_data': parsed_data
                })
        
        logger.info(f"Found {len(formatted_results)} relevant tables")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error searching tables: {e}")
        return []

def get_all_documents() -> List[Dict[str, Any]]:
    """
    ChromaDB에 저장된 모든 문서(document_id, 파일명 등) 목록을 반환합니다.
    각 문서는 source_document_id 기준으로 그룹화되며, 미리보기 텍스트와 청크 개수 등도 포함할 수 있습니다.
    """
    if not text_collection:
        logger.error("Text collection is not available. Cannot get documents.")
        raise VectorDBError("Text collection not available", "COLLECTION_UNAVAILABLE")
    
    try:
        # 모든 메타데이터만 쿼리 (최대 10000개 제한)
        results = text_collection.get(include=["metadatas"], limit=10000)
        metadatas = results.get("metadatas", [])
        # source_document_id별로 그룹화
        doc_map = {}
        for meta in metadatas:
            doc_id = meta.get("source_document_id")
            if not doc_id:
                continue
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "document_id": doc_id,
                    "chunk_count": 0,
                    "first_chunk_preview": meta.get("original_text_preview", "")
                }
            doc_map[doc_id]["chunk_count"] += 1
        logger.info(f"Retrieved {len(doc_map)} unique documents from ChromaDB")
        return list(doc_map.values())
    except Exception as e:
        logger.error(f"Error getting all documents from ChromaDB: {e}")
        raise VectorDBError(f"Failed to get documents: {e}", "GET_DOCUMENTS_ERROR")

def delete_document(document_id: str) -> bool:
    """
    특정 문서 ID에 해당하는 모든 벡터와 청크를 삭제합니다.
    
    Args:
        document_id (str): 삭제할 문서의 ID
        
    Returns:
        bool: 삭제 성공 여부
    """
    return delete_multimodal_document(document_id)

def delete_all_documents() -> int:
    """
    모든 문서와 벡터를 삭제합니다.
    
    Returns:
        int: 삭제된 문서의 개수
    """
    return delete_all_multimodal_documents()

def get_document_info(document_id: str) -> dict:
    """
    특정 문서의 상세 정보를 반환합니다.
    
    Args:
        document_id (str): 조회할 문서의 ID
        
    Returns:
        dict: 문서 정보 (chunk_count, first_chunk_preview 등)
    """
    return get_multimodal_document_info(document_id)


# Additional utility functions for multimodal content
def delete_multimodal_document(document_id: str) -> bool:
    """
    Deletes all content (text, images, tables) for a specific document.
    """
    deleted = False
    
    try:
        # Delete from text collection
        if text_collection:
            text_results = text_collection.get(where={"$and": [{"source_document_id": document_id}, {"content_type": "text"}]})
            if text_results['ids']:
                text_collection.delete(ids=text_results['ids'])
                logger.info(f"Deleted {len(text_results['ids'])} text chunks for document {document_id}")
                deleted = True
        
        # Delete from images collection
        if images_collection:
            image_results = images_collection.get(where={"$and": [{"source_document_id": document_id}, {"content_type": "image"}]})
            if image_results['ids']:
                images_collection.delete(ids=image_results['ids'])
                logger.info(f"Deleted {len(image_results['ids'])} images for document {document_id}")
                deleted = True
        
        # Delete from tables collection
        if tables_collection:
            table_results = tables_collection.get(where={"$and": [{"source_document_id": document_id}, {"content_type": "table"}]})
            if table_results['ids']:
                tables_collection.delete(ids=table_results['ids'])
                logger.info(f"Deleted {len(table_results['ids'])} tables for document {document_id}")
                deleted = True
                
    except Exception as e:
        logger.error(f"Error deleting multimodal document {document_id}: {e}")
        raise VectorDBError(f"Failed to delete document: {e}", "DELETE_ERROR")
    
    return deleted

def delete_all_multimodal_documents() -> int:
    """
    Deletes all content (text, images, tables) from all multimodal collections.
    
    Returns:
        int: Number of unique documents deleted
    """
    deleted_documents = set()
    
    try:
        # Delete from text collection
        if text_collection:
            text_results = text_collection.get(include=["metadatas"])
            if text_results['ids']:
                text_collection.delete(ids=text_results['ids'])
                # Extract unique document IDs
                metadatas = text_results.get("metadatas", [])
                if metadatas:
                    for meta in metadatas:
                        if meta and "source_document_id" in meta:
                            deleted_documents.add(meta["source_document_id"])
                logger.info(f"Deleted {len(text_results['ids'])} text chunks from multimodal collection")
        
        # Delete from images collection  
        if images_collection:
            image_results = images_collection.get(include=["metadatas"])
            if image_results['ids']:
                images_collection.delete(ids=image_results['ids'])
                # Extract unique document IDs
                metadatas = image_results.get("metadatas", [])
                if metadatas:
                    for meta in metadatas:
                        if meta and "source_document_id" in meta:
                            deleted_documents.add(meta["source_document_id"])
                logger.info(f"Deleted {len(image_results['ids'])} images from multimodal collection")
        
        # Delete from tables collection
        if tables_collection:
            table_results = tables_collection.get(include=["metadatas"])
            if table_results['ids']:
                tables_collection.delete(ids=table_results['ids'])
                # Extract unique document IDs
                metadatas = table_results.get("metadatas", [])
                if metadatas:
                    for meta in metadatas:
                        if meta and "source_document_id" in meta:
                            deleted_documents.add(meta["source_document_id"])
                logger.info(f"Deleted {len(table_results['ids'])} tables from multimodal collection")
        
        deleted_count = len(deleted_documents)
        logger.info(f"Deleted all multimodal content. Total unique documents: {deleted_count}")
        return deleted_count
        
    except Exception as e:
        logger.error(f"Error deleting all multimodal documents: {e}")
        raise VectorDBError(f"Failed to delete all multimodal documents: {e}", "DELETE_ALL_ERROR")

def get_multimodal_document_info(document_id: str) -> Optional[Dict[str, Any]]:
    """
    Gets information about all content types for a specific document.
    """
    try:
        info = {
            'document_id': document_id,
            'text_chunks': 0,
            'images': 0,
            'tables': 0,
            'first_chunk_preview': None
        }
        
        # Get text info
        if text_collection:
            text_results = text_collection.get(where={"$and": [{"source_document_id": document_id}, {"content_type": "text"}]})
            info['text_chunks'] = len(text_results['ids'])
            if text_results['documents']:
                info['first_chunk_preview'] = text_results['documents'][0][:200]
        
        # Get images info
        if images_collection:
            image_results = images_collection.get(where={"$and": [{"source_document_id": document_id}, {"content_type": "image"}]})
            info['images'] = len(image_results['ids'])
        
        # Get tables info
        if tables_collection:
            table_results = tables_collection.get(where={"$and": [{"source_document_id": document_id}, {"content_type": "table"}]})
            info['tables'] = len(table_results['ids'])
        
        return info if any([info['text_chunks'], info['images'], info['tables']]) else None
        
    except Exception as e:
        logger.error(f"Error getting document info for {document_id}: {e}")
        return None

if __name__ == '__main__':
    # 간단한 테스트용
    print(f"Vector DB service module loaded. Chroma client: {'Initialized' if client else 'Failed'}. Collection: {'Initialized' if collection else 'Failed'}")

    if client and collection:
        print("\n--- Testing ChromaDB Operations (Example) ---")

        # 테스트용 데이터
        test_doc_id = "test_document.pdf"
        test_chunks = [
            "이것은 첫 번째 테스트 청크입니다. ChromaDB 저장 테스트.",
            "두 번째 청크는 약간 다른 내용을 담고 있습니다.",
            "세 번째 청크는 검색 테스트를 위해 고유한 키워드를 포함합니다: '코코넛'."
        ]
        # 실제 임베딩 대신 더미 임베딩 사용 (차원 수는 실제 모델과 맞추는 것이 좋음, 예: 384)
        # SentenceTransformer 모델 로드가 필요하므로, 여기서는 간단히 처리
        # from app.services.text_processing_service import get_embeddings as get_real_embeddings
        # test_vectors = get_real_embeddings(test_chunks) # 실제로는 이렇게 해야 함

        # 임시 더미 벡터 (실제로는 text_processing_service.get_embeddings를 사용해야 함)
        # text_processing_service가 로드될 때 모델을 다운로드하므로, 의존성을 피하기 위해 여기서는 더미 사용
        dummy_embedding_dim = 384 # 사용하는 임베딩 모델의 차원 수와 일치시켜야 함
        test_vectors = [[float(i/100.0)] * dummy_embedding_dim for i in range(len(test_chunks))]


        if not all(test_vectors) or not all(len(v) == dummy_embedding_dim for v in test_vectors):
             print("Dummy vectors are not correctly generated. Skipping store/search test.")
        else:
            print(f"\nStoring {len(test_chunks)} test vectors for document '{test_doc_id}'...")
            store_text_vectors(test_doc_id, test_chunks, test_vectors)

            print("\nSearching for vectors similar to the first chunk's vector...")
            # 첫 번째 청크의 벡터로 검색 (자기 자신도 결과에 포함될 수 있음)
            query_vec = test_vectors[0]
            similar_results = search_text_vectors(query_vec, top_k=2)

            if similar_results:
                print("Search results:")
                for res in similar_results:
                    print(f"  ID: {res['id']}, Distance: {res['distance']:.4f}, Text: {res['text'][:50]}...")
            else:
                print("No similar results found or error during search.")

            print("\nSearching with a specific metadata filter (if applicable)...")
            # 예시: 특정 문서 ID로 필터링하여 검색
            # 이 테스트에서는 test_doc_id만 있으므로, 필터링 결과는 위와 유사할 것임
            filtered_results = search_text_vectors(query_vec, top_k=2, filter_metadata={'source_document_id': test_doc_id})
            if filtered_results:
                print("Filtered search results:")
                for res in filtered_results:
                     print(f"  ID: {res['id']}, Distance: {res['distance']:.4f}, Text: {res['text'][:50]}...")
            else:
                print("No results found with the specified filter.")

        # Test multimodal storage and search
        print("\n--- Testing Multimodal Storage and Search ---")
        multimodal_test_doc_id = "multimodal_test_doc.pdf"
        multimodal_content_data = {
            "text_chunks": [
                "멀티모달 테스트 문서의 첫 번째 텍스트 청크입니다.",
                "이것은 두 번째 텍스트 청크이며, 이미지와 표도 포함됩니다."
            ],
            "images": [
                {"filename": "image1.png", "path": "/tmp/image1.png", "page": 1, "description": "A sample image of a graph.", "metadata": {"source_document_id": multimodal_test_doc_id, "page": 1}}
            ],
            "tables": [
                {"raw_text": "Header1\tHeader2\nData1\tData2", "parsed_data": [["Header1", "Header2"], ["Data1", "Data2"]], "page": 2, "metadata": {"source_document_id": multimodal_test_doc_id, "page": 2}}
            ]
        }
        multimodal_text_vectors = [[0.1]*dummy_embedding_dim, [0.2]*dummy_embedding_dim]

        print(f"\nStoring multimodal content for document '{multimodal_test_doc_id}'...")
        store_multimodal_content(multimodal_test_doc_id, multimodal_content_data, multimodal_text_vectors)

        print("\nSearching multimodal content...")
        query_for_multimodal = [0.15]*dummy_embedding_dim # Query vector for multimodal search
        multimodal_search_results = search_multimodal_content(query_for_multimodal, top_k=1, doc_ids=[multimodal_test_doc_id])

        print("Multimodal Search Results:")
        print(f"  Text results: {len(multimodal_search_results['text'])}")
        for res in multimodal_search_results['text']:
            print(f"    Text ID: {res['id']}, Text: {res['text'][:50]}...")
        print(f"  Image results: {len(multimodal_search_results['images'])}")
        for res in multimodal_search_results['images']:
            print(f"    Image ID: {res['id']}, Description: {res['description'][:50]}...")
        print(f"  Table results: {len(multimodal_search_results['tables'])}")
        for res in multimodal_search_results['tables']:
            print(f"    Table ID: {res['id']}, Content: {res['content'][:50]}...")

        print("\nGetting multimodal document info...")
        info = get_multimodal_document_info(multimodal_test_doc_id)
        print(f"Multimodal document info: {info}")

        print("\nDeleting multimodal document...")
        delete_multimodal_document(multimodal_test_doc_id)
        info_after_delete = get_multimodal_document_info(multimodal_test_doc_id)
        print(f"Multimodal document info after delete: {info_after_delete}")

        print("\nDeleting all documents...")
        deleted_count = delete_all_documents()
        print(f"Total documents deleted: {deleted_count}")

    else:
        print("\nSkipping ChromaDB operations test as client or collection failed to initialize.")

    print("\nNote: For real testing, ensure 'text_processing_service.embedding_model' is loaded and used for generating actual embeddings.")