"""
Multimodal Vector Database Service
Handles storage and retrieval of text, images, and tables from PDF documents.
"""

import chromadb
import os
import json
import base64
from typing import List, Dict, Any, Optional
from app.config import settings
from app.utils.logging_config import get_logger
from app.utils.exceptions import VectorDBError

logger = get_logger(__name__)

# ChromaDB 클라이언트 초기화
try:
    client = chromadb.PersistentClient(path=settings.CHROMA_DATA_PATH)
    logger.info(f"ChromaDB client initialized at: {settings.CHROMA_DATA_PATH}")
except Exception as e:
    logger.error(f"Error initializing ChromaDB PersistentClient at '{settings.CHROMA_DATA_PATH}': {e}")
    client = None

# 컬렉션들 가져오기 또는 생성
try:
    if client:
        # Text collection (기존)
        text_collection = client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
        )
        
        # Images collection (temporarily disabled)
        # images_collection = client.get_or_create_collection(
        #     name=f"{settings.COLLECTION_NAME}_images",
        # )
        images_collection = None
        
        # Tables collection (새로 추가)
        tables_collection = client.get_or_create_collection(
            name=f"{settings.COLLECTION_NAME}_tables",
        )
        
        logger.info(f"ChromaDB collections loaded/created successfully")
        logger.info(f"  - Text: '{settings.COLLECTION_NAME}'")
        logger.info(f"  - Images: '{settings.COLLECTION_NAME}_images'")
        logger.info(f"  - Tables: '{settings.COLLECTION_NAME}_tables'")
    else:
        text_collection = None
        images_collection = None
        tables_collection = None
        logger.error("ChromaDB client is not available. Collections cannot be loaded/created.")
except Exception as e:
    text_collection = None
    images_collection = None
    tables_collection = None
    logger.error(f"Error getting or creating ChromaDB collections: {e}")

# Backward compatibility
collection = text_collection

def store_multimodal_content(document_id: str, content_data: Dict[str, Any], text_vectors: List[List[float]], text_metadatas: List[Dict[str, Any]] = None):
    """
    Stores multimodal content (text, images, tables) in respective ChromaDB collections.
    
    Args:
        document_id (str): A unique identifier for the source document.
        content_data (Dict[str, Any]): Dictionary containing 'text', 'images', 'tables' data.
        text_vectors (List[List[float]]): Vector embeddings for text chunks.
        text_metadatas (List[Dict[str, Any]], optional): Metadata for text chunks.
    """
    if not all([text_collection, tables_collection]):
        logger.error("ChromaDB collections are not available. Cannot store content.")
        raise VectorDBError("ChromaDB collections not available", "COLLECTIONS_UNAVAILABLE")
    
    try:
        # Store text content (기존 방식 + 개선)
        if content_data.get('text') and text_vectors:
            # Simple chunking - could be improved with better chunking strategy
            text_content = content_data['text']
            text_chunks = [chunk.strip() for chunk in text_content.split('\n\n') if chunk.strip()]
            
            if len(text_chunks) != len(text_vectors):
                logger.warning(f"Text chunks ({len(text_chunks)}) and vectors ({len(text_vectors)}) count mismatch")
                # Adjust to match - take minimum
                min_len = min(len(text_chunks), len(text_vectors))
                text_chunks = text_chunks[:min_len]
                text_vectors = text_vectors[:min_len]
            
            if text_chunks and text_vectors:
                store_text_vectors(document_id, text_chunks, text_vectors, text_metadatas)
        
        # Store images (temporarily disabled)
        # if content_data.get('images'):
        #     store_images(document_id, content_data['images'])
        
        # Store tables
        if content_data.get('tables'):
            store_tables(document_id, content_data['tables'])
            
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

def search_multimodal_content(query_vector: List[float], top_k: int = 5, filter_metadata: Dict[str, Any] = None, include_images: bool = True, include_tables: bool = True) -> Dict[str, List[Dict[str, Any]]]:
    """
    Searches across all content types (text, images, tables) for relevant information.
    
    Returns:
        Dict containing 'text', 'images', 'tables' results
    """
    results = {
        'text': [],
        'images': [],
        'tables': []
    }
    
    try:
        # Search text content (vector similarity)
        if text_collection and query_vector:
            text_results = search_text_vectors(query_vector, top_k, filter_metadata)
            results['text'] = text_results
        
        # Search images (metadata/description search)
        # Image search temporarily disabled
        # if include_images and images_collection:
        #     image_results = search_images(filter_metadata, top_k)
        #     results['images'] = image_results
        
        # Search tables (content search)
        if include_tables and tables_collection:
            table_results = search_tables(filter_metadata, top_k)
            results['tables'] = table_results
            
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
            where=where_clause
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
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
            limit=top_k
        )
        
        # Format results
        formatted_results = []
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
            limit=top_k
        )
        
        # Format results
        formatted_results = []
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

# Legacy functions for backward compatibility
def store_vectors(document_id: str, text_chunks: List[str], vectors: List[List[float]], metadatas: List[Dict[str, Any]] = None):
    """
    Legacy function for backward compatibility.
    """
    return store_text_vectors(document_id, text_chunks, vectors, metadatas)

def search_similar_vectors(query_vector: List[float], top_k: int = 5, filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Legacy function for backward compatibility.
    """
    return search_text_vectors(query_vector, top_k, filter_metadata)

# Additional utility functions
def delete_multimodal_document(document_id: str) -> bool:
    """
    Deletes all content (text, images, tables) for a specific document.
    """
    deleted = False
    
    try:
        # Delete from text collection
        if text_collection:
            text_results = text_collection.get(where={"source_document_id": document_id})
            if text_results['ids']:
                text_collection.delete(ids=text_results['ids'])
                logger.info(f"Deleted {len(text_results['ids'])} text chunks for document {document_id}")
                deleted = True
        
        # Delete from images collection
        # Image deletion temporarily disabled
        # if images_collection:
        #     image_results = images_collection.get(where={"source_document_id": document_id})
        #     if image_results['ids']:
        #         images_collection.delete(ids=image_results['ids'])
        #         logger.info(f"Deleted {len(image_results['ids'])} images for document {document_id}")
        #         deleted = True
        
        # Delete from tables collection
        if tables_collection:
            table_results = tables_collection.get(where={"source_document_id": document_id})
            if table_results['ids']:
                tables_collection.delete(ids=table_results['ids'])
                logger.info(f"Deleted {len(table_results['ids'])} tables for document {document_id}")
                deleted = True
                
    except Exception as e:
        logger.error(f"Error deleting multimodal document {document_id}: {e}")
        raise VectorDBError(f"Failed to delete document: {e}", "DELETE_ERROR")
    
    return deleted

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
            text_results = text_collection.get(where={"source_document_id": document_id})
            info['text_chunks'] = len(text_results['ids'])
            if text_results['documents']:
                info['first_chunk_preview'] = text_results['documents'][0][:200]
        
        # Get images info
        if images_collection:
            image_results = images_collection.get(where={"source_document_id": document_id})
            info['images'] = len(image_results['ids'])
        
        # Get tables info
        if tables_collection:
            table_results = tables_collection.get(where={"source_document_id": document_id})
            info['tables'] = len(table_results['ids'])
        
        return info if any([info['text_chunks'], info['images'], info['tables']]) else None
        
    except Exception as e:
        logger.error(f"Error getting document info for {document_id}: {e}")
        return None