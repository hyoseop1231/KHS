from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
import shutil
import os
import uuid # For unique document IDs
import requests

# 서비스 모듈 임포트
from app.services.ocr_service import extract_multimodal_content_from_pdf, extract_text_from_pdf
from app.services.text_processing_service import split_text_into_chunks, get_embeddings
from app.services.vector_db_service import get_all_documents, delete_document, delete_all_documents, get_document_info
from app.services.multimodal_vector_db_service import store_multimodal_content, search_multimodal_content, delete_multimodal_document, get_multimodal_document_info
from app.services.llm_service import process_llm_chat_request
from app.services.multimodal_llm_service import process_multimodal_llm_chat_request, enhance_response_with_media_references
from app.config import settings
from app.utils.logging_config import get_logger
from app.utils.security import FileValidator, sanitize_input, validate_document_id
from app.utils.file_manager import DocumentFileManager
from app.utils.exceptions import ValidationError, FileProcessingError, OCRError, VectorDBError, EmbeddingError, LLMError

logger = get_logger(__name__)

# Pydantic 모델 (요청/응답 본문 정의)
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ChatRequest(BaseModel):
    query: str
    document_id: Optional[str] = None # (하위 호환)
    document_ids: Optional[List[str]] = None # 여러 문서 ID에 대해 질문할 경우
    model_name: Optional[str] = settings.OLLAMA_DEFAULT_MODEL
    lang: Optional[str] = "ko"

class ChatResponse(BaseModel):
    query: str
    response: str
    source_document_id: Optional[str] = None
    retrieved_chunks_preview: Optional[List[str]] = None # 디버깅/정보용
    content_summary: Optional[Dict[str, int]] = None # 멀티모달 컨텐츠 요약
    media_references: Optional[Dict[str, Any]] = None # 이미지/표 참조 정보

router = APIRouter()

# 업로드 디렉토리는 settings에서 관리

# PDF 처리 상태 저장 (간단한 인메모리 방식)
pdf_processing_status = {}

def process_pdf_background(file_path: str, document_id: str):
    """
    백그라운드에서 멀티모달 PDF 처리 작업을 수행합니다.
    OCR -> Image/Table Extraction -> Chunking -> Embedding -> Storage
    """
    logger.info(f"Background task started: Processing multimodal PDF {document_id} from {file_path}")
    try:
        pdf_processing_status[document_id] = {"step": "OCR", "message": "OCR 및 멀티모달 컨텐트 추출 진행 중..."}
        
        # 1. Multimodal Content Extraction (OCR + Images + Tables)
        logger.info(f"[Task {document_id}] Step 1: Extracting multimodal content...")
        try:
            content_data = extract_multimodal_content_from_pdf(file_path, document_id)
            
            extracted_text = content_data.get('text', '')
            images_data = content_data.get('images', [])
            tables_data = content_data.get('tables', [])
            
            if not extracted_text and not images_data and not tables_data:
                raise OCRError("No content extracted from PDF", "EMPTY_EXTRACTION")
                
            logger.info(f"[Task {document_id}] Extracted: text={len(extracted_text)} chars, images={len(images_data)}, tables={len(tables_data)}")
            
        except (OCRError, FileProcessingError) as e:
            pdf_processing_status[document_id] = {"step": "Error", "message": f"OCR 오류: {e.message}"}
            logger.error(f"[Task {document_id}] OCR error: {e}")
            return

        pdf_processing_status[document_id] = {"step": "Chunking", "message": "텍스트 청크 분할 중..."}
        
        # 2. Text Chunking
        logger.info(f"[Task {document_id}] Step 2: Splitting text into chunks...")
        text_chunks = []
        if extracted_text:
            try:
                text_chunks = split_text_into_chunks(extracted_text)
                if not text_chunks:
                    logger.warning(f"[Task {document_id}] No text chunks generated")
            except EmbeddingError as e:
                pdf_processing_status[document_id] = {"step": "Error", "message": f"청킹 오류: {e.message}"}
                logger.error(f"[Task {document_id}] Chunking error: {e}")
                return

        pdf_processing_status[document_id] = {"step": "Embedding", "message": "임베딩 생성 중..."}
        
        # 3. Embedding Generation (only for text)
        embeddings = []
        if text_chunks:
            logger.info(f"[Task {document_id}] Step 3: Generating embeddings for {len(text_chunks)} chunks...")
            try:
                embeddings = get_embeddings(text_chunks)
                if not embeddings or not all(e for e in embeddings):
                    logger.warning(f"[Task {document_id}] Some embeddings are empty or invalid")
                    embeddings = []
            except EmbeddingError as e:
                pdf_processing_status[document_id] = {"step": "Error", "message": f"임베딩 오류: {e.message}"}
                logger.error(f"[Task {document_id}] Embedding error: {e}")
                return
        
        # 유효한 데이터 준비
        valid_chunks = text_chunks
        valid_embeddings = embeddings

        pdf_processing_status[document_id] = {"step": "Storing", "message": "멀티모달 컨텐트 저장 중..."}
        
        # 4. Store Multimodal Content in Vector DB
        logger.info(f"[Task {document_id}] Step 4: Storing multimodal content in Vector DB...")
        logger.info(f"  - Text chunks: {len(valid_chunks)}")
        logger.info(f"  - Images: {len(images_data)}")
        logger.info(f"  - Tables: {len(tables_data)}")
        
        try:
            # Prepare metadata for text chunks
            text_metadatas = None
            if valid_chunks:
                text_metadatas = [
                    {
                        'source_document_id': document_id, 
                        'chunk_index': i,
                        'content_type': 'text'
                    } 
                    for i in range(len(valid_chunks))
                ]
            
            # Store all multimodal content
            store_multimodal_content(
                document_id=document_id,
                content_data=content_data,
                text_vectors=valid_embeddings,
                text_metadatas=text_metadatas
            )
            
        except VectorDBError as e:
            pdf_processing_status[document_id] = {"step": "Error", "message": f"저장 오류: {e.message}"}
            logger.error(f"[Task {document_id}] Vector DB error: {e}")
            return

        pdf_processing_status[document_id] = {
            "step": "Done", 
            "message": f"멀티모달 PDF 처리 완료! (텍스트: {len(valid_chunks)}청크, 이미지: {len(images_data)}개, 표: {len(tables_data)}개)"
        }
        logger.info(f"[Task {document_id}] Successfully processed and stored multimodal PDF: {document_id}")

    except Exception as e:
        pdf_processing_status[document_id] = {"step": "Error", "message": f"예외 발생: {str(e)}"}
        logger.error(f"[Task {document_id}] Unexpected error during background PDF processing: {e}", exc_info=True)
    finally:
        # 선택 사항: 처리 후 원본 PDF 파일 삭제 (필요하다면)
        # try:
        #     os.remove(file_path)
        #     print(f"[Task {document_id}] Cleaned up uploaded file: {file_path}")
        # except OSError as e:
        #     print(f"[Task {document_id}] Error deleting file {file_path}: {e}")
        pass


@router.post("/upload_pdf/")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Handles PDF file upload with comprehensive security validation.
    The file is saved and processing (OCR, chunking, embedding, storage) is offloaded to a background task.
    """
    logger.info(f"Upload request received for file: {file.filename}")
    
    # Basic validation
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")
    
    # Check file size
    file_size = 0
    content = await file.read()
    file_size = len(content)
    
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    
    if file_size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB"
        )
    
    # Generate secure document ID and filename
    document_id = f"{os.path.splitext(file.filename)[0]}_{str(uuid.uuid4())[:8]}"
    safe_filename = FileValidator.generate_safe_filename(file.filename, document_id)
    file_path = os.path.join(settings.UPLOAD_DIR, safe_filename)
    
    try:
        # Save file
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        logger.info(f"File saved: {file_path} ({file_size} bytes)")
        
        # Validate uploaded file
        validation_result = FileValidator.validate_uploaded_file(file_path, file.filename, file_size)
        
        if not validation_result["is_valid"]:
            # Remove invalid file
            os.remove(file_path)
            logger.warning(f"File validation failed: {validation_result['errors']}")
            raise HTTPException(
                status_code=400, 
                detail=f"File validation failed: {'; '.join(validation_result['errors'])}"
            )
        
        # Start background processing
        background_tasks.add_task(process_pdf_background, file_path, document_id)
        
        logger.info(f"Background processing started for document: {document_id}")
        
        return JSONResponse(
            content={
                "message": "File uploaded successfully. Processing started in the background.",
                "filename": file.filename,
                "document_id": document_id,
                "file_hash": validation_result["file_hash"],
                "detail": "The PDF is being processed. This may take some time depending on the file size and content."
            },
            status_code=202
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up file after error: {file_path}")
            except OSError as cleanup_error:
                logger.error(f"Error cleaning up file {file_path}: {cleanup_error}")
        
        logger.error(f"Unexpected error during file upload: {e}")
        raise HTTPException(status_code=500, detail="Could not save or start processing file")
    finally:
        file.file.close()


@router.post("/chat/", response_model=ChatResponse)
async def chat_with_llm(chat_request: ChatRequest):
    """
    Handles chat requests with input validation and security checks.
    1. Validates and sanitizes user input
    2. Embeds the user query
    3. Searches for relevant chunks in the Vector DB (optionally filtered by document_id)
    4. Constructs a RAG prompt and gets a response from the LLM
    """
    # Input validation and sanitization
    query = sanitize_input(chat_request.query, max_length=2000)
    if not query:
        raise HTTPException(status_code=400, detail="Query not provided or invalid.")
    
    # Validate document IDs
    document_ids = chat_request.document_ids or ([] if chat_request.document_id is None else [chat_request.document_id])
    if document_ids:
        for doc_id in document_ids:
            if not validate_document_id(doc_id):
                raise HTTPException(status_code=400, detail=f"Invalid document ID: {doc_id}")
    
    model_name = chat_request.model_name or settings.OLLAMA_DEFAULT_MODEL
    lang = chat_request.lang or "ko"

    logger.info(f"Chat request: Query length={len(query)}, DocIDs={len(document_ids)}, Model={model_name}, Lang={lang}")
    logger.debug(f"Query preview: {query[:100]}...")

    try:
        # 1. Embed user query
        logger.info("Step 1: Embedding user query...")
        query_embedding_list = get_embeddings([query])
        if not query_embedding_list or not query_embedding_list[0]:
            raise EmbeddingError("Could not generate embedding for the query", "QUERY_EMBEDDING_FAILED")
        query_embedding = query_embedding_list[0]

        # 2. Search Multimodal Content
        logger.info("Step 2: Searching multimodal content for relevant information...")
        filter_metadata = None
        if document_ids and len(document_ids) > 0:
            if len(document_ids) == 1:
                filter_metadata = {"source_document_id": document_ids[0]}
            else:
                filter_metadata = {"source_document_id": {"$in": document_ids}}
            logger.debug(f"Applying filter: {filter_metadata}")
        
        # Search across all content types
        multimodal_results = search_multimodal_content(
            query_vector=query_embedding, 
            top_k=5, 
            filter_metadata=filter_metadata,
            include_images=True,
            include_tables=True
        )
        
        # Extract text results for backward compatibility
        retrieved_chunks = multimodal_results.get('text', [])
        retrieved_images = multimodal_results.get('images', [])
        retrieved_tables = multimodal_results.get('tables', [])

        if not any([retrieved_chunks, retrieved_images, retrieved_tables]):
            logger.warning("No relevant content found in Vector DB")

        retrieved_chunk_texts_preview = [chunk.get('text', '')[:100] + "..." for chunk in retrieved_chunks]

        # 3. Get LLM response via Multimodal RAG
        logger.info("Step 3: Getting LLM response using multimodal RAG...")
        
        # Combine all retrieved content for LLM processing
        all_retrieved_content = {
            'text_chunks': retrieved_chunks,
            'images': retrieved_images,
            'tables': retrieved_tables
        }
        
        # Use multimodal LLM service
        llm_response_text = process_multimodal_llm_chat_request(
            user_query=query,
            multimodal_content=all_retrieved_content,
            model_name=model_name,
            lang=lang
        )
        
        # Enhance response with media references for UI display
        enhanced_response = enhance_response_with_media_references(
            llm_response_text,
            retrieved_images,
            retrieved_tables
        )
    
    except EmbeddingError as e:
        logger.error(f"Embedding error in chat: {e}")
        raise HTTPException(status_code=500, detail=f"임베딩 생성 오류: {e.message}")
    except VectorDBError as e:
        logger.error(f"Vector DB error in chat: {e}")
        raise HTTPException(status_code=500, detail=f"벡터 검색 오류: {e.message}")
    except LLMError as e:
        logger.error(f"LLM error in chat: {e}")
        raise HTTPException(status_code=500, detail=f"AI 모델 오류: {e.message}")
    except Exception as e:
        logger.error(f"Unexpected error in chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="채팅 처리 중 예상치 못한 오류가 발생했습니다.")

    # Enhanced response with multimodal content info
    response_data = {
        'query': query,
        'response': enhanced_response.get('text', llm_response_text),
        'source_document_id': document_ids[0] if document_ids else None,
        'retrieved_chunks_preview': retrieved_chunk_texts_preview,
        'content_summary': {
            'text_chunks': len(retrieved_chunks),
            'images': len(retrieved_images),
            'tables': len(retrieved_tables)
        },
        'media_references': {
            'images': enhanced_response.get('referenced_images', []),
            'tables': enhanced_response.get('referenced_tables', []),
            'has_media': enhanced_response.get('has_media', False)
        }
    }
    
    return ChatResponse(**response_data)

@router.get("/upload_status/{document_id}")
def get_upload_status(document_id: str):
    status = pdf_processing_status.get(document_id)
    if status:
        return status
    else:
        return {"step": "Unknown", "message": "해당 문서의 상태 정보를 찾을 수 없습니다."}

@router.get("/ollama/status")
def ollama_status():
    """
    Ollama 서버가 실행 중인지 확인합니다.
    """
    try:
        # Ollama 서버의 상태를 확인하기 위해 /api/tags 엔드포인트에 요청
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            return {"status": "running"}
        else:
            return {"status": "unreachable", "detail": response.text}
    except Exception as e:
        return {"status": "not running", "detail": str(e)}

@router.get("/ollama/models")
def ollama_models():
    """
    Ollama에 다운로드된 모델 목록을 반환합니다.
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            # Ollama의 /api/tags는 {"models": [{"name": ...}, ...]} 형태
            models = [m["name"] for m in data.get("models", [])]
            return {"models": models}
        else:
            return {"models": [], "detail": response.text}
    except Exception as e:
        return {"models": [], "detail": str(e)}

@router.get("/documents")
def list_documents():
    """
    DB에 저장된 모든 문서(document_id, chunk 개수, 미리보기 등) 목록을 반환합니다.
    """
    try:
        docs = get_all_documents()
        return {"documents": docs}
    except VectorDBError as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"문서 목록 조회 오류: {e.message}")

@router.delete("/documents/{document_id}")
def delete_document_by_id(document_id: str):
    """
    특정 문서 ID에 해당하는 문서를 DB와 파일 시스템에서 삭제합니다.
    """
    if not validate_document_id(document_id):
        raise HTTPException(status_code=400, detail=f"Invalid document ID: {document_id}")
    
    logger.info(f"Delete request for document: {document_id}")
    
    try:
        # 1. 벡터 DB에서 멀티모달 컨텐트 삭제
        db_deleted = delete_multimodal_document(document_id)
        
        # 2. 파일 시스템에서 삭제
        file_deleted = DocumentFileManager.delete_file_by_document_id(document_id)
        
        # 3. 처리 상태에서도 제거
        if document_id in pdf_processing_status:
            del pdf_processing_status[document_id]
        
        if db_deleted or file_deleted:
            logger.info(f"Successfully deleted document: {document_id} (DB: {db_deleted}, File: {file_deleted})")
            return {
                "message": f"Document {document_id} deleted successfully",
                "document_id": document_id,
                "deleted_from_db": db_deleted,
                "deleted_from_files": file_deleted
            }
        else:
            logger.warning(f"Document not found: {document_id}")
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
            
    except VectorDBError as e:
        logger.error(f"Vector DB error during delete: {e}")
        raise HTTPException(status_code=500, detail=f"DB 삭제 오류: {e.message}")
    except FileProcessingError as e:
        logger.error(f"File processing error during delete: {e}")
        raise HTTPException(status_code=500, detail=f"파일 삭제 오류: {e.message}")
    except Exception as e:
        logger.error(f"Unexpected error during delete: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="문서 삭제 중 예상치 못한 오류가 발생했습니다")

@router.delete("/documents")
def delete_all_documents_endpoint():
    """
    모든 문서를 DB와 파일 시스템에서 삭제합니다.
    """
    logger.info("Delete all documents request received")
    
    try:
        # 1. 벡터 DB에서 모든 문서 삭제
        db_deleted_count = delete_all_documents()
        
        # 2. 파일 시스템에서 모든 파일 삭제
        file_deleted_count = DocumentFileManager.delete_all_files()
        
        # 3. 처리 상태 초기화
        pdf_processing_status.clear()
        
        logger.info(f"Successfully deleted all documents (DB: {db_deleted_count} docs, Files: {file_deleted_count} files)")
        
        return {
            "message": "All documents deleted successfully",
            "deleted_documents_count": db_deleted_count,
            "deleted_files_count": file_deleted_count
        }
        
    except VectorDBError as e:
        logger.error(f"Vector DB error during delete all: {e}")
        raise HTTPException(status_code=500, detail=f"DB 전체 삭제 오류: {e.message}")
    except FileProcessingError as e:
        logger.error(f"File processing error during delete all: {e}")
        raise HTTPException(status_code=500, detail=f"파일 전체 삭제 오류: {e.message}")
    except Exception as e:
        logger.error(f"Unexpected error during delete all: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="전체 문서 삭제 중 예상치 못한 오류가 발생했습니다")

@router.get("/documents/{document_id}")
def get_document_details(document_id: str):
    """
    특정 문서의 상세 정보를 반환합니다.
    """
    if not validate_document_id(document_id):
        raise HTTPException(status_code=400, detail=f"Invalid document ID: {document_id}")
    
    try:
        # DB에서 멀티모달 문서 정보 조회
        db_info = get_multimodal_document_info(document_id)
        if not db_info:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found in database")
        
        # 파일 시스템에서 파일 정보 조회
        file_info = DocumentFileManager.get_file_info(document_id)
        
        # 정보 통합
        result = {
            "document_id": document_id,
            "db_info": db_info,
            "file_info": file_info,
            "processing_status": pdf_processing_status.get(document_id)
        }
        
        return result
        
    except VectorDBError as e:
        logger.error(f"Vector DB error getting document details: {e}")
        raise HTTPException(status_code=500, detail=f"문서 정보 조회 오류: {e.message}")
    except FileProcessingError as e:
        logger.error(f"File processing error getting document details: {e}")
        raise HTTPException(status_code=500, detail=f"파일 정보 조회 오류: {e.message}")

@router.post("/documents/cleanup")
def cleanup_orphaned_files():
    """
    벡터 DB에 없는 고아 파일들을 정리합니다.
    """
    logger.info("Cleanup orphaned files request received")
    
    try:
        # 1. 벡터 DB에서 유효한 document_id 목록 가져오기
        documents = get_all_documents()
        valid_document_ids = [doc["document_id"] for doc in documents]
        
        # 2. 고아 파일 정리
        orphaned_count = DocumentFileManager.cleanup_orphaned_files(valid_document_ids)
        
        logger.info(f"Cleaned up {orphaned_count} orphaned files")
        
        return {
            "message": "Orphaned files cleanup completed",
            "cleaned_files_count": orphaned_count,
            "valid_documents_count": len(valid_document_ids)
        }
        
    except (VectorDBError, FileProcessingError) as e:
        logger.error(f"Error during cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"정리 작업 오류: {e.message}")
    except Exception as e:
        logger.error(f"Unexpected error during cleanup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="정리 작업 중 예상치 못한 오류가 발생했습니다")

@router.get("/storage/stats")
def get_storage_statistics():
    """
    저장소 사용량 통계를 반환합니다.
    """
    try:
        # 파일 시스템 통계
        file_stats = DocumentFileManager.get_storage_stats()
        
        # 벡터 DB 통계
        documents = get_all_documents()
        total_chunks = sum(doc.get("chunk_count", 0) for doc in documents)
        
        return {
            "file_storage": file_stats,
            "vector_db": {
                "total_documents": len(documents),
                "total_chunks": total_chunks
            }
        }
        
    except (VectorDBError, FileProcessingError) as e:
        logger.error(f"Error getting storage stats: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 오류: {e.message}")
    except Exception as e:
        logger.error(f"Unexpected error getting storage stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="통계 조회 중 예상치 못한 오류가 발생했습니다")
