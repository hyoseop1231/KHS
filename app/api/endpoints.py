from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
import shutil
import os
import uuid # For unique document IDs

# 서비스 모듈 임포트
from app.services.ocr_service import extract_text_from_pdf
from app.services.text_processing_service import split_text_into_chunks, get_embeddings
from app.services.vector_db_service import store_vectors, search_similar_vectors
from app.services.llm_service import process_llm_chat_request, DEFAULT_MODEL

# Pydantic 모델 (요청/응답 본문 정의)
from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    query: str
    document_id: Optional[str] = None # 특정 문서 ID에 대해 질문할 경우
    model_name: Optional[str] = DEFAULT_MODEL
    lang: Optional[str] = "ko"

class ChatResponse(BaseModel):
    query: str
    response: str
    source_document_id: Optional[str] = None
    retrieved_chunks_preview: Optional[List[str]] = None # 디버깅/정보용

router = APIRouter()

# 업로드 디렉토리 설정
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def process_pdf_background(file_path: str, document_id: str):
    """
    백그라운드에서 PDF 처리 작업을 수행합니다.
    OCR -> Chunking -> Embedding -> Storage
    """
    print(f"Background task started: Processing PDF {document_id} from {file_path}")
    try:
        # 1. OCR
        print(f"[Task {document_id}] Step 1: Extracting text using OCR...")
        extracted_text = extract_text_from_pdf(pdf_path)
        if not extracted_text or extracted_text.startswith("Error") or extracted_text.startswith("OCR Error"):
            print(f"[Task {document_id}] Error during OCR: {extracted_text}")
            # TODO: 사용자에게 오류 알림 메커니즘 필요 (예: 웹소켓, 상태 폴링)
            return

        # 2. Text Chunking
        print(f"[Task {document_id}] Step 2: Splitting text into chunks...")
        text_chunks = split_text_into_chunks(extracted_text)
        if not text_chunks:
            print(f"[Task {document_id}] No text chunks generated.")
            return

        # 3. Embedding Generation
        print(f"[Task {document_id}] Step 3: Generating embeddings for {len(text_chunks)} chunks...")
        embeddings = get_embeddings(text_chunks)
        if not embeddings or not all(e for e in embeddings): # 일부 임베딩 생성 실패 확인
            print(f"[Task {document_id}] Error generating embeddings or empty embeddings list.")
            return

        # 유효한 임베딩과 청크만 필터링 (get_embeddings가 빈 리스트를 반환할 수 있으므로)
        valid_chunks = []
        valid_embeddings = []
        for i, emb in enumerate(embeddings):
            if emb: # 비어있지 않은 임베딩만 사용
                valid_chunks.append(text_chunks[i])
                valid_embeddings.append(emb)

        if not valid_chunks:
            print(f"[Task {document_id}] No valid embeddings generated to store.")
            return

        # 4. Store in Vector DB
        print(f"[Task {document_id}] Step 4: Storing {len(valid_chunks)} chunks and embeddings in Vector DB...")
        # 메타데이터 준비 (각 청크가 어떤 원본 문서에서 왔는지 명시)
        metadatas = [{'source_document_id': document_id, 'chunk_index': i} for i in range(len(valid_chunks))]
        store_vectors(document_id=document_id, text_chunks=valid_chunks, vectors=valid_embeddings, metadatas=metadatas)

        print(f"[Task {document_id}] Successfully processed and stored PDF: {document_id}")

    except Exception as e:
        print(f"[Task {document_id}] Error during background PDF processing for {document_id}: {e}")
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
    Handles PDF file upload. The file is saved, and processing (OCR, chunking, embedding, storage)
    is offloaded to a background task.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")

    # 고유한 문서 ID 생성 (예: 파일 이름과 UUID 결합 또는 UUID만 사용)
    # 여기서는 단순화를 위해 원본 파일 이름을 사용하나, 충돌 가능성 있음. UUID 권장.
    # document_id = str(uuid.uuid4())
    # document_id = file.filename # 중복 파일명 처리 필요

    # 파일명에서 확장자 제거하고, 안전한 이름으로 변경 (선택적)
    original_filename = file.filename
    safe_filename = "".join(c if c.isalnum() or c in ['.', '_'] else '_' for c in original_filename)
    document_id = f"{os.path.splitext(safe_filename)[0]}_{str(uuid.uuid4())[:8]}" # 파일명 기반 + 짧은 UUID

    file_path = os.path.join(UPLOAD_DIR, f"{document_id}_{original_filename}") # 저장 시 고유 ID 포함

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # PDF 처리 작업을 백그라운드로 넘김
        background_tasks.add_task(process_pdf_background, file_path, document_id)

        return JSONResponse(
            content={
                "message": "File uploaded successfully. Processing started in the background.",
                "filename": original_filename,
                "document_id": document_id, # 클라이언트가 이 ID를 사용하여 상태를 추적하거나, 채팅 시 문맥 지정 가능
                "detail": "The PDF is being processed. This may take some time depending on the file size and content."
            },
            status_code=202 # Accepted
        )
    except Exception as e:
        # 여기서 오류 발생 시 파일이 저장되었을 수 있으므로 정리 로직 추가 가능
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Could not save or start processing file: {str(e)}")
    finally:
        file.file.close()


@router.post("/chat/", response_model=ChatResponse)
async def chat_with_llm(chat_request: ChatRequest):
    """
    Handles chat requests.
    1. Embeds the user query.
    2. Searches for relevant chunks in the Vector DB (optionally filtered by document_id).
    3. Constructs a RAG prompt and gets a response from the LLM.
    """
    query = chat_request.query
    document_id_filter = chat_request.document_id
    model_name = chat_request.model_name or DEFAULT_MODEL # Pydantic 모델에서 기본값 처리 가능
    lang = chat_request.lang or "ko"

    if not query:
        raise HTTPException(status_code=400, detail="Query not provided.")

    print(f"Chat request received: Query='{query}', DocID='{document_id_filter}', Model='{model_name}', Lang='{lang}'")

    # 1. Embed user query
    print("Step 1: Embedding user query...")
    query_embedding_list = get_embeddings([query]) # get_embeddings는 리스트를 받음
    if not query_embedding_list or not query_embedding_list[0]:
        raise HTTPException(status_code=500, detail="Could not generate embedding for the query.")
    query_embedding = query_embedding_list[0]

    # 2. Search Vector DB
    print("Step 2: Searching Vector DB for relevant chunks...")
    filter_metadata = {}
    if document_id_filter:
        filter_metadata = {"source_document_id": document_id_filter}
        print(f"Applying filter: {filter_metadata}")

    # top_k 값을 얼마로 할지 결정 (예: 3-5개)
    retrieved_chunks = search_similar_vectors(query_embedding, top_k=3, filter_metadata=filter_metadata)

    if not retrieved_chunks:
        print("No relevant chunks found in Vector DB.")
        # LLM에 문맥 없이 질문하거나, 사용자에게 알림
        # 여기서는 문맥 없이 LLM에 직접 질문하거나, 특정 메시지 반환
        # llm_response_text = f"죄송합니다, '{query}'에 대한 관련 정보를 찾을 수 없습니다." (직접 응답)
        # 또는, LLM에게 문맥 없이 질문 (이는 llm_service.process_llm_chat_request에서 처리됨)
        pass

    retrieved_chunk_texts_preview = [chunk.get('text', '')[:100] + "..." for chunk in retrieved_chunks] # 미리보기용

    # 3. Get LLM response via RAG
    print("Step 3: Getting LLM response using RAG...")
    llm_response_text = process_llm_chat_request(
        user_query=query,
        retrieved_docs=retrieved_chunks, # 검색된 청크 전달
        model_name=model_name,
        lang=lang
    )

    return ChatResponse(
        query=query,
        response=llm_response_text,
        source_document_id=document_id_filter,
        retrieved_chunks_preview=retrieved_chunk_texts_preview
    )

# TODO: (선택 사항) PDF 처리 상태를 확인할 수 있는 엔드포인트 추가
# @router.get("/upload_status/{document_id}")
# async def get_upload_status(document_id: str):
#     # 이 부분은 백그라운드 작업의 상태를 저장하고 조회하는 메커니즘이 필요합니다.
#     # (예: Redis, DB, 또는 간단한 인메모리 딕셔너리 - 프로덕션에는 부적합)
#     return {"document_id": document_id, "status": "processing_status_placeholder"}
