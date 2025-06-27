from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, WebSocket # WebSocket 추가
from fastapi.responses import JSONResponse
import os
import uuid
from typing import List, Optional, Dict, Any # 기존 typing 임포트 유지

# from pydantic import BaseModel # schemas.py로 이동
from app.schemas import UploadResponse # schemas.py에서 UploadResponse 임포트

import logging
logger = logging.getLogger(__name__)
# 기본 로깅 레벨 설정 (main.py 또는 config에서 중앙 관리 권장)
if not logger.hasHandlers(): # 중복 핸들러 방지
    logging.basicConfig(level=logging.INFO)


# Celery 앱 및 태스크 임포트
from app.tasks import celery_app
# from app.tasks import ingest_pdf # 태스크 직접 임포트 대신 send_task 사용

# 설정값 (예: 업로드 디렉토리) - config.py 또는 환경변수에서 로드
# from app.config import settings
# 여기서는 간단하게 환경변수 또는 기본값 사용
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads/")


# UploadResponse는 schemas.py로 이동됨

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_pdf_endpoint( # 함수 이름 변경 (upload_pdf -> upload_pdf_endpoint)
    file: UploadFile = File(...)
):
    logger.info(f"Upload request received for file: {file.filename}")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")

    # 파일 저장 디렉토리 확인 및 생성
    # UPLOAD_DIR 경로는 컨테이너 내부 경로 기준이어야 함.
    # Dockerfile에서 COPY backend/uploads /app/uploads 로 했고, WORKDIR /app 이므로,
    # 'uploads/'는 /app/uploads/ 를 의미하게 됨.
    # os.makedirs(UPLOAD_DIR, exist_ok=True) # Dockerfile에서 이미 생성되거나 볼륨 마운트됨

    document_id = f"{os.path.splitext(file.filename)[0]}_{str(uuid.uuid4())[:8]}"
    safe_filename = f"{document_id}.pdf" # Celery 태스크에서 이 파일을 읽어야 함

    # UPLOAD_DIR이 /app/uploads를 가리키도록 WORKDIR /app 기준으로 설정
    # 또는 절대 경로 사용: file_path = os.path.join("/app", UPLOAD_DIR, safe_filename)
    # 여기서는 UPLOAD_DIR이 'uploads/' 이고 WORKDIR이 /app 이므로 /app/uploads/ 가 됨.
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    # 컨테이너 내 절대 경로로 변환 (Celery 워커가 접근할 수 있도록)
    # 이 부분은 Docker 볼륨 마운트 설정과 일치해야 함.
    # 현재 docker-compose.yml에서 backend/uploads가 /app/uploads로 마운트되므로,
    # /app/uploads/{safe_filename} 경로를 Celery 태스크에 전달해야 함.
    # UPLOAD_DIR 자체가 이미 /app 내부의 상대 경로 'uploads/'를 의미하도록 사용.
    # file_path_for_celery = os.path.abspath(file_path) # 이건 API 서버 컨테이너 내의 절대경로. 워커도 동일한 마운트 구조를 가져야 함.
    # Dockerfile에서 WORKDIR /app 이므로, file_path는 /app/uploads/safe_filename 형태가 됨.
    # Celery 워커도 동일한 WORKDIR과 볼륨 마운트를 사용하므로 이 경로 그대로 사용 가능.

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        # UPLOAD_DIR (예: /app/uploads)에 파일 저장
        # 디렉토리가 없으면 생성 (프로덕션에서는 시작 시점에 생성 권장)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as buffer:
            buffer.write(content)

        logger.info(f"File saved to: {file_path} for document_id: {document_id}")

        task = celery_app.send_task(
            "app.tasks.ingest_pdf", # tasks.py에서 정의한 태스크 이름
            args=[file_path, document_id] # Celery 태스크에 전달할 인자
        )

        logger.info(f"Celery task sent for document_id {document_id}. Task ID: {task.id}")

        return UploadResponse(
            task_id=task.id,
            filename=file.filename,
            document_id=document_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during file upload or task sending for {file.filename}: {e}", exc_info=True)
        if os.path.exists(file_path): # 실패 시 임시 파일 정리
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file after error: {file_path}")
            except OSError as cleanup_error:
                logger.error(f"Error cleaning up temporary file {file_path}: {cleanup_error}")
        raise HTTPException(status_code=500, detail=f"Could not process file: {str(e)}")
    finally:
        if hasattr(file, 'close') and callable(file.close):
             await file.close()


# WebSocket 엔드포인트 구현
from fastapi import WebSocketDisconnect # WebSocketDisconnect 임포트
from app.rag import stream_answer # rag.py의 stream_answer 함수 임포트

@router.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted from client.")

    try:
        embed_model = websocket.app.state.embed_model
        faiss_index = websocket.app.state.faiss_index
        meta_db = websocket.app.state.meta_db

        if embed_model is None or faiss_index is None or meta_db is None:
            error_msg = "Error: Server core components (embed_model, faiss_index, meta_db) are not initialized."
            logger.error(error_msg + " Check server startup logs.")
            await websocket.send_text(error_msg)
            await websocket.close(code=1011) # Internal server error
            return

    except AttributeError as ae:
        error_msg = "Error: Server configuration error (app.state attributes not found)."
        logger.error(error_msg + f" Details: {ae}", exc_info=True)
        await websocket.send_text(error_msg)
        await websocket.close(code=1011)
        return
    except Exception as e_state: # 기타 app.state 접근 오류
        error_msg = "Error: Failed to access server state."
        logger.error(error_msg + f" Details: {e_state}", exc_info=True)
        await websocket.send_text(error_msg)
        await websocket.close(code=1011)
        return

    try:
        while True:
            question = await websocket.receive_text()
            logger.info(f"WebSocket received question (first 100 chars): '{question[:100]}...'")

            full_response_for_log = ""
            try:
                async for chunk in stream_answer(question, embed_model, faiss_index, meta_db):
                    await websocket.send_text(chunk)
                    if not chunk.startswith("Error:"): # 에러 메시지는 전체 로깅하지 않음 (rag.py에서 이미 로깅)
                        full_response_for_log += chunk.strip() + " " # 로깅용으로 공백 추가
            except Exception as e_stream: # stream_answer 내부에서 처리되지 않은 예외
                logger.error(f"Error during stream_answer execution: {e_stream}", exc_info=True)
                await websocket.send_text("Error: An error occurred while processing your question.")
                # 스트림 중단 후 다음 질문 대기 또는 연결 종료 고려
                continue # 다음 질문 대기

            logger.info(f"WebSocket finished streaming answer for: '{question[:100]}...'. Response (first 100 chars of accumulated log): '{full_response_for_log[:100]}...'")

    except WebSocketDisconnect:
        logger.info(f"WebSocket connection closed by client: {websocket.client}")
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket chat endpoint: {e}", exc_info=True)
        # 연결이 아직 유효하다면 에러 메시지 전송 시도
        if websocket.client_state == websocket.client_state.CONNECTED:
            try:
                await websocket.send_text("Error: An unexpected server error occurred. Please try reconnecting.")
            except Exception as send_err:
                logger.error(f"Failed to send error message to WebSocket client: {send_err}")
        # FastAPI가 연결을 자동으로 닫아줄 수 있지만, 명시적으로 닫는 것도 고려
        # await websocket.close(code=1011)
    finally:
        logger.info(f"WebSocket connection terminated for client: {websocket.client}")


# --- 더 이상 사용되지 않거나 수정이 필요한 기존 엔드포인트들은 주석 처리 또는 삭제 ---
# (ChatRequest, ChatResponse, chat_with_llm 등은 RAG 및 WebSocket 구현 시 재검토)
# (get_upload_status, ollama_status, ollama_models, 문서 관리 엔드포인트 등)

# 예시: Celery 태스크 상태 확인 엔드포인트 (선택 사항)
@router.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    task_result = celery_app.AsyncResult(task_id)
    response = {
        "task_id": task_id,
        "status": task_result.status,
        "result": task_result.result if task_result.ready() else None,
    }
    if task_result.failed():
        response["error_message"] = str(task_result.info) # 에러 정보 포함
    return response

# main.py에서 app.include_router(router, prefix="/api") 등으로 사용될 것임.
# from app.main import app # 순환 참조 주의
# app.include_router(router, prefix="/api") # 이것은 main.py에서 수행
