from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates # React 프론트 사용으로 주석 처리
from fastapi.responses import HTMLResponse # 기본 루트용으로 유지 가능
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.middleware.trustedhost import TrustedHostMiddleware # 필요시 활성화

import uvicorn
import os
import logging # 표준 로깅 사용

# --- 신규 임포트 ---
import faiss
from tinydb import TinyDB, Query
from sentence_transformers import SentenceTransformer # 타입 힌팅용
from app.utils.embedding import get_embed_model, MODEL_NAME as EMBED_MODEL_NAME
# from app.config import settings # settings 사용 방식 재검토 (환경변수 직접 사용 등)

# --- 로깅 설정 ---
# setup_logging() # 기존 로깅 설정 (필요시 유지 또는 표준 로깅으로 대체)
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # 핸들러 중복 추가 방지
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())


# --- FAISS 및 TinyDB 설정 ---
# Docker 볼륨 마운트 경로와 일치해야 함: ./backend/faiss_data:/faiss_data (docker-compose.yml)
# 컨테이너 내부에서는 /faiss_data 로 접근하게 됨.
# FAISS_DATA_DIR = os.getenv("FAISS_DATA_DIR", "faiss_data") # backend/faiss_data가 볼륨 마운트됨
# UPLOAD_DIR과 마찬가지로, WORKDIR /app 기준으로 상대 경로 또는 절대 경로 사용
# 여기서는 /faiss_data (절대 경로)를 사용 (docker-compose.yml의 볼륨 정의와 일치시킴)
FAISS_DATA_DIR = os.getenv("FAISS_DATA_PATH_CONTAINER", "/faiss_data") # 컨테이너 내 경로
FAISS_INDEX_PATH = os.path.join(FAISS_DATA_DIR, "hnsw_index.faiss")
TINYDB_PATH = os.path.join(FAISS_DATA_DIR, "metadata_db.json")
EMBEDDING_DIMENSION = 768 # jhgan/ko-sbert-nli 모델의 임베딩 차원


# --- FastAPI 애플리케이션 생성 ---
app = FastAPI(
    title="RAG API with FAISS and Ollama",
    description="PDF Upload -> OCR -> Embedding -> FAISS -> RAG with Ollama",
    version="1.0.0",
    # OpenAPI 문서 경로 (개발 시에만 활성화 권장)
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# --- 스타트업 이벤트 핸들러 ---
@app.on_event("startup")
async def startup_event_handler(): # 함수 이름 변경 (예: startup_event -> startup_event_handler)
    logger.info("Application startup: Loading resources...")

    # 데이터 디렉토리 생성 (FAISS_DATA_DIR)
    # 이 경로는 Docker 볼륨에 의해 호스트와 연결되므로, 컨테이너 시작 시 존재해야 함.
    # os.makedirs는 컨테이너 내부에서 실행됨.
    try:
        os.makedirs(FAISS_DATA_DIR, exist_ok=True)
        logger.info(f"Data directory '{FAISS_DATA_DIR}' ensured/created.")
    except OSError as e:
        logger.error(f"Could not create data directory {FAISS_DATA_DIR}: {e}", exc_info=True)
        # 디렉토리 생성 실패 시 심각한 문제이므로, 추가 처리 필요할 수 있음.

    # 1. 임베딩 모델 로드
    try:
        app.state.embed_model = get_embed_model()
        logger.info(f"Embedding model '{EMBED_MODEL_NAME}' loaded and attached to app.state.")
    except RuntimeError as e:
        logger.error(f"CRITICAL: Failed to load embedding model during startup: {e}", exc_info=True)
        app.state.embed_model = None
        # 모델 로드 실패 시, 이후 FAISS 인덱스 초기화 등도 영향을 받음.

    # 2. FAISS 인덱스 로드 또는 생성
    if getattr(app.state, 'embed_model', None) is not None:
        try:
            if os.path.exists(FAISS_INDEX_PATH):
                logger.info(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
                app.state.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
                logger.info("FAISS index loaded successfully.")
                if hasattr(app.state.faiss_index, 'd'): logger.info(f"  Index dimension: {app.state.faiss_index.d}")
                if hasattr(app.state.faiss_index, 'ntotal'): logger.info(f"  Total vectors in index: {app.state.faiss_index.ntotal}")
            else:
                logger.info(f"FAISS index file not found at {FAISS_INDEX_PATH}. Creating a new HNSWFlat index.")
                # IndexHNSWFlat(dimension, M) - M은 이웃 수 (default 32)
                index_hnsw = faiss.IndexHNSWFlat(EMBEDDING_DIMENSION, 32)
                # index_hnsw.hnsw.efConstruction = 40 # 기본값 사용 또는 필요시 조정
                # index_hnsw.hnsw.efSearch 는 검색 시점에 설정
                app.state.faiss_index = index_hnsw # IndexHNSWFlat은 Index 인터페이스를 따름
                logger.info(f"New FAISS IndexHNSWFlat created (dim={EMBEDDING_DIMENSION}, M=32).")
        except Exception as e:
            logger.error(f"Failed to load or create FAISS index: {e}", exc_info=True)
            app.state.faiss_index = None
    else:
        logger.warning("Embedding model not loaded, FAISS index initialization skipped.")
        app.state.faiss_index = None

    # 3. TinyDB 로드
    try:
        app.state.meta_db = TinyDB(TINYDB_PATH)
        logger.info(f"TinyDB loaded/created at {TINYDB_PATH}. Total records: {len(app.state.meta_db)}")
    except Exception as e:
        logger.error(f"Failed to load or create TinyDB: {e}", exc_info=True)
        app.state.meta_db = None

    logger.info("Application startup sequence complete.")

# --- 셧다운 이벤트 핸들러 ---
@app.on_event("shutdown")
async def shutdown_event_handler(): # 함수 이름 변경
    logger.info("Application shutdown: Saving resources...")
    if hasattr(app.state, 'faiss_index') and app.state.faiss_index is not None:
        try:
            # FAISS_DATA_DIR이 존재하는지 다시 한번 확인 (컨테이너 종료 시점)
            if not os.path.exists(FAISS_DATA_DIR):
                os.makedirs(FAISS_DATA_DIR, exist_ok=True)
                logger.info(f"Data directory '{FAISS_DATA_DIR}' created during shutdown for saving index.")

            logger.info(f"Saving FAISS index to {FAISS_INDEX_PATH}...")
            faiss.write_index(app.state.faiss_index, FAISS_INDEX_PATH)
            logger.info("FAISS index saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}", exc_info=True)

    if hasattr(app.state, 'meta_db') and app.state.meta_db is not None:
        try:
            app.state.meta_db.close()
            logger.info("TinyDB connection closed.")
        except Exception as e:
            logger.error(f"Error closing TinyDB: {e}", exc_info=True)

    logger.info("Application shutdown sequence complete.")

# --- 미들웨어 설정 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 실제 프로덕션에서는 특정 도메인 목록으로 제한
    allow_credentials=True,
    allow_methods=["*"], # ["GET", "POST", "PUT", "DELETE", "OPTIONS"] 등 필요한 것만
    allow_headers=["*"],
)

# --- API 라우터 포함 ---
from app.api import router as api_router
app.include_router(api_router, prefix="/api") # /api 접두사 추가

# --- 기본 루트 엔드포인트 ---
@app.get("/", include_in_schema=False) # API 문서에는 불필요
async def read_root_redirect(request: Request):
    # API 문서 페이지로 리디렉션 또는 간단한 환영 메시지
    # from fastapi.responses import RedirectResponse
    # return RedirectResponse(url="/api/docs")
    return HTMLResponse(content="<h1>RAG API Backend is running</h1><p>Access API docs at <a href='/api/docs'>/api/docs</a> or <a href='/api/redoc'>/api/redoc</a>.</p>")

# Docker 환경에서는 uvicorn 명령어로 직접 실행되므로, 아래 __main__ 블록은 보통 사용되지 않음.
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
# reload=True는 개발 중에만 사용
