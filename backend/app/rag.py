import asyncio
import logging
from typing import AsyncGenerator, List # List 추가

# 타입 힌팅용 (실제 인스턴스는 FastAPI 의존성 주입으로 전달)
from sentence_transformers import SentenceTransformer
import faiss # faiss 라이브러리 직접 임포트
from tinydb import TinyDB, Query
import numpy as np

logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # 핸들러 중복 추가 방지
    logging.basicConfig(level=logging.INFO)

OLLAMA_MODEL_NAME = "mistral"
OLLAMA_CLI_PATH = "ollama" # 시스템 PATH에 등록 가정

async def stream_answer(
    question: str,
    embed_model: SentenceTransformer,
    faiss_index: faiss.Index, # faiss.Index로 타입 명시
    meta_db: TinyDB,
    top_k: int = 4
) -> AsyncGenerator[str, None]:
    logger.info(f"RAG stream_answer invoked for question (first 50 chars): '{question[:50]}...'")

    if not question.strip():
        yield "Error: Question cannot be empty.\n"
        return

    if embed_model is None:
        logger.error("Embedding model is not available.")
        yield "Error: Embedding model not initialized.\n"
        return
    if faiss_index is None:
        logger.error("FAISS index is not available.")
        yield "Error: FAISS index not initialized.\n"
        return
    if meta_db is None:
        logger.error("MetaDB (TinyDB) is not available.")
        yield "Error: Metadata database not initialized.\n"
        return

    try:
        logger.debug("1. Embedding the question...")
        query_embedding = embed_model.encode([question], convert_to_numpy=True)
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        logger.debug(f"2. Searching FAISS index (top_k={top_k}). Index ntotal: {faiss_index.ntotal}")
        if faiss_index.ntotal == 0:
            logger.warning("FAISS index is empty. Cannot perform search.")
            yield "현재 지식 베이스에 정보가 없습니다. 문서를 먼저 업로드해주세요.\n"
            return

        # HNSW 인덱스 검색 파라미터 설정 (선택 사항)
        if isinstance(faiss_index, faiss.IndexHNSW): # IndexHNSWFlat 등 HNSW 계열
            # faiss.ParameterSpace().set_index_parameter(faiss_index, "efSearch", 64) # 예시
            # logger.debug(f"Set FAISS HNSW efSearch to 64 (example value)")
            pass


        distances, indices = faiss_index.search(query_embedding, top_k)

        if indices.size == 0 or (indices[0][0] == -1 and len(indices[0]) == 1) : # 결과 없거나 [-1] 반환 시
            logger.warning("No relevant chunks found in FAISS index for the question.")
            yield "관련 정보를 찾을 수 없습니다. 다른 질문을 시도해보세요.\n"
            return

        logger.debug(f"FAISS search results - Indices: {indices}, Distances: {distances}")

        # 3. TinyDB에서 컨텍스트 검색 (개선 필요 지점)
        # 현재 FAISS ID와 TinyDB 레코드 간의 명확한 매핑 전략 부재.
        # 임시로, FAISS ID가 TinyDB의 Document ID (1-based) - 1 이라고 가정.
        # 이 가정은 매우 취약하며, 실제 구현에서는 수정 필요.
        # (예: ingest_pdf에서 각 청크에 UUID를 부여하고, FAISS에는 add_with_ids로 (UUID 정수 해시) 저장,
        # TinyDB에는 UUID와 텍스트 저장. 검색 시 UUID로 TinyDB 조회)
        context_texts: List[str] = []
        retrieved_faiss_ids: List[int] = indices[0].tolist() # numpy 배열을 리스트로 변환

        ChunkQuery = Query() # TinyDB Query 객체

        # === 임시 컨텍스트 검색 로직 (개선 필요) ===
        # 아래 로직은 FAISS ID와 TinyDB 간의 직접적인 연결고리가 없다는 가정 하에,
        # FAISS ID를 TinyDB의 내부 doc_id로 매핑하려는 시도입니다.
        # 이 방식은 데이터 추가/삭제에 매우 취약합니다.
        # 올바른 방법은 ingest_pdf 단계에서 FAISS ID 또는 고유 식별자를 TinyDB에 함께 저장하는 것입니다.

        # 현재 TinyDB 스키마: {"document_id": str, "chunk_index_in_doc": int, "text": str}
        # FAISS ID (0-based)를 사용하여 위 스키마에서 텍스트를 찾아야 함.
        # 이는 FAISS에 저장된 모든 벡터에 대해 전역적인 순서가 있고,
        # TinyDB에도 동일한 순서로 모든 청크가 저장되어 있어야 가능.

        # 예시: TinyDB에 저장된 모든 청크를 가져와서 FAISS ID로 인덱싱 (메모리 사용량 주의)
        # all_chunks_in_tinydb = meta_db.all() # 모든 레코드 로드
        # for faiss_id_val in retrieved_faiss_ids:
        #     if 0 <= faiss_id_val < len(all_chunks_in_tinydb):
        #         record = all_chunks_in_tinydb[faiss_id_val]
        #         if record and 'text' in record:
        #             context_texts.append(record['text'])
        #             logger.debug(f"Retrieved context for FAISS ID {faiss_id_val} (via all_chunks_in_tinydb offset): {record['text'][:50]}...")
        #         else:
        #             logger.warning(f"Could not retrieve text for FAISS ID {faiss_id_val} using offset from all_chunks_in_tinydb.")
        #     else:
        #         logger.warning(f"FAISS ID {faiss_id_val} is out of bounds for all_chunks_in_tinydb (len: {len(all_chunks_in_tinydb)}).")

        # 더 나은 임시 방안: 설계서의 `meta_db.get(idx)["text"]`를 따르기 위해,
        # idx를 TinyDB의 내부 doc_id로 가정. FAISS ID (0-based)를 doc_id (1-based)로 변환.
        for faiss_id_val in retrieved_faiss_ids:
            if faiss_id_val == -1 : continue # 유효하지 않은 ID는 건너뜀
            # 이 매핑은 FAISS에 벡터가 추가된 순서와 TinyDB에 문서가 삽입된 순서가
            # 1:1로 일치하고, 삭제/변경이 없다는 매우 강한 가정에 의존합니다.
            # 실제로는 `faiss_id` 필드를 TinyDB에 저장하고 `ChunkQuery.faiss_id == faiss_id_val`로 검색해야 합니다.
            retrieved_doc = meta_db.get(doc_id=int(faiss_id_val) + 1)
            if retrieved_doc and 'text' in retrieved_doc:
                context_texts.append(retrieved_doc['text'])
                logger.debug(f"Retrieved context for FAISS ID {faiss_id_val} (mapped to TinyDB doc_id {int(faiss_id_val) + 1}): {retrieved_doc['text'][:50]}...")
            else:
                logger.warning(f"Could not retrieve text for FAISS ID {faiss_id_val} by mapping to TinyDB doc_id {int(faiss_id_val) + 1}.")
        # === 임시 컨텍스트 검색 로직 끝 ===


        if not context_texts:
            logger.warning("No context could be retrieved from TinyDB based on FAISS results.")
            yield "관련 컨텍스트 정보를 찾을 수 없어 답변을 생성할 수 없습니다.\n"
            return

        context_str = "\n\n".join(context_texts)
        logger.debug(f"Retrieved context. Total length: {len(context_str)} chars. Preview: {context_str[:200]}...")

        prompt = f"""너는 전문 문서 비서야. 다음 context만 참고하여 한국어로 정확히 답하라.
### 컨텍스트:
{context_str}
###
질문: {question}
답변:"""
        logger.debug(f"Generated prompt for Ollama (first 200 chars): \n{prompt[:200]}\n...")

        # Ollama CLI 비동기 실행
        # OLLAMA_HOST 환경 변수가 Docker Compose에서 설정되어 있다면, Ollama CLI가 이를 사용함.
        process = await asyncio.create_subprocess_exec(
            OLLAMA_CLI_PATH, 'run', OLLAMA_MODEL_NAME,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        logger.info(f"Ollama process started (model: {OLLAMA_MODEL_NAME}) for question: '{question[:50]}...'")

        if process.stdin:
            process.stdin.write(prompt.encode('utf-8'))
            await process.stdin.drain()
            process.stdin.close() # EOF
        else:
            logger.error("Failed to get stdin for Ollama process.")
            yield "Error: Could not send prompt to Ollama.\n"
            return

        full_response = ""
        if process.stdout:
            async for line_bytes in process.stdout:
                line = line_bytes.decode('utf-8', errors='ignore').strip()
                # Ollama CLI 'run'은 응답을 바로 텍스트로 스트리밍.
                # 만약 JSON 객체 ({"response": "...", "done": ...}) 형태라면 파싱 필요.
                # 현재는 순수 텍스트로 가정.
                if line: # 빈 줄은 무시
                    yield line # 설계서에서는 공백 추가했으나, 여기서는 원본 그대로 전달 후 프론트에서 처리
                    full_response += line # 전체 응답 누적 (로깅용)
            logger.info("Finished streaming from Ollama stdout.")
        else:
            logger.error("Failed to get stdout for Ollama process.")
            yield "Error: No response received from Ollama.\n"
            return

        # 로깅을 위해 전체 응답의 일부를 표시
        logger.debug(f"Full Ollama response (first 200 chars): {full_response[:200]}")

        stderr_bytes = await process.stderr.read() if process.stderr else b''
        stderr_output = stderr_bytes.decode('utf-8', errors='ignore').strip()
        if stderr_output:
            logger.error(f"Ollama stderr output: {stderr_output}")

        await process.wait()
        if process.returncode != 0:
            logger.error(f"Ollama process exited with code {process.returncode}.")
            # 스트리밍 중 오류 발생 시 이미 일부 응답이 전송되었을 수 있음.
            # 추가 에러 메시지를 yield 할 수 있지만, 사용자 경험 고려.
            # yield f"Error: Ollama process error (code: {process.returncode}).\n"

    except faiss.FaissException as faiss_err: # FAISS 명시적 예외
        logger.error(f"A FAISS error occurred: {faiss_err}", exc_info=True)
        yield "Error: Vector database operation failed.\n"
    except Exception as e:
        logger.error(f"An unexpected error occurred in stream_answer: {e}", exc_info=True)
        yield "Error: An unexpected error occurred while generating the answer.\n"
    finally:
        logger.info(f"RAG stream_answer finished for question: '{question[:50]}...'")
