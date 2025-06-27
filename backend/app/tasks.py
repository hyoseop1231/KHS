from celery import Celery
import os
import fitz # PyMuPDF
import logging

# 로거 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Celery 애플리케이션 생성
redis_host = os.environ.get('REDIS_HOST', 'redis')
redis_port = os.environ.get('REDIS_PORT', '6379')
broker_url = f"redis://{redis_host}:{redis_port}/0"
backend_url = f"redis://{redis_host}:{redis_port}/0"

celery_app = Celery(
    "worker",
    broker=broker_url,
    backend=backend_url,
    include=['app.tasks'] # 태스크 모듈 자동 발견
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Seoul',
    enable_utc=True,
)

def extract_images_from_page(page: fitz.Page) -> list[bytes]:
    """Helper function to extract images from a single PDF page."""
    images = []
    for img_info in page.get_images(full=True): # 변경: img_index 제거
        xref = img_info[0]
        base_image = page.parent.extract_image(xref)
        if base_image:
            images.append(base_image["image"])
    return images

def clean_text(text: str) -> str:
    """Basic text cleaning function."""
    if not text:
        return ""
    text = ' '.join(text.split()) # Normalize whitespace
    # Add more specific cleaning rules as needed
    return text.strip()

def make_chunks(text: str, size: int, overlap: int) -> list[str]:
    """Creates overlapping text chunks."""
    if not text or size <= 0:
        return []
    if overlap >= size : # 오버랩은 청크 크기보다 작아야 함
        overlap = int(size / 4) # 기본 오버랩 (예: 25%)

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)

        if end >= text_len:
            break

        start += (size - overlap)
        # 오버랩으로 인해 start가 이전 end보다 커질 수 있으나, 다음 루프에서 처리
        if start >= text_len and text_len > (size-overlap) : # 마지막 청크가 너무 짧아지는 것을 방지하기 위한 조정은 아님.
             # 이 조건은 마지막 청크가 생성된 후 루프를 빠져나가는 것을 보장.
             pass


    return [c for c in chunks if c.strip()]


@celery_app.task(name="app.tasks.ingest_pdf", bind=True) # bind=True로 태스크 인스턴스 접근 가능
def ingest_pdf(self, pdf_path: str, document_id: str):
    """
    Celery task to process a PDF file.
    """
    logger.info(f"[{document_id}] Starting PDF ingestion task for: {pdf_path}. Task ID: {self.request.id}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"[{document_id}] Failed to open PDF {pdf_path}: {e}")
        # 실패 상태 업데이트 (Celery의 상태 업데이트 기능 사용 가능)
        self.update_state(state='FAILURE', meta={'error': str(e), 'document_id': document_id})
        return {"status": "Error", "message": f"Failed to open PDF: {str(e)}", "document_id": document_id}

    all_page_data_for_ocr = []
    extracted_images_info = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        current_page_num = page_num + 1
        logger.info(f"[{document_id}] Processing page {current_page_num}/{len(doc)}")

        # 페이지 내 실제 이미지 객체 추출
        page_images_bytes = extract_images_from_page(page)
        if page_images_bytes:
            for i, img_bytes in enumerate(page_images_bytes):
                all_page_data_for_ocr.append({
                    "type": "extracted_image_object",
                    "page_num": current_page_num,
                    "item_index_in_page": i,
                    "image_bytes": img_bytes
                })
            extracted_images_info.append({
                "page": current_page_num,
                "images_found": len(page_images_bytes),
                "method": "extract_from_page_objects"
            })
            logger.info(f"[{document_id}] Page {current_page_num}: Found {len(page_images_bytes)} image objects.")

        # 이미지가 없거나, 추가적으로 페이지 전체 텍스트도 OCR 하고 싶다면 페이지 렌더링
        # 여기서는 우선 추출된 이미지만 처리하고, 필요시 페이지 렌더링 로직 추가
        # 만약 페이지에 추출할 이미지가 없다면, 페이지 전체를 이미지로 렌더링
        if not page_images_bytes:
            try:
                pix = page.get_pixmap(dpi=200) # DPI를 높여 OCR 품질 향상 시도
                img_bytes = pix.tobytes("png")
                all_page_data_for_ocr.append({
                    "type": "rendered_full_page",
                    "page_num": current_page_num,
                    "item_index_in_page": 0,
                    "image_bytes": img_bytes
                })
                extracted_images_info.append({
                    "page": current_page_num,
                    "images_found": 0, # 명시적 이미지는 없음
                    "method": "render_full_page_for_ocr"
                })
                logger.info(f"[{document_id}] Page {current_page_num}: No image objects found, rendered full page for OCR.")
            except Exception as render_err:
                logger.error(f"[{document_id}] Page {current_page_num}: Failed to render page for OCR: {render_err}")


    doc.close()
    if os.path.exists(pdf_path): # 임시 파일 삭제 (필요에 따라)
        # os.remove(pdf_path)
        # logger.info(f"[{document_id}] Temporary PDF file {pdf_path} removed.")
        pass

    logger.info(f"[{document_id}] Finished extracting images/pages for OCR from {pdf_path}")
    logger.info(f"[{document_id}] Extracted image summary: {extracted_images_info}")
    logger.info(f"[{document_id}] Total items for OCR: {len(all_page_data_for_ocr)}")

    from app.utils.ocr import paddle_ocr # PaddleOCR 함수 임포트

    texts_from_ocr_results = [] # {"page_num": ..., "text": "...", "type": "...", "item_index": ...}
    ocr_success_count = 0
    ocr_failure_count = 0

    if not all_page_data_for_ocr:
        logger.warning(f"[{document_id}] No items found to perform OCR on.")
    else:
        logger.info(f"[{document_id}] Starting OCR for {len(all_page_data_for_ocr)} items...")
        for idx, item_data in enumerate(all_page_data_for_ocr):
            self.update_state(state='PROGRESS', meta={
                'document_id': document_id,
                'progress': f"OCR processing item {idx+1}/{len(all_page_data_for_ocr)}",
                'page_num': item_data.get("page_num")
            })
            logger.debug(f"[{document_id}] Performing OCR on item {idx+1} (Page {item_data.get('page_num')}, Type: {item_data.get('type')})")

            ocr_text = paddle_ocr(item_data["image_bytes"])

            if ocr_text and ocr_text.strip():
                texts_from_ocr_results.append({
                    "page_num": item_data["page_num"],
                    "text": ocr_text,
                    "type": item_data.get("type"), # extracted_image_object or rendered_full_page
                    "item_index_in_page": item_data.get("item_index_in_page")
                })
                ocr_success_count +=1
                logger.debug(f"[{document_id}] OCR successful for item {idx+1}. Text length: {len(ocr_text)}")
            else:
                ocr_failure_count +=1
                logger.warning(f"[{document_id}] OCR for item {idx+1} (Page {item_data.get('page_num')}) yielded no text or failed.")

    logger.info(f"[{document_id}] OCR step completed. Success: {ocr_success_count}, Failure/Empty: {ocr_failure_count} out of {len(all_page_data_for_ocr)} items.")

    if not texts_from_ocr_results:
        logger.warning(f"[{document_id}] No text was extracted from any page/image after OCR.")
        # 이 경우, 처리를 중단하거나 빈 결과로 진행할 수 있음
        # 여기서는 빈 결과로 진행하여 청킹 단계에서 빈 리스트가 나오도록 함

    # 페이지 번호 순서대로 텍스트를 합치거나, 페이지 정보를 유지하며 청킹할 수 있음
    # 여기서는 모든 텍스트를 합침. 페이지 정보를 유지하려면 구조 변경 필요.
    combined_text_for_chunking = " ".join([res["text"] for res in texts_from_ocr_results if res["text"] and res["text"].strip()])
    cleaned_document_text = clean_text(combined_text_for_chunking)
    logger.info(f"[{document_id}] Text cleaning completed. Total length: {len(cleaned_document_text)} chars.")

    chunks = make_chunks(cleaned_document_text, size=230, overlap=60)
    logger.info(f"[{document_id}] Text chunking completed. Got {len(chunks)} chunks.")

    # --- 임베딩 및 저장 단계 시작 ---
    final_status = "Processing" # 최종 상태 추적용

    if not chunks:
        logger.warning(f"[{document_id}] No chunks to embed and store.")
        final_status = "Completed (No content to process)"
    else:
        logger.info(f"[{document_id}] Starting embedding and storage for {len(chunks)} chunks.")
        self.update_state(state='PROGRESS', meta={
            'document_id': document_id,
            'progress': f"Embedding {len(chunks)} chunks..."
        })

        try:
            from app.utils.embedding import get_embed_model
            import faiss
            from tinydb import TinyDB
            import numpy as np

            # 경로 설정 (main.py와 일관성 유지 또는 config 모듈 통해 공유)
            FAISS_DATA_DIR_TASK = os.getenv("FAISS_DATA_PATH_CONTAINER", "/faiss_data")
            FAISS_INDEX_PATH_TASK = os.path.join(FAISS_DATA_DIR_TASK, "hnsw_index.faiss")
            TINYDB_PATH_TASK = os.path.join(FAISS_DATA_DIR_TASK, "metadata_db.json")
            EMBEDDING_DIMENSION_TASK = 768

            embed_model = get_embed_model()
            if embed_model is None:
                raise RuntimeError("Embedding model could not be loaded in Celery task.")

            embeddings_np = embed_model.encode(chunks, batch_size=32, show_progress_bar=False)
            if embeddings_np.dtype != np.float32:
                embeddings_np = embeddings_np.astype(np.float32)
            logger.info(f"[{document_id}] Embedded {len(chunks)} chunks. Shape: {embeddings_np.shape}")

            os.makedirs(FAISS_DATA_DIR_TASK, exist_ok=True)
            current_faiss_index = None
            if os.path.exists(FAISS_INDEX_PATH_TASK):
                try:
                    current_faiss_index = faiss.read_index(FAISS_INDEX_PATH_TASK)
                    logger.info(f"[{document_id}] Loaded FAISS index. Ntotal: {current_faiss_index.ntotal}")
                except Exception as e_read:
                    logger.error(f"[{document_id}] Error reading FAISS index: {e_read}. Creating new.", exc_info=True)

            if current_faiss_index is None:
                current_faiss_index = faiss.IndexHNSWFlat(EMBEDDING_DIMENSION_TASK, 32)
                logger.info(f"[{document_id}] Created new FAISS IndexHNSWFlat.")

            current_faiss_index.add(embeddings_np)
            logger.info(f"[{document_id}] Added {embeddings_np.shape[0]} vectors. New ntotal: {current_faiss_index.ntotal}")
            faiss.write_index(current_faiss_index, FAISS_INDEX_PATH_TASK)
            logger.info(f"[{document_id}] FAISS index saved to {FAISS_INDEX_PATH_TASK}.")

            meta_records = []
            # 페이지 정보 매핑 개선 필요 (현재는 부정확할 수 있음)
            # texts_from_ocr_results의 각 항목이 페이지 정보를 포함하고, 청킹 시 이 정보가 유지되어야 함.
            # 여기서는 임시로 각 청크가 어떤 페이지에서 왔는지 알 수 없다고 가정하고 페이지 정보는 None으로 처리.
            # 또는, 모든 청크를 첫 페이지 또는 문서 전체로 귀속시킬 수 있음.
            # 가장 간단하게는, 청크 순서대로 페이지 번호를 할당 (만약 OCR 결과가 페이지별 텍스트의 리스트라면)
            # 현재 combined_text_for_chunking으로 합쳐서 청킹하므로, 페이지 정보가 유실됨.
            # 정확한 페이지 매핑은 추후 개선 과제.
            for i, chunk_text in enumerate(chunks):
                meta_records.append({
                    "document_id": document_id,
                    "chunk_index_in_doc": i,
                    "text": chunk_text,
                    # "page_num": None # 페이지 정보 추적 개선 필요
                })

            if meta_records:
                meta_db = TinyDB(TINYDB_PATH_TASK)
                # document_id 기준으로 기존 메타데이터 삭제 후 새로 삽입 (멱등성 확보)
                # 또는 청크별 고유 ID를 생성하여 관리
                # 여기서는 단순 append. 중복 처리는 검색 시 또는 별도 정리 로직에서.
                # 더 나은 방법: 문서 단위로 기존 청크 삭제 후 삽입
                Chunk = Query()
                meta_db.remove(Chunk.document_id == document_id) # 해당 문서의 기존 청크 삭제
                meta_db.insert_multiple(meta_records)
                meta_db.close()
                logger.info(f"[{document_id}] Upserted {len(meta_records)} records into TinyDB for document.")

            final_status = "Completed"
            logger.info(f"[{document_id}] Embedding and storage completed successfully.")

        except Exception as e:
            logger.error(f"[{document_id}] Error during embedding or storage: {e}", exc_info=True)
            self.update_state(state='FAILURE', meta={'error': str(e), 'document_id': document_id, 'step': 'embedding_storage'})
            # result_summary를 여기서 반환하여 에러 상태 명시
            summary_on_error = {
                "document_id": document_id, "pdf_path": pdf_path, "status": "Error in Embedding/Storage",
                "task_id": self.request.id, "error_message": str(e)
            }
            return summary_on_error

    # 최종 결과 요약 업데이트
    result_summary = {
        "document_id": document_id,
        "pdf_path": pdf_path,
        "status": final_status, # 최종 처리 상태
        "task_id": self.request.id,
        "total_items_for_ocr": len(all_page_data_for_ocr),
        "total_ocr_results": len(texts_from_ocr_results), # texts_from_ocr_results는 OCR 단계에서 정의됨
        "cleaned_text_length": len(cleaned_document_text), # cleaned_document_text는 이전 단계에서 정의됨
        "num_chunks": len(chunks),
        "faiss_index_path": FAISS_INDEX_PATH_TASK if final_status == "Completed" else None, # 성공 시에만 경로 정보 제공
        "tinydb_path": TINYDB_PATH_TASK if final_status == "Completed" else None
        # "extracted_content_summary": extracted_images_info # extracted_images_info는 이전 단계에서 정의됨
    }
    # extracted_images_info 변수가 정의되지 않았을 수 있으므로, 조건부로 추가하거나 이전 단계에서 확실히 전달되도록 해야함.
    # 여기서는 일단 제외하고, 필요시 추가.
    if 'extracted_images_info' in locals() or 'extracted_images_info' in globals():
        result_summary["extracted_content_summary"] = extracted_images_info


    logger.info(f"[{document_id}] PDF ingestion task final summary: {result_summary}")

    if final_status == "Completed" or final_status == "Completed (No content to process)":
        self.update_state(state='SUCCESS', meta=result_summary)
    # 에러는 이미 try-except 블록에서 FAILURE로 처리하고 반환됨.

    return result_summary
