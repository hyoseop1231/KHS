import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os
import base64
import json
from typing import List, Dict, Any, Tuple
import pandas as pd
import cv2
import numpy as np
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
from app.services.term_correction_service import correct_foundry_terms
from app.services.ocr_correction_service import correct_ocr_text
from app.config import settings
from app.utils.logging_config import get_logger
from app.utils.exceptions import OCRError, FileProcessingError

logger = get_logger(__name__)

# Ensure required directories exist
def ensure_content_directories():
    """Create necessary directories for storing extracted content."""
    if not os.path.exists(settings.UPLOAD_DIR):
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        logger.info(f"Created upload directory: {settings.UPLOAD_DIR}")

ensure_content_directories()

# OCR 병렬 처리를 위한 글로벌 설정
MAX_WORKERS = min(8, mp.cpu_count())  # CPU 코어 수에 따라 조정
OCR_BATCH_SIZE = max(1, MAX_WORKERS // 2)  # 배치 크기 최적화

def process_page_ocr_simple(pdf_path: str, page_num: int, document_id: str) -> Dict[str, Any]:
    """
    개별 페이지 OCR 처리 (Thread-safe)
    """
    try:
        # 각 스레드에서 독립적으로 PDF 열기
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        
        # OCR 처리
        pix = page.get_pixmap(dpi=settings.OCR_DPI)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        
        # 텍스트 추출
        text = pytesseract.image_to_string(img, lang=settings.OCR_LANGUAGES)
        text = correct_foundry_terms(text)
        
        # 기본 표 추출 (간단한 버전)
        tables = []
        
        doc.close()
        
        return {
            'page_num': page_num + 1,
            'text': text,
            'tables': tables,
            'status': 'success',
            'text_length': len(text),
            'tables_extracted': len(tables)
        }
        
    except Exception as e:
        logger.error(f"Error processing page {page_num + 1}: {e}")
        return {
            'page_num': page_num + 1,
            'text': '',
            'tables': [],
            'status': 'error',
            'error': str(e),
            'text_length': 0,
            'tables_extracted': 0
        }


def extract_images_from_page(page: fitz.Page, images_dir: str, document_id: str, page_num: int) -> List[Dict[str, Any]]:
    """
    Extracts images from a single PDF page.
    """
    images = []
    img_list = page.get_images(full=True)
    for img_index, img_info in enumerate(img_list):
        xref = img_info[0]
        base_image = page.parent.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]

        # Generate a unique filename for the image
        image_filename = f"{document_id}_page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
        image_path = os.path.join(images_dir, image_filename)

        try:
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            
            images.append({
                "filename": image_filename,
                "path": image_path,
                "page": page_num + 1,
                "index": img_index + 1,
                "size_bytes": len(image_bytes)
            })
            logger.debug(f"Extracted image: {image_filename}")
        except Exception as e:
            logger.warning(f"Could not save image {image_filename}: {e}")
    return images

def extract_multimodal_content_from_pdf(
    pdf_path: str,
    document_id: str,
    ocr_correction_enabled: bool = None,
    llm_correction_enabled: bool = None,
    progress_callback: callable = None,
    max_memory_mb: int = 512
) -> Dict[str, Any]:
    """
    Extracts text, images, and tables from PDF file.
    Returns a dictionary containing:
    - text: extracted text content
    - images: list of extracted images with metadata
    - tables: list of extracted tables with data
    - content_dir: directory where extracted content is stored
    - total_pages: total number of pages in the PDF
    - processed_pages: number of pages successfully processed
    - failed_pages: number of pages that failed processing
    - page_results: list of results for each page
    """
    if not os.path.exists(pdf_path):
        raise FileProcessingError(f"PDF file not found: {pdf_path}", "FILE_NOT_FOUND")
    
    logger.info(f"Starting multimodal content extraction for PDF: {pdf_path}")
    
    # Create directories for storing extracted content
    content_dir = os.path.join(settings.UPLOAD_DIR, f"{document_id}_content")
    images_dir = os.path.join(content_dir, "images")
    tables_dir = os.path.join(content_dir, "tables")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    
    full_text = []
    extracted_images = []
    extracted_tables = []
    page_results = []
    processed_pages_count = 0
    failed_pages_count = 0
    
    # Memory optimization: process in batches based on memory limit
    import psutil
    available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
    batch_size = min(10, max(1, int(max_memory_mb / 50)))  # Adjust batch size based on memory
    logger.info(f"Processing PDF in batches of {batch_size} pages (Available memory: {available_memory_mb:.0f}MB)")
    
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        logger.info(f"PDF opened successfully. Total pages: {total_pages}")
    except Exception as e:
        logger.error(f"Error opening PDF file {pdf_path}: {e}")
        raise FileProcessingError(f"Could not open PDF file: {e}", "PDF_OPEN_ERROR")

    # Process pages in batches to optimize memory usage
    for batch_start in range(0, total_pages, batch_size):
        batch_end = min(batch_start + batch_size, total_pages)
        logger.debug(f"Processing batch {batch_start + 1}-{batch_end}/{total_pages}")
        
        # Process current batch
        batch_text = []
        batch_images = []
        batch_tables = []
        
        for page_num in range(batch_start, batch_end):
            page_result = {"page_num": page_num + 1, "status": "success", "error": None, "text_length": 0, "images_extracted": 0, "tables_extracted": 0}
            try:
                logger.debug(f"Processing page {page_num + 1}/{total_pages}")
                page = doc.load_page(page_num)

                # 1. Extract text
                if progress_callback:
                    progress_callback(page_num + 1, total_pages, "text", f"텍스트 추출 중... ({page_num + 1}/{total_pages})")
                
                text = page.get_text()
                if not text.strip() or settings.OCR_FORCE_OCR: # Fallback to OCR if no text or force OCR
                    pix = page.get_pixmap(dpi=settings.OCR_DPI)
                    img_bytes = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_bytes))
                    text = pytesseract.image_to_string(img, lang=settings.OCR_LANGUAGES)
                    # Free memory immediately
                    del img_bytes, img
                
                text = correct_foundry_terms(text)
                batch_text.append(text)
                page_result["text_length"] = len(text)

                # 2. Extract images
                if progress_callback:
                    progress_callback(page_num + 1, total_pages, "images", f"이미지 추출 중... ({page_num + 1}/{total_pages})")
                
                page_images = extract_images_from_page(page, images_dir, document_id, page_num)
                batch_images.extend(page_images)
                page_result["images_extracted"] = len(page_images)

                # 3. Extract tables
                if progress_callback:
                    progress_callback(page_num + 1, total_pages, "tables", f"표 추출 중... ({page_num + 1}/{total_pages})")
                
                # Render page to image for table detection
                pix = page.get_pixmap(dpi=settings.OCR_DPI)
                img_bytes = pix.tobytes("png")
                page_image = Image.open(io.BytesIO(img_bytes))
                
                page_tables = extract_tables_from_page(page_image, page_num, tables_dir, document_id, progress_callback, total_pages)
                batch_tables.extend(page_tables)
                page_result["tables_extracted"] = len(page_tables)
                
                # Free memory immediately after processing each page
                del pix, img_bytes, page_image
                
                processed_pages_count += 1

            except pytesseract.TesseractNotFoundError:
                error_msg = "Tesseract is not installed or not in your PATH. Please install Tesseract and try again."
                logger.error(error_msg)
                page_result["status"] = "failed"
                page_result["error"] = error_msg
                batch_text.append(f"[OCR Error on page {page_num + 1}: {error_msg}]")
                failed_pages_count += 1
            except Exception as page_error:
                error_msg = f"Error processing page {page_num + 1}: {page_error}"
                logger.warning(error_msg)
                page_result["status"] = "failed"
                page_result["error"] = error_msg
                batch_text.append(f"[Error processing page {page_num + 1}: {page_error}]")
                failed_pages_count += 1
            finally:
                page_results.append(page_result)
        
        # Add batch results to main collections
        full_text.extend(batch_text)
        extracted_images.extend(batch_images)
        extracted_tables.extend(batch_tables)
        
        # Force garbage collection after each batch
        import gc
        gc.collect()
        
        logger.debug(f"Completed batch {batch_start + 1}-{batch_end}, memory freed")

    doc.close()
    extracted_text = "\n".join(full_text)
    # Determine correction flags (override settings if provided)
    ocr_enabled = (
        ocr_correction_enabled if ocr_correction_enabled is not None
        else settings.OCR_CORRECTION_ENABLED
    )
    llm_enabled = (
        llm_correction_enabled if llm_correction_enabled is not None
        else settings.OCR_CORRECTION_USE_LLM
    )
    # OCR 교정 최적화: 전체 텍스트에 대해 배치 단위로 LLM 기반 교정 수행
    if ocr_enabled and llm_enabled:
        try:
            # 큰 텍스트를 청크로 나누어 배치 교정
            from app.services.ocr_correction_service import correct_text_in_batches
            corrected_full = correct_text_in_batches(
                extracted_text,
                batch_size=5000,  # 5000자 단위로 처리
                use_llm=True
            )
            extracted_text = corrected_full
            logger.info(f"[Task {document_id}] 배치 OCR 교정 완료 (LLM: True)")
        except Exception as e:
            logger.warning(f"[Task {document_id}] 배치 OCR 교정 실패: {e}")
            # Fallback to pattern-based correction
            try:
                corrected_full = correct_ocr_text(extracted_text, use_llm=False)
                extracted_text = corrected_full
                logger.info(f"[Task {document_id}] 패턴 기반 OCR 교정 완료")
            except Exception as fallback_error:
                logger.warning(f"[Task {document_id}] 패턴 기반 OCR 교정도 실패: {fallback_error}")
    elif ocr_enabled:
        # Pattern-based correction only
        try:
            corrected_full = correct_ocr_text(extracted_text, use_llm=False)
            extracted_text = corrected_full
            logger.info(f"[Task {document_id}] 패턴 기반 OCR 교정 완료")
        except Exception as e:
            logger.warning(f"[Task {document_id}] 패턴 기반 OCR 교정 실패: {e}")
    
    logger.info(f"Multimodal extraction completed for {pdf_path}:")
    logger.info(f"  - Total pages: {total_pages}")
    logger.info(f"  - Successfully processed pages: {processed_pages_count}")
    logger.info(f"  - Failed pages: {failed_pages_count}")
    logger.info(f"  - Total text length: {len(extracted_text)} chars")
    logger.info(f"  - Images extracted: {len(extracted_images)}")
    logger.info(f"  - Tables extracted: {len(extracted_tables)}")
    
    return {
        "text": extracted_text,
        "images": extracted_images,
        "tables": extracted_tables,
        "content_dir": content_dir,
        "total_pages": total_pages,
        "processed_pages": processed_pages_count,
        "failed_pages": failed_pages_count,
        "page_results": page_results
    }

def extract_tables_from_page(page_image: Image.Image, page_num: int, tables_dir: str, document_id: str, progress_callback: callable = None, total_pages: int = 0) -> List[Dict[str, Any]]:
    """
    Extract tables from a PDF page using image processing and OCR.
    """
    tables = []
    
    try:
        # 상세 진행률 업데이트 - 이미지 전처리
        if progress_callback:
            progress_callback(page_num + 1, total_pages, "table_preprocessing", f"표 감지 전처리 중... ({page_num + 1}/{total_pages})")
        
        # Convert PIL image to OpenCV format
        cv_image = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect table-like structures using contours
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # 상세 진행률 업데이트 - 윤곽선 검출
        if progress_callback:
            progress_callback(page_num + 1, total_pages, "table_detection", f"표 구조 분석 중... ({page_num + 1}/{total_pages})")
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        table_index = 0
        total_contours = len([c for c in contours if cv2.contourArea(c) > 5000])
        
        for contour_idx, contour in enumerate(contours):
            # Filter contours by area (potential tables should be reasonably large)
            area = cv2.contourArea(contour)
            if area > 5000:  # Minimum area threshold for tables
                # 상세 진행률 업데이트 - 개별 표 처리
                if progress_callback:
                    progress_callback(page_num + 1, total_pages, "table_processing", 
                                    f"표 {table_index + 1}/{total_contours} 처리 중... ({page_num + 1}/{total_pages})")
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract table region
                table_region = cv_image[y:y+h, x:x+w]
                
                # Save table image
                table_filename = f"{document_id}_page_{page_num+1}_table_{table_index+1}.png"
                table_path = os.path.join(tables_dir, table_filename)
                cv2.imwrite(table_path, table_region)
                
                # 상세 진행률 업데이트 - OCR 처리
                if progress_callback:
                    progress_callback(page_num + 1, total_pages, "table_ocr", 
                                    f"표 {table_index + 1} OCR 중... ({page_num + 1}/{total_pages})")
                
                # Try to extract table data using OCR
                try:
                    table_pil = Image.fromarray(cv2.cvtColor(table_region, cv2.COLOR_BGR2RGB))
                    table_text = pytesseract.image_to_string(table_pil, lang=settings.OCR_LANGUAGES)
                    
                    # OCR 교정은 전체 텍스트에서 한 번만 수행하므로 여기서는 스킵
                    # 패턴 기반 교정만 적용 (주조 전문용어)
                    table_text = correct_foundry_terms(table_text)
                    
                    # Parse table data (simple approach)
                    table_data = parse_table_text(table_text)
                    
                    table_info = {
                        "filename": table_filename,
                        "path": table_path,
                        "page": page_num + 1,
                        "index": table_index + 1,
                        "x": x, "y": y, "width": w, "height": h,
                        "raw_text": table_text.strip(),
                        "parsed_data": table_data,
                        "size_bytes": os.path.getsize(table_path) if os.path.exists(table_path) else 0
                    }
                    
                    tables.append(table_info)
                    table_index += 1
                    logger.debug(f"Extracted table: {table_filename}")
                    
                except Exception as e:
                    logger.warning(f"Error processing table {table_index} on page {page_num + 1}: {e}")
        
    except Exception as e:
        logger.warning(f"Error extracting tables from page {page_num + 1}: {e}")
    
    return tables

def parse_table_text(table_text: str) -> List[List[str]]:
    """
    Simple table text parser. Attempts to structure table data from OCR text.
    """
    if not table_text.strip():
        return []
    
    lines = [line.strip() for line in table_text.split('\n') if line.strip()]
    table_data = []
    
    for line in lines:
        # Try to split by multiple spaces or tabs
        cells = [cell.strip() for cell in line.split('  ') if cell.strip()]
        if not cells:
            # Try splitting by single space if multiple spaces didn't work
            cells = [cell.strip() for cell in line.split(' ') if cell.strip()]
        
        if cells:
            table_data.append(cells)
    
    return table_data

def extract_text_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Extracts only text from PDF (no images or tables).
    Returns a dictionary containing:
    - text: extracted text content
    - total_pages: total number of pages in the PDF
    - processed_pages: number of pages successfully processed
    - failed_pages: number of pages that failed processing
    - page_results: list of results for each page
    """
    logger.info(f"Using text-only extraction for: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        raise FileProcessingError(f"PDF file not found: {pdf_path}", "FILE_NOT_FOUND")
    
    full_text = []
    page_results = []
    processed_pages_count = 0
    failed_pages_count = 0
    
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        logger.info(f"PDF opened successfully. Total pages: {total_pages}")
    except Exception as e:
        logger.error(f"Error opening PDF file {pdf_path}: {e}")
        raise FileProcessingError(f"Could not open PDF file: {e}", "PDF_OPEN_ERROR")

    for page_num in range(total_pages):
        page_result = {"page_num": page_num + 1, "status": "success", "error": None, "text_length": 0}
        try:
            logger.debug(f"Processing page {page_num + 1}/{total_pages}")
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=settings.OCR_DPI)

            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))

            try:
                text = pytesseract.image_to_string(img, lang=settings.OCR_LANGUAGES)
                text = correct_foundry_terms(text)
                full_text.append(text)
                page_result["text_length"] = len(text)
                processed_pages_count += 1
                logger.debug(f"Page {page_num + 1} OCR completed. Text length: {len(text)} chars")
                
            except pytesseract.TesseractNotFoundError:
                error_msg = "Tesseract is not installed or not in your PATH. Please install Tesseract and try again."
                logger.error(error_msg)
                page_result["status"] = "failed"
                page_result["error"] = error_msg
                full_text.append(f"[OCR Error on page {page_num + 1}: {error_msg}]")
                failed_pages_count += 1
            except Exception as ocr_error:
                error_msg = f"OCR error on page {page_num + 1}: {ocr_error}"
                logger.warning(error_msg)
                page_result["status"] = "failed"
                page_result["error"] = error_msg
                full_text.append(f"[OCR Error on page {page_num + 1}: {ocr_error}]")
                failed_pages_count += 1

        except Exception as page_error:
            error_msg = f"Error processing page {page_num + 1}: {page_error}"
            logger.warning(error_msg)
            page_result["status"] = "failed"
            page_result["error"] = error_msg
            full_text.append(f"[Error processing page {page_num + 1}: {page_error}]")
            failed_pages_count += 1
        finally:
            page_results.append(page_result)

    doc.close()
    extracted_text = "\n".join(full_text)
    logger.info(f"OCR processing completed. Total extracted text length: {len(extracted_text)} chars")
    
    return {
        "text": extracted_text,
        "total_pages": total_pages,
        "processed_pages": processed_pages_count,
        "failed_pages": failed_pages_count,
        "page_results": page_results
    }

if __name__ == '__main__':
    print("Multimodal OCR service module loaded.")
    print("Ensure Tesseract OCR is installed and 'kor' language data is available.")
    print("OpenCV and pandas are required for image and table extraction.")
    print("For Windows, you might need to set 'pytesseract.tesseract_cmd'.")
    print("Refer to README.md for installation instructions.")