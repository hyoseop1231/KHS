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
from app.services.text_processing_service import correct_foundry_terms
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

def extract_multimodal_content_from_pdf(pdf_path: str, document_id: str) -> Dict[str, Any]:
    """
    Extracts text, images, and tables from PDF file.
    Returns a dictionary containing:
    - text: extracted text content
    - images: list of extracted images with metadata
    - tables: list of extracted tables with data
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
    
    try:
        doc = fitz.open(pdf_path)
        logger.info(f"PDF opened successfully. Total pages: {len(doc)}")
    except Exception as e:
        logger.error(f"Error opening PDF file {pdf_path}: {e}")
        raise FileProcessingError(f"Could not open PDF file: {e}", "PDF_OPEN_ERROR")

    for page_num in range(len(doc)):
        try:
            logger.debug(f"Processing page {page_num + 1}/{len(doc)}")
            page = doc.load_page(page_num)
            
            # Extract text using OCR
            pix = page.get_pixmap(dpi=settings.OCR_DPI)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            
            try:
                text = pytesseract.image_to_string(img, lang=settings.OCR_LANGUAGES)
                text = correct_foundry_terms(text)
                full_text.append(text)
                logger.debug(f"Page {page_num + 1} OCR completed. Text length: {len(text)} chars")
                
            except pytesseract.TesseractNotFoundError:
                error_msg = "Tesseract is not installed or not in your PATH. Please install Tesseract and try again."
                logger.error(error_msg)
                raise OCRError(error_msg, "TESSERACT_NOT_FOUND")
            except Exception as ocr_error:
                logger.warning(f"OCR error on page {page_num + 1}: {ocr_error}")
                full_text.append(f"[OCR Error on page {page_num + 1}]")
            
            # Image extraction temporarily disabled for stability
            
            # Extract tables from page
            page_tables = extract_tables_from_page(img, page_num, tables_dir, document_id)
            extracted_tables.extend(page_tables)
            
        except Exception as page_error:
            logger.warning(f"Error processing page {page_num + 1}: {page_error}")
            full_text.append(f"[Error processing page {page_num + 1}]")

    doc.close()
    extracted_text = "\n".join(full_text)
    
    logger.info(f"Multimodal extraction completed:")
    logger.info(f"  - Text length: {len(extracted_text)} chars")
    logger.info(f"  - Images extracted: {len(extracted_images)}")
    logger.info(f"  - Tables extracted: {len(extracted_tables)}")
    
    return {
        "text": extracted_text,
        "images": extracted_images,
        "tables": extracted_tables,
        "content_dir": content_dir
    }

# Image extraction temporarily removed for stability

def extract_tables_from_page(page_image: Image.Image, page_num: int, tables_dir: str, document_id: str) -> List[Dict[str, Any]]:
    """
    Extract tables from a PDF page using image processing and OCR.
    """
    tables = []
    
    try:
        # Convert PIL image to OpenCV format
        cv_image = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect table-like structures using contours
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        table_index = 0
        for contour in contours:
            # Filter contours by area (potential tables should be reasonably large)
            area = cv2.contourArea(contour)
            if area > 5000:  # Minimum area threshold for tables
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract table region
                table_region = cv_image[y:y+h, x:x+w]
                
                # Save table image
                table_filename = f"{document_id}_page_{page_num+1}_table_{table_index+1}.png"
                table_path = os.path.join(tables_dir, table_filename)
                cv2.imwrite(table_path, table_region)
                
                # Try to extract table data using OCR
                try:
                    table_pil = Image.fromarray(cv2.cvtColor(table_region, cv2.COLOR_BGR2RGB))
                    table_text = pytesseract.image_to_string(table_pil, lang=settings.OCR_LANGUAGES)
                    
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

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Legacy function for backward compatibility.
    Extracts only text from PDF (no images or tables).
    """
    logger.info(f"Using legacy text-only extraction for: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        raise FileProcessingError(f"PDF file not found: {pdf_path}", "FILE_NOT_FOUND")
    
    full_text = []
    
    try:
        doc = fitz.open(pdf_path)
        logger.info(f"PDF opened successfully. Total pages: {len(doc)}")
    except Exception as e:
        logger.error(f"Error opening PDF file {pdf_path}: {e}")
        raise FileProcessingError(f"Could not open PDF file: {e}", "PDF_OPEN_ERROR")

    for page_num in range(len(doc)):
        try:
            logger.debug(f"Processing page {page_num + 1}/{len(doc)}")
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=settings.OCR_DPI)

            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))

            try:
                text = pytesseract.image_to_string(img, lang=settings.OCR_LANGUAGES)
                text = correct_foundry_terms(text)
                full_text.append(text)
                logger.debug(f"Page {page_num + 1} OCR completed. Text length: {len(text)} chars")
                
            except pytesseract.TesseractNotFoundError:
                error_msg = "Tesseract is not installed or not in your PATH. Please install Tesseract and try again."
                logger.error(error_msg)
                raise OCRError(error_msg, "TESSERACT_NOT_FOUND")
            except Exception as ocr_error:
                logger.warning(f"OCR error on page {page_num + 1}: {ocr_error}")
                full_text.append(f"[OCR Error on page {page_num + 1}]")

        except Exception as page_error:
            logger.warning(f"Error processing page {page_num + 1}: {page_error}")
            full_text.append(f"[Error processing page {page_num + 1}]")

    doc.close()
    extracted_text = "\n".join(full_text)
    logger.info(f"OCR processing completed. Total extracted text length: {len(extracted_text)} chars")
    
    return extracted_text

if __name__ == '__main__':
    print("Multimodal OCR service module loaded.")
    print("Ensure Tesseract OCR is installed and 'kor' language data is available.")
    print("OpenCV and pandas are required for image and table extraction.")
    print("For Windows, you might need to set 'pytesseract.tesseract_cmd'.")
    print("Refer to README.md for installation instructions.")
