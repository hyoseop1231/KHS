import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os

# Tesseract 경로 설정 (필요한 경우)
# Windows 사용자의 경우, Tesseract 설치 경로를 명시해야 할 수 있습니다.
# 예: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Linux/macOS에서는 보통 PATH에 자동으로 잡히지만, 문제가 발생하면 해당 경로를 지정해주세요.
# Docker 환경에서는 Dockerfile에 Tesseract 설치 및 PATH 설정을 포함해야 합니다.

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from all pages of a PDF file using OCR.
    Uses PyMuPDF to convert PDF pages to images, and Pytesseract for OCR.
    Attempts to use Korean and English for OCR.
    """
    full_text = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF file {pdf_path}: {e}")
        return f"Error opening PDF: {e}"

    for page_num in range(len(doc)):
        try:
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=300)  # Higher DPI for better OCR quality

            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))

            # OCR 수행 (한국어 + 영어)
            # 사용자가 Tesseract용 한국어 데이터('kor')를 설치했다고 가정합니다.
            # 설치 명령어는 README.md에 포함되어 있습니다.
            try:
                text = pytesseract.image_to_string(img, lang='kor+eng')
                full_text.append(text)
            except pytesseract.TesseractNotFoundError:
                error_msg = "Tesseract is not installed or not in your PATH. Please install Tesseract and try again."
                print(error_msg)
                # 애플리케이션 전체에서 이 오류를 더 잘 처리할 수 있도록 예외를 다시 발생시킬 수 있습니다.
                # 여기서는 일단 오류 메시지를 반환합니다.
                return f"OCR Error: {error_msg}"
            except Exception as ocr_error: # Other Tesseract errors
                print(f"Error during OCR on page {page_num + 1}: {ocr_error}")
                full_text.append(f"[OCR Error on page {page_num + 1}]")

        except Exception as page_error:
            print(f"Error processing page {page_num + 1} of {pdf_path}: {page_error}")
            full_text.append(f"[Error processing page {page_num + 1}]")

    doc.close()

    return "\n".join(full_text)

if __name__ == '__main__':
    # 간단한 테스트용 (실제 운영에서는 이 부분을 사용하지 않음)
    # 테스트용 PDF 파일을 생성하거나 경로를 지정해야 합니다.
    # 예시: test_pdf_path = "path/to/your/test.pdf"
    # if os.path.exists(test_pdf_path):
    #     print(f"Testing OCR with: {test_pdf_path}")
    #     extracted_text = extract_text_from_pdf(test_pdf_path)
    #     print("\n--- Extracted Text ---")
    #     print(extracted_text[:500]) # 처음 500자만 출력
    #     print("\n--- End of Text ---")
    # else:
    #     print(f"Test PDF file not found at: {test_pdf_path}")

    # 사용자가 Tesseract 설치 및 kor 데이터 파일 설치를 확인하도록 안내합니다.
    print("OCR service module loaded.")
    print("Ensure Tesseract OCR is installed and 'kor' language data is available.")
    print("For Windows, you might need to set 'pytesseract.tesseract_cmd'.")
    print("Refer to README.md for Tesseract installation instructions.")
