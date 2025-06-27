from paddleocr import PaddleOCR
import logging
import os

logger = logging.getLogger(__name__)

# PaddleOCR 초기화
# use_gpu=False를 명시적으로 설정하여 CPU 사용을 보장합니다.
# lang="korean" 대신 "ko"를 사용해야 할 수 있습니다. PaddleOCR 문서 확인 필요.
# 설계서에는 "korean"으로 되어 있으나, 실제 라이브러리에서는 "ko"를 사용할 수 있습니다.
# 일단 "korean"으로 시도하고, 문제 발생 시 "ko"로 변경합니다.
# show_log=False는 PaddleOCR 자체의 로그를 줄여줍니다.
try:
    # CPU를 사용하도록 명시적으로 설정합니다.
    # det=True, rec=True, cls=True는 기본값이지만 명시적으로 지정 가능
    ocr_instance = PaddleOCR(use_angle_cls=True, lang="korean", use_gpu=False, show_log=False)
    logger.info("PaddleOCR instance initialized successfully (lang='korean', use_gpu=False).")
except Exception as e:
    logger.error(f"Failed to initialize PaddleOCR with lang='korean': {e}", exc_info=True)
    try:
        logger.info("Attempting to initialize PaddleOCR with lang='ko'.")
        ocr_instance = PaddleOCR(use_angle_cls=True, lang="ko", use_gpu=False, show_log=False)
        logger.info("PaddleOCR instance initialized successfully (lang='ko', use_gpu=False).")
    except Exception as e_ko:
        logger.error(f"Failed to initialize PaddleOCR with lang='ko' as well: {e_ko}", exc_info=True)
        ocr_instance = None # 초기화 실패 시 None으로 설정

def paddle_ocr(img_bytes: bytes) -> str:
    """
    Performs OCR on the given image bytes using PaddleOCR.

    Args:
        img_bytes: Bytes of the image to perform OCR on.

    Returns:
        The extracted text as a single string, or an empty string if OCR fails or no text is found.
    """
    if ocr_instance is None:
        logger.error("PaddleOCR instance is not available. OCR cannot be performed.")
        return ""

    if not img_bytes:
        logger.warning("Received empty image bytes for OCR.")
        return ""

    try:
        # result = ocr_instance.ocr(img_bytes, cls=True) # img_bytes를 직접 전달
        # PaddleOCR은 이미지 경로 또는 numpy 배열을 주로 받습니다. 바이트를 직접 지원하는지 확인.
        # 대부분의 OCR 라이브러리는 파일 경로 또는 OpenCV 이미지 객체(numpy array)를 선호합니다.
        # 만약 바이트를 직접 지원하지 않는다면, 임시 파일로 저장하거나 numpy 배열로 변환해야 합니다.
        # PaddleOCR 문서에 따르면 numpy array를 지원합니다.
        import numpy as np
        import cv2

        # 바이트 데이터를 numpy 배열로 디코딩
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # OpenCV로 이미지 디코딩

        if img_np is None:
            logger.error("Failed to decode image bytes using OpenCV.")
            return ""

        result = ocr_instance.ocr(img_np, cls=True) # numpy 배열 전달

        if result and result[0] is not None: # 최신 버전의 PaddleOCR은 결과가 [[lines]] 형태일 수 있음
            # result 구조: [[box, (text, confidence)], ...] for each detected text line
            # 또는 [[[box, (text, confidence)], ...]] for newer versions

            # 결과가 비어있거나 None일 수 있음
            if not result[0]: # result = [None] or [[]]
                 logger.info("OCR result is empty or None.")
                 return ""

            # 각 라인에서 텍스트만 추출하여 합침
            # PaddleOCR의 결과는 보통 리스트의 리스트 형태입니다.
            # 예: [[[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ('text', score)], ...]
            # 또는 [[(text, score), (text, score), ...]] for each line (older versions)
            # 설계서: " ".join([word for line in res for word, _ in line])
            # 위 코드는 res가 [[('word1', score), ('word2', score)], [('word3', score)]] 와 같은 구조를 가정.
            # 실제 PaddleOCR 결과 구조에 맞춰 파싱해야 합니다.

            lines = result[0]
            extracted_texts = []
            for line_info in lines:
                # line_info는 [ [[좌표]], (텍스트, 신뢰도) ] 형태
                if line_info and len(line_info) == 2:
                    text_part = line_info[1] # (텍스트, 신뢰도)
                    if isinstance(text_part, tuple) and len(text_part) == 2:
                        extracted_texts.append(text_part[0])
                elif isinstance(line_info, list) and line_info: # 다른 가능한 구조 처리
                    # 예: line_info가 그냥 [ (텍스트, 신뢰도) ] 일 경우
                    if isinstance(line_info[0], tuple) and len(line_info[0]) == 2:
                         extracted_texts.append(line_info[0][0])


            return " ".join(filter(None, extracted_texts)) # None이나 빈 문자열 필터링
        else:
            logger.info("OCR process returned no text or an unexpected result structure.")
            return ""

    except Exception as e:
        logger.error(f"Error during PaddleOCR process: {e}", exc_info=True)
        return ""

if __name__ == '__main__':
    # 간단한 테스트 (실제 이미지 파일 필요)
    logger.info("Testing PaddleOCR module...")
    if ocr_instance:
        logger.info("PaddleOCR instance available.")
        # 예시: 빈 이미지 바이트로 테스트
        # test_img_bytes = b''
        # try:
        #     # 실제 이미지 파일로 테스트해야 의미가 있음
        #     # with open("path/to/your/test_image.png", "rb") as f:
        #     #    test_img_bytes = f.read()
        #     # if test_img_bytes:
        #     #    ocr_text = paddle_ocr(test_img_bytes)
        #     #    logger.info(f"Test OCR result: '{ocr_text}'")
        #     # else:
        #     #    logger.warning("Test image file not found or empty.")
        #     logger.info("To test paddle_ocr function, provide a valid image byte stream.")
        # except Exception as e:
        #     logger.error(f"Test failed: {e}")
        pass
    else:
        logger.error("PaddleOCR instance not available, cannot run test.")
