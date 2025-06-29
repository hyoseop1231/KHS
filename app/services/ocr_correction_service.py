import json
import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.services.llm_service import get_llm_response
from app.config import settings
from app.utils.logging_config import get_logger
from app.utils.exceptions import LLMError

logger = get_logger(__name__)

class OCRCorrectionService:
    """
    OCR 추출 텍스트를 주조 용어집 기반으로 LLM을 통해 교정하는 서비스
    """
    
    def __init__(self):
        self.terminology_dict = self._load_terminology_dict()
        self.all_terms = self._extract_all_terms()
        self.ocr_error_patterns = self._load_ocr_error_patterns()
        
    def _load_terminology_dict(self) -> Dict[str, Any]:
        """주조 용어집 사전 로드"""
        try:
            terminology_path = Path(__file__).parent.parent / "data" / "foundry_terminology.json"
            with open(terminology_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load foundry terminology: {e}")
            return {"categories": {}, "common_ocr_errors": {}}
    
    def _extract_all_terms(self) -> List[str]:
        """모든 주조 용어를 추출"""
        all_terms = []
        for category in self.terminology_dict.get("categories", {}).values():
            all_terms.extend(category.get("terms", []))
        return list(set(all_terms))  # 중복 제거
    
    def _load_ocr_error_patterns(self) -> Dict[str, List[str]]:
        """OCR 오류 패턴 로드"""
        return self.terminology_dict.get("common_ocr_errors", {})
    
    def _quick_pattern_correction(self, text: str) -> str:
        """
        빠른 패턴 기반 교정 (LLM 호출 전)
        """
        corrected_text = text
        
        # 알려진 OCR 오류 패턴 교정
        for correct_term, error_patterns in self.ocr_error_patterns.items():
            for error_pattern in error_patterns:
                corrected_text = corrected_text.replace(error_pattern, correct_term)
        
        # 일반적인 OCR 오류 패턴 교정
        common_patterns = {
            r'(?<=[가-힣])O(?=[가-힣])': '0',  # 한글 사이의 O를 0으로
            r'(?<=[가-힣])I(?=[가-힣])': '1',  # 한글 사이의 I를 1으로
            r'(?<=[가-힣])l(?=[가-힣])': '1',  # 한글 사이의 l을 1으로
            r'(?<=\d)O(?=\d)': '0',            # 숫자 사이의 O를 0으로
            r'(?<=\d)I(?=\d)': '1',            # 숫자 사이의 I를 1으로
            r'(?<=\d)l(?=\d)': '1',            # 숫자 사이의 l을 1으로
        }
        
        for pattern, replacement in common_patterns.items():
            corrected_text = re.sub(pattern, replacement, corrected_text)
        
        return corrected_text
    
    def _create_correction_prompt(self, text: str) -> str:
        """
        LLM을 위한 교정 프롬프트 생성
        """
        # 카테고리별로 대표 용어들을 선별하여 포함
        foundry_terms_by_category = {}
        for category_key, category_data in self.terminology_dict.get("categories", {}).items():
            category_name = category_data.get("name", "")
            key_terms = category_data.get("terms", [])[:10]  # 각 카테고리에서 처음 10개만
            foundry_terms_by_category[category_name] = key_terms
        
        # 카테고리별 용어 목록 생성
        terms_by_category = []
        for category, terms in foundry_terms_by_category.items():
            terms_by_category.append(f"• {category}: {', '.join(terms)}")
        
        prompt = f"""다음은 주조(Foundry) 기술 문서에서 OCR로 추출한 텍스트입니다. 
OCR 과정에서 발생할 수 있는 오류를 수정하고, 주조 전문 용어를 정확하게 교정해주세요.

**주조 전문 용어 카테고리별 참고:**
{chr(10).join(terms_by_category)}

**중요한 주조 용어 예시:**
- 주형(mold), 사형(sand mold), 금형(metal mold)
- 탕구(sprue), 러너(runner), 게이트(gate), 라이저(riser)
- 코어(core), 패턴(pattern), 플라스크(flask)
- 쿠폴라(cupola), 용해(melting), 응고(solidification)
- 결함(defect), 수축(shrinkage), 기포(porosity)

**수정 규칙:**
1. 주조 전문 용어의 오타나 잘못된 인식을 올바른 표준 용어로 수정
2. 숫자와 문자의 혼동 수정 (O↔0, I↔1, l↔1 등)
3. 불필요한 공백이나 특수문자 제거
4. 문맥상 올바른 한자 표기 사용 (예: 鑄型, 湯口, 砂型 등)
5. 영문 전문 용어의 정확한 표기 확인

원본 텍스트:
{text}

수정된 텍스트만 출력하세요. 추가 설명이나 주석은 하지 마세요."""

        return prompt
    
    def correct_ocr_text(self, text: str, use_llm: bool = True) -> str:
        """
        OCR 텍스트 교정
        
        Args:
            text: OCR로 추출한 원본 텍스트
            use_llm: LLM을 사용한 고급 교정 여부
            
        Returns:
            str: 교정된 텍스트
        """
        if not text or not text.strip():
            return text
        
        # 1단계: 빠른 패턴 기반 교정
        corrected_text = self._quick_pattern_correction(text)
        
        # 2단계: LLM 기반 고급 교정 (선택적)
        if use_llm and len(text.strip()) > 10:  # 너무 짧은 텍스트는 제외
            try:
                corrected_text = self._llm_correction(corrected_text)
            except Exception as e:
                logger.warning(f"LLM correction failed, using pattern-based correction only: {e}")
        
        return corrected_text
    
    def _llm_correction(self, text: str) -> str:
        """
        LLM을 사용한 텍스트 교정
        """
        # 너무 긴 텍스트는 청크로 나누어 처리
        max_chunk_size = 1000
        if len(text) <= max_chunk_size:
            return self._correct_single_chunk(text)
        
        # 긴 텍스트를 청크로 나누어 처리
        chunks = self._split_text_into_chunks(text, max_chunk_size)
        corrected_chunks = []
        
        for chunk in chunks:
            try:
                corrected_chunk = self._correct_single_chunk(chunk)
                corrected_chunks.append(corrected_chunk)
            except Exception as e:
                logger.warning(f"Failed to correct chunk, using original: {e}")
                corrected_chunks.append(chunk)
        
        return '\n'.join(corrected_chunks)
    
    def _correct_single_chunk(self, text: str) -> str:
        """
        단일 텍스트 청크 교정
        """
        prompt = self._create_correction_prompt(text)
        
        try:
            response = get_llm_response(
                prompt, 
                model_name=settings.OLLAMA_DEFAULT_MODEL,
                options={"num_predict": settings.LLM_NUM_PREDICT_TEXT, "temperature": settings.LLM_TEMPERATURE}
            )
            
            if response and response.strip():
                return response.strip()
            else:
                logger.warning("Empty LLM response, returning original text")
                return text
                
        except LLMError as e:
            logger.error(f"LLM correction failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during LLM correction: {e}")
            raise LLMError(f"Text correction failed: {e}", "CORRECTION_ERROR")
    
    def _split_text_into_chunks(self, text: str, max_size: int) -> List[str]:
        """
        텍스트를 의미 있는 단위로 청크 분할
        """
        # 문장 단위로 분할 우선 시도
        sentences = re.split(r'[.!?]\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def correct_text_chunks(self, text_chunks: List[str], use_llm: bool = True) -> List[str]:
        """
        텍스트 청크 목록을 일괄 교정
        
        Args:
            text_chunks: 교정할 텍스트 청크 리스트
            use_llm: LLM 사용 여부
            
        Returns:
            List[str]: 교정된 텍스트 청크 리스트
        """
        corrected_chunks = []
        
        for i, chunk in enumerate(text_chunks):
            try:
                corrected_chunk = self.correct_ocr_text(chunk, use_llm)
                corrected_chunks.append(corrected_chunk)
                logger.info(f"Corrected chunk {i+1}/{len(text_chunks)}")
            except Exception as e:
                logger.warning(f"Failed to correct chunk {i+1}, using original: {e}")
                corrected_chunks.append(chunk)
        
        return corrected_chunks
    
    def correct_text_in_batches(self, text: str, batch_size: int = 5000, use_llm: bool = True) -> str:
        """
        긴 텍스트를 배치 단위로 나누어 효율적으로 교정
        
        Args:
            text: 교정할 텍스트
            batch_size: 배치 크기 (문자 수)
            use_llm: LLM 사용 여부
            
        Returns:
            str: 교정된 텍스트
        """
        if not text or not text.strip():
            return text
        
        # 텍스트를 배치로 분할
        batches = []
        current_pos = 0
        
        while current_pos < len(text):
            # 배치 크기만큼 자르되, 문장 경계에서 자르도록 시도
            end_pos = min(current_pos + batch_size, len(text))
            
            # 배치 경계를 문장 끝에 맞추기
            if end_pos < len(text):
                # 마지막 문장 구분자 찾기
                last_sentence_end = text.rfind('.', current_pos, end_pos)
                if last_sentence_end == -1:
                    last_sentence_end = text.rfind('!', current_pos, end_pos)
                if last_sentence_end == -1:
                    last_sentence_end = text.rfind('?', current_pos, end_pos)
                if last_sentence_end == -1:
                    last_sentence_end = text.rfind('\n', current_pos, end_pos)
                
                if last_sentence_end > current_pos:
                    end_pos = last_sentence_end + 1
            
            batch = text[current_pos:end_pos].strip()
            if batch:
                batches.append(batch)
            
            current_pos = end_pos
        
        logger.info(f"텍스트를 {len(batches)}개 배치로 분할하여 교정 시작")
        
        # 배치별로 교정 수행
        corrected_batches = []
        for i, batch in enumerate(batches):
            try:
                logger.debug(f"배치 {i+1}/{len(batches)} 교정 중... (크기: {len(batch)} 문자)")
                corrected_batch = self.correct_ocr_text(batch, use_llm)
                corrected_batches.append(corrected_batch)
            except Exception as e:
                logger.warning(f"배치 {i+1} 교정 실패, 원본 사용: {e}")
                corrected_batches.append(batch)
        
        result = '\n'.join(corrected_batches)
        logger.info(f"배치 교정 완료: {len(text)} -> {len(result)} 문자")
        return result
    
    def get_correction_statistics(self, original_text: str, corrected_text: str) -> Dict[str, Any]:
        """
        교정 통계 정보 생성
        """
        original_words = set(original_text.split())
        corrected_words = set(corrected_text.split())
        
        changed_words = original_words.symmetric_difference(corrected_words)
        
        # 주조 용어 인식률 계산
        foundry_terms_found = 0
        for term in self.all_terms:
            if term in corrected_text:
                foundry_terms_found += 1
        
        return {
            "original_length": len(original_text),
            "corrected_length": len(corrected_text),
            "words_changed": len(changed_words),
            "foundry_terms_found": foundry_terms_found,
            "correction_ratio": len(changed_words) / max(len(original_words), 1)
        }

# 전역 인스턴스
ocr_correction_service = OCRCorrectionService()

def correct_ocr_text(text: str, use_llm: bool = True) -> str:
    """
    OCR 텍스트 교정 함수 (편의성을 위한 래퍼)
    """
    return ocr_correction_service.correct_ocr_text(text, use_llm)

def correct_text_chunks(text_chunks: List[str], use_llm: bool = True) -> List[str]:
    """
    텍스트 청크 목록 교정 함수 (편의성을 위한 래퍼)
    """
    return ocr_correction_service.correct_text_chunks(text_chunks, use_llm)

def correct_text_in_batches(text: str, batch_size: int = 5000, use_llm: bool = True) -> str:
    """
    긴 텍스트 배치 교정 함수 (편의성을 위한 래퍼)
    """
    return ocr_correction_service.correct_text_in_batches(text, batch_size, use_llm)