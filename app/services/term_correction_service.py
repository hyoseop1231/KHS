"""
주조 용어 교정 및 표준화 서비스
"""
import json
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

class TermCorrectionService:
    """
    주조 기술 용어의 표준화와 교정을 담당하는 서비스
    """
    
    def __init__(self):
        self.terminology_dict = self._load_terminology_dict()
        self.standard_terms = self._build_standard_terms_map()
        self.common_errors = self._build_common_error_patterns()
        
    def _load_terminology_dict(self) -> Dict:
        """주조 용어집 로드"""
        try:
            terminology_path = Path(__file__).parent.parent / "data" / "foundry_terminology.json"
            with open(terminology_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load foundry terminology: {e}")
            return {"categories": {}, "common_ocr_errors": {}}
    
    def _build_standard_terms_map(self) -> Dict[str, str]:
        """표준 용어 매핑 딕셔너리 생성"""
        standard_terms = {}
        
        # 기본 주조 용어들
        foundry_terms = {
            # 주형 관련
            "주형": ["주형", "鑄型", "mold", "틀"],
            "사형": ["사형", "砂型", "sand mold", "모래형"],
            "금형": ["금형", "金型", "metal mold", "다이"],
            
            # 게이팅 시스템
            "탕구": ["탕구", "湯口", "sprue", "탕도"],
            "러너": ["러너", "runner", "유로"],
            "게이트": ["게이트", "gate", "게이트"],
            "라이저": ["라이저", "riser", "압탕", "피더", "feeder"],
            
            # 코어 관련
            "코어": ["코어", "core", "중자"],
            "코어박스": ["코어박스", "core box", "중자함"],
            
            # 패턴 관련  
            "패턴": ["패턴", "pattern", "모형"],
            "플라스크": ["플라스크", "flask", "주형틀"],
            
            # 용해 관련
            "쿠폴라": ["쿠폴라", "cupola", "큐폴라"],
            "용해": ["용해", "熔解", "melting", "녹이기"],
            "용탕": ["용탕", "熔湯", "molten metal", "쇳물"],
            
            # 응고 관련
            "응고": ["응고", "凝固", "solidification", "굳기"],
            "수축": ["수축", "收縮", "shrinkage", "줄어들기"],
            
            # 결함 관련
            "결함": ["결함", "缺陷", "defect", "불량"],
            "기포": ["기포", "氣泡", "porosity", "구멍"],
            "균열": ["균열", "龜裂", "crack", "갈라짐"],
            "냉금": ["냉금", "冷金", "cold shut", "미용착"],
        }
        
        # 각 표준 용어에 대해 변형들을 매핑
        for standard_term, variations in foundry_terms.items():
            for variation in variations:
                standard_terms[variation.lower()] = standard_term
                standard_terms[variation] = standard_term
        
        return standard_terms
    
    def _build_common_error_patterns(self) -> List[Tuple[str, str]]:
        """일반적인 OCR/타이핑 오류 패턴들"""
        return [
            # 숫자와 문자 혼동
            (r'(?<=[가-힣])O(?=[가-힣])', '0'),
            (r'(?<=[가-힣])I(?=[가-힣])', '1'),
            (r'(?<=[가-힣])l(?=[가-힣])', '1'),
            (r'(?<=\d)O(?=\d)', '0'),
            (r'(?<=\d)I(?=\d)', '1'),
            (r'(?<=\d)l(?=\d)', '1'),
            
            # 자주 틀리는 용어들
            (r'주혈', '주형'),
            (r'주형사', '주형'),
            (r'수형', '주형'),
            (r'탕도', '탕구'),
            (r'톤구', '탕구'),
            (r'탕꾸', '탕구'),
            (r'래이저', '라이저'),
            (r'라이져', '라이저'),
            (r'리이저', '라이저'),
            (r'코어박수', '코어박스'),
            (r'쿠플라', '쿠폴라'),
            (r'큐폴라', '쿠폴라'),
            (r'응고온도', '응고'),
            (r'수촉', '수축'),
            (r'수척', '수축'),
        ]
    
    def correct_text(self, text: str) -> str:
        """
        주조 기술 텍스트의 용어를 교정하고 표준화
        
        Args:
            text: 교정할 텍스트
            
        Returns:
            str: 교정된 텍스트
        """
        if not text:
            return text
            
        corrected_text = text
        
        # 1. 일반적인 OCR 오류 패턴 교정
        for pattern, replacement in self.common_errors:
            corrected_text = re.sub(pattern, replacement, corrected_text)
        
        # 2. 알려진 OCR 오류 패턴 교정 (foundry_terminology.json에서)
        ocr_errors = self.terminology_dict.get("common_ocr_errors", {})
        for correct_term, error_patterns in ocr_errors.items():
            for error_pattern in error_patterns:
                corrected_text = corrected_text.replace(error_pattern, correct_term)
        
        # 3. 표준 용어로 변환
        corrected_text = self._standardize_terms(corrected_text)
        
        return corrected_text.strip()
    
    def _standardize_terms(self, text: str) -> str:
        """주조 용어를 표준 용어로 변환"""
        # 단어 경계를 고려한 용어 교체
        for term, standard in self.standard_terms.items():
            if term != standard:  # 자기 자신으로의 교체 방지
                # 완전한 단어 매치를 위한 패턴
                pattern = r'\b' + re.escape(term) + r'\b'
                text = re.sub(pattern, standard, text, flags=re.IGNORECASE)
        
        return text
    
    def get_term_suggestions(self, query: str) -> List[str]:
        """
        입력된 질의에서 관련 주조 용어 제안
        
        Args:
            query: 사용자 질의
            
        Returns:
            List[str]: 관련 용어 목록
        """
        suggestions = []
        query_lower = query.lower()
        
        # 표준 용어에서 관련 용어 찾기
        for term in self.standard_terms.values():
            if any(keyword in query_lower for keyword in [
                '주형', '탕구', '라이저', '코어', '패턴', '용해', '응고', '결함'
            ]):
                if term not in suggestions:
                    suggestions.append(term)
        
        return suggestions[:10]  # 최대 10개만 반환
    
    def validate_technical_terms(self, text: str) -> Dict[str, List[str]]:
        """
        텍스트에서 주조 기술 용어의 정확성 검증
        
        Returns:
            Dict: {
                'correct_terms': [...],
                'suspicious_terms': [...],
                'suggestions': [...]
            }
        """
        words = re.findall(r'[가-힣A-Za-z]+', text)
        correct_terms = []
        suspicious_terms = []
        
        for word in words:
            if word in self.standard_terms.values():
                correct_terms.append(word)
            elif any(keyword in word.lower() for keyword in [
                '주', '형', '탕', '구', '라이', '코어', '패턴', '용해', '응고'
            ]):
                suspicious_terms.append(word)
        
        return {
            'correct_terms': list(set(correct_terms)),
            'suspicious_terms': list(set(suspicious_terms)),
            'suggestions': self.get_term_suggestions(text)
        }

# 전역 인스턴스
term_correction_service = TermCorrectionService()

def correct_foundry_terms(text: str) -> str:
    """주조 용어 교정 함수 (편의성을 위한 래퍼)"""
    return term_correction_service.correct_text(text)

def validate_foundry_terms(text: str) -> Dict[str, List[str]]:
    """주조 용어 검증 함수 (편의성을 위한 래퍼)"""
    return term_correction_service.validate_technical_terms(text)