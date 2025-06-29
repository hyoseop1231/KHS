"""
Ollama 모델 정보 캐싱 및 관리 서비스
LLM으로부터 직접 모델 정보를 가져와서 캐싱
"""
import json
import time
import requests
from typing import Dict, List, Optional, Any
from pathlib import Path
from app.config import settings
from app.utils.logging_config import get_logger
from app.services.llm_service import get_llm_response

logger = get_logger(__name__)

class ModelInfoCache:
    """모델 정보 캐싱 서비스"""
    
    def __init__(self):
        self.cache_file = Path("app/data/model_info_cache.json")
        self.cache_file.parent.mkdir(exist_ok=True)
        self.cache_data = self._load_cache()
        self.cache_ttl = 24 * 60 * 60  # 24시간
        
    def _load_cache(self) -> Dict[str, Any]:
        """캐시 파일 로드"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load model info cache: {e}")
        
        return {
            "last_updated": 0,
            "model_count": 0,
            "models": {}
        }
    
    def _save_cache(self):
        """캐시 파일 저장"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Model info cache saved: {len(self.cache_data['models'])} models")
        except Exception as e:
            logger.error(f"Failed to save model info cache: {e}")
    
    def _get_current_models(self) -> List[str]:
        """현재 Ollama에서 사용 가능한 모델 목록 가져오기"""
        base = settings.OLLAMA_API_URL
        urls = [
            base.replace('/api/generate', '/api/models'),
            base.replace('/api/generate', '/api/tags'),
        ]
        
        for url in urls:
            try:
                resp = requests.get(url, timeout=settings.OLLAMA_TIMEOUT)
                if resp.status_code == 200:
                    data = resp.json()
                    
                    # 응답 구조 파싱
                    if isinstance(data, dict) and 'models' in data:
                        mlist = data.get('models', [])
                        if mlist and isinstance(mlist[0], dict) and 'name' in mlist[0]:
                            return [m.get('name', '') for m in mlist]
                        else:
                            return list(mlist)
                    elif isinstance(data, list):
                        return data
                    
            except Exception as e:
                logger.warning(f"Failed to fetch models from {url}: {e}")
                continue
        
        logger.error("Failed to fetch model list from Ollama")
        return []
    
    def _get_model_info_from_llm(self, model_name: str) -> Dict[str, str]:
        """LLM에게 모델 정보를 질의 (간단하고 빠르게)"""
        prompt = f"""Model: {model_name}

Please provide brief info in JSON format:
{{
    "display_name": "Clean display name",
    "provider": "Company (Meta/Google/Mistral/etc)",
    "size": "Model size (7B/13B/etc)",
    "type": "Main purpose (Chat/Code/Reasoning/etc)"
}}

Keep it very brief and factual. Use "Unknown" if unsure."""

        try:
            # 더 짧은 타임아웃과 간단한 옵션으로 빠르게 처리
            response = get_llm_response(
                prompt, 
                model_name=model_name, 
                options={"num_predict": 200, "temperature": 0.1},  # 짧고 일관된 응답
                stream=False
            )
            
            # JSON 추출 시도
            response = response.strip()
            if response.startswith('```'):
                lines = response.split('\n')
                json_lines = []
                in_json = False
                for line in lines:
                    if line.startswith('```') and 'json' in line:
                        in_json = True
                        continue
                    elif line.startswith('```'):
                        break
                    elif in_json:
                        json_lines.append(line)
                response = '\n'.join(json_lines)
            
            # JSON 파싱
            info = json.loads(response)
            
            # 간단한 형식으로 변환
            simplified_info = {
                "display_name": info.get("display_name", model_name),
                "provider": info.get("provider", "Unknown"),
                "model_size": info.get("size", "Unknown"),
                "specialization": info.get("type", "Unknown"),
                "description": f"{info.get('provider', 'Unknown')} {info.get('type', 'AI')} 모델",
                "strengths": info.get("type", "Unknown"),
                "best_for": info.get("type", "Unknown")
            }
            
            logger.info(f"Successfully retrieved info for model: {model_name}")
            return simplified_info
            
        except Exception as e:
            logger.warning(f"Failed to get model info for {model_name}: {e}")
        
        # 실패 시 기본 정보 반환 (빠르게)
        return {
            "display_name": model_name.split(':')[0].title(),
            "provider": "Unknown",
            "model_size": "Unknown",
            "specialization": "AI Assistant", 
            "description": f"{model_name} AI 모델",
            "strengths": "General AI",
            "best_for": "General tasks"
        }
    
    def _needs_update(self, current_models: List[str]) -> bool:
        """캐시 업데이트가 필요한지 확인"""
        # TTL 확인
        current_time = time.time()
        if current_time - self.cache_data.get("last_updated", 0) > self.cache_ttl:
            logger.info("Model cache TTL expired")
            return True
        
        # 모델 수 변경 확인
        if len(current_models) != self.cache_data.get("model_count", 0):
            logger.info(f"Model count changed: {self.cache_data.get('model_count', 0)} -> {len(current_models)}")
            return True
        
        # 새로운 모델 추가 확인
        cached_models = set(self.cache_data.get("models", {}).keys())
        current_models_set = set(current_models)
        
        if not current_models_set.issubset(cached_models):
            new_models = current_models_set - cached_models
            logger.info(f"New models detected: {list(new_models)}")
            return True
        
        return False
    
    def get_model_info(self, force_refresh: bool = False) -> Dict[str, Any]:
        """모델 정보 가져오기 (캐싱 적용)"""
        current_models = self._get_current_models()
        
        if not current_models:
            logger.error("No models available from Ollama")
            return {"models": [], "summaries": [], "tips": []}
        
        # 업데이트 필요성 확인
        if force_refresh or self._needs_update(current_models):
            logger.info("Updating model info cache...")
            self._update_cache(current_models)
        else:
            logger.info("Using cached model info")
        
        # 캐시된 정보 반환
        cached_models = self.cache_data.get("models", {})
        
        summaries = []
        for model_name in current_models:
            if model_name in cached_models:
                info = cached_models[model_name]
                summaries.append({
                    "name": model_name,
                    "display_name": info.get("display_name", model_name),
                    "provider": info.get("provider", "알 수 없음"),
                    "params": f"{info.get('model_size', 'N/A')} | {info.get('specialization', 'N/A')}",
                    "key_points": info.get("description", "")
                })
        
        # 사용 팁 생성
        tips = [
            "💡 추론·코딩 문제 해결 → Qwen, DeepSeek 계열 모델 추천",
            "🇰🇷 한국어 질의응답 → 한국어 특화 모델 또는 다국어 모델 사용",
            "📊 RAG·검색 임베딩 → SFR-Embedding, BGE 계열 모델 추천",
            "⚡ 리소스 제약 환경 → 7B 이하 경량 모델 추천"
        ]
        
        return {
            "models": current_models,
            "summaries": summaries,
            "tips": tips
        }
    
    def _update_cache(self, current_models: List[str]):
        """캐시 업데이트"""
        updated_models = {}
        
        # 기존 캐시된 모델 정보 유지
        existing_models = self.cache_data.get("models", {})
        
        for model_name in current_models:
            if model_name in existing_models:
                # 기존 정보 유지
                updated_models[model_name] = existing_models[model_name]
                logger.debug(f"Keeping cached info for: {model_name}")
            else:
                # 새 모델 정보 가져오기
                logger.info(f"Fetching info for new model: {model_name}")
                model_info = self._get_model_info_from_llm(model_name)
                updated_models[model_name] = model_info
                time.sleep(0.5)  # LLM 요청 간 지연 (단축)
        
        # 캐시 업데이트
        self.cache_data = {
            "last_updated": time.time(),
            "model_count": len(current_models),
            "models": updated_models
        }
        
        self._save_cache()
        logger.info(f"Model cache updated with {len(updated_models)} models")
    
    def refresh_model_info(self, model_name: str = None):
        """특정 모델 또는 전체 모델 정보 강제 새로고침"""
        if model_name:
            logger.info(f"Refreshing info for model: {model_name}")
            model_info = self._get_model_info_from_llm(model_name)
            
            if "models" not in self.cache_data:
                self.cache_data["models"] = {}
            
            self.cache_data["models"][model_name] = model_info
            self._save_cache()
        else:
            logger.info("Refreshing all model info")
            self.get_model_info(force_refresh=True)

# 전역 인스턴스
model_info_cache = ModelInfoCache()

def get_cached_model_info(force_refresh: bool = False) -> Dict[str, Any]:
    """모델 정보 가져오기 (편의 함수)"""
    return model_info_cache.get_model_info(force_refresh)

def refresh_model_cache(model_name: str = None):
    """모델 캐시 새로고침 (편의 함수)"""
    model_info_cache.refresh_model_info(model_name)