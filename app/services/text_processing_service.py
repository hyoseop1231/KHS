from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import threading
import re
from typing import List
from app.config import settings
from app.utils.logging_config import get_logger
from app.utils.exceptions import EmbeddingError

logger = get_logger(__name__)

# Thread-safe singleton for embedding model
class EmbeddingModelManager:
    _instance = None
    _lock = threading.Lock()
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
                        self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
                        logger.info("Embedding model loaded successfully")
                    except Exception as e:
                        logger.error(f"Error loading SentenceTransformer model '{settings.EMBEDDING_MODEL}': {e}")
                        raise EmbeddingError(f"Could not load embedding model: {e}", "MODEL_LOAD_ERROR")
        return self._model

# Global model manager instance
model_manager = EmbeddingModelManager()

# 주조(주물) 분야 용어 교정 사전 (예시)
FOUNDRY_TERM_MAP = {
    '주물': '주조', '몰드': '주형', '코어': '심', '캐비티': '형공', '패턴': '주형모형', '인게이트': '주입구',
    '슬래그': '슬래그(불순물)', '샌드': '주형사', '포어': '기공', '블로홀': '기공', '쉘': '쉘(껍질)',
    '필터': '여과기', '플라스크': '플라스크(주형틀)', '스프루': '주입구', '러너': '주입로', '라이저': '승강구',
    '포어시티': '기공성', '쇼트': '주조불량', '핀홀': '기공', '버': '이물질', '플래시': '이물질',
    '게이트': '주입구', '매니폴드': '분배기', '노즐': '노즐(분사구)', '디플렉터': '방향전환기', '슬리브': '슬리브(보호관)',
    '플러그': '플러그(마개)', '인서트': '인서트(삽입물)', '인베스트먼트': '정밀주조', '다이': '금형',
    '다이캐스팅': '다이캐스팅(압출주조)', '샌드몰드': '주형사주형', '샌드캐스팅': '주형사주조', '로스트왁스': '정밀주조',
    '왁스패턴': '왁스모형', '그린샌드': '습식주형사', '드라이샌드': '건식주형사', '클레이': '점토',
    '실리카': '규사', '매그네시아': '마그네시아', '크로마이트': '크로마이트(광물)', '지르코니아': '지르코니아(광물)',
    '페놀': '페놀(수지)', '레진': '레진(수지)', '바인더': '결합제', '하드너': '경화제', '카본': '탄소',
    '그래파이트': '흑연', '페라이트': '페라이트(철)', '오스테나이트': '오스테나이트(철)', '마르텐사이트': '마르텐사이트(철)',
    '펄라이트': '펄라이트(철)', '시멘타이트': '시멘타이트(철)', '주철': '주철(철)', '강': '강(철)',
    '스테인리스': '스테인리스(철)', '알루미늄': '알루미늄(금속)', '구리': '구리(금속)', '아연': '아연(금속)',
    '마그네슘': '마그네슘(금속)', '티타늄': '티타늄(금속)', '니켈': '니켈(금속)', '크롬': '크롬(금속)',
    '몰리브덴': '몰리브덴(금속)', '텅스텐': '텅스텐(금속)', '코발트': '코발트(금속)', '실리콘': '실리콘(원소)',
    '망간': '망간(원소)', '인': '인(원소)', '황': '황(원소)', '납': '납(금속)', '주강': '주강(철)',
    '주석': '주석(금속)', '브론즈': '청동', '황동': '황동(합금)', '주물강': '주강(철)', '주물주철': '주철(철)',
    '주물알루미늄': '알루미늄(금속)', '주물구리': '구리(금속)', '주물아연': '아연(금속)', '주물마그네슘': '마그네슘(금속)',
    '주물티타늄': '티타늄(금속)', '주물니켈': '니켈(금속)', '주물크롬': '크롬(금속)', '주물코발트': '코발트(금속)',
    '주물실리콘': '실리콘(원소)', '주물망간': '망간(원소)', '주물인': '인(원소)', '주물황': '황(원소)',
    '주물납': '납(금속)', '주물주강': '주강(철)', '주물주석': '주석(금속)', '주물브론즈': '청동', '주물황동': '황동(합금)'
}

def correct_foundry_terms(text: str) -> str:
    """
    주조(주물) 분야 용어를 표준 용어로 교정합니다.
    """
    import re
    def replace(match):
        word = match.group(0)
        return FOUNDRY_TERM_MAP.get(word, word)
    # 단어 단위로만 치환 (한글, 영문, 숫자 포함)
    pattern = re.compile(r'|'.join(map(re.escape, sorted(FOUNDRY_TERM_MAP, key=len, reverse=True))))
    return pattern.sub(replace, text)

def split_text_into_chunks(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """
    Splits a long text into smaller, overlapping chunks using Langchain's RecursiveCharacterTextSplitter.
    """
    if not text:
        return []
    
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks (size: {chunk_size}, overlap: {chunk_overlap})")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text: {e}")
        raise EmbeddingError(f"Text splitting failed: {e}", "TEXT_SPLIT_ERROR")

def get_embeddings(text_chunks: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Converts text chunks into vector embeddings with optimized batch processing.
    Returns a list of embeddings, where each embedding is a list of floats.
    """
    if not text_chunks:
        return []
    
    try:
        model = model_manager.get_model()
        logger.info(f"Generating embeddings for {len(text_chunks)} chunks using '{settings.EMBEDDING_MODEL}'")
        
        # Process in batches for better memory management
        all_embeddings = []
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1}/{(len(text_chunks) + batch_size - 1)//batch_size}")
            
            # Generate embeddings for this batch
            batch_embeddings = model.encode(
                batch, 
                convert_to_tensor=False,
                show_progress_bar=False,
                batch_size=min(batch_size, len(batch))
            )
            
            # Convert to list format
            batch_embeddings_list = [embedding.tolist() for embedding in batch_embeddings]
            all_embeddings.extend(batch_embeddings_list)
        
        logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
        return all_embeddings
        
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}")
        raise EmbeddingError(f"Embedding generation failed: {e}", "EMBEDDING_GENERATION_ERROR")

if __name__ == '__main__':
    # 간단한 테스트용
    print(f"Text processing service module loaded. Embedding model: '{MODEL_NAME if embedding_model else 'Failed to load'}'")

    sample_text = """이것은 긴 샘플 텍스트입니다. 문장 분할과 청크 생성을 테스트하기 위한 것입니다.
여러 문단으로 구성될 수 있으며, 각 문단은 여러 문장을 포함할 수 있습니다.
Langchain의 RecursiveCharacterTextSplitter는 이러한 텍스트를 효과적으로 나눌 수 있어야 합니다.
한글 텍스트에 대해서도 잘 동작하는지 확인이 필요합니다.

두 번째 문단입니다. 이 문단도 여러 문장으로 이루어져 있습니다.
임베딩 모델은 각 청크를 숫자 벡터로 변환하여 의미적 유사성을 계산할 수 있도록 합니다.
이것이 RAG 시스템의 핵심 요소 중 하나입니다.
Sentence Transformers 라이브러리는 다양한 사전 훈련된 모델을 제공합니다.
"""
    print("\n--- Testing Text Splitting ---")
    chunks = split_text_into_chunks(sample_text, chunk_size=100, chunk_overlap=20)
    if chunks:
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: {chunk} (Length: {len(chunk)})")
    else:
        print("No chunks generated.")

    if embedding_model and chunks:
        print("\n--- Testing Embedding Generation ---")
        embeddings = get_embeddings(chunks)
        if embeddings and all(e for e in embeddings): # 모든 임베딩이 비어있지 않은지 확인
            print(f"Generated {len(embeddings)} embeddings.")
            print(f"Dimension of the first embedding: {len(embeddings[0]) if embeddings[0] else 'N/A'}")
            # print(f"First embedding (first 5 values): {embeddings[0][:5] if embeddings[0] else 'N/A'}")
        else:
            print("Failed to generate embeddings or some embeddings are empty.")
    elif not chunks:
        print("\nSkipping embedding generation test as no chunks were created.")
    else:
        print("\nSkipping embedding generation test as embedding model failed to load.")
