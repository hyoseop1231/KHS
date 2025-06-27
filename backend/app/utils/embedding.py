from sentence_transformers import SentenceTransformer
import logging
import os

logger = logging.getLogger(__name__)

_embed_model: SentenceTransformer | None = None
MODEL_NAME = "jhgan/ko-sbert-nli" # 설계서 기반 모델
DEVICE = "cpu" # CPU 사용 명시

def get_embed_model() -> SentenceTransformer:
    """
    Loads and returns the SentenceTransformer model.
    Initializes it only once.
    """
    global _embed_model
    if _embed_model is None:
        try:
            logger.info(f"Loading SentenceTransformer model: {MODEL_NAME} on device: {DEVICE}")
            # 모델 다운로드 경로를 지정하여 Docker 이미지 빌드 시 포함시키거나,
            # 실행 시점에 다운로드 받도록 할 수 있습니다.
            # 여기서는 실행 시점에 ~/.cache/torch/sentence_transformers 에 다운로드됩니다.
            # 특정 경로에 저장하고 싶다면 cache_folder 인자 사용 가능.
            # cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "sentence_transformers")
            # os.makedirs(cache_dir, exist_ok=True)

            _embed_model = SentenceTransformer(MODEL_NAME, device=DEVICE) # cache_folder=cache_dir
            logger.info(f"SentenceTransformer model '{MODEL_NAME}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{MODEL_NAME}': {e}", exc_info=True)
            # 모델 로드 실패 시 애플리케이션이 정상 동작하기 어려우므로,
            # 여기서 예외를 다시 발생시키거나, 프로그램 종료를 고려할 수 있습니다.
            # 또는 _embed_model이 None으로 유지되어 이후 호출에서 오류가 발생하도록 둡니다.
            raise RuntimeError(f"Could not load SentenceTransformer model: {MODEL_NAME}") from e

    return _embed_model

# 애플리케이션 시작 시 모델을 미리 로드하고 싶다면, 이 모듈을 임포트하는 시점에 호출하거나
# FastAPI의 startup 이벤트 핸들러에서 get_embed_model()을 호출합니다.
# 여기서는 embed_model 변수를 통해 필요시 로드되도록 합니다.
# embed_model = get_embed_model() # 모듈 로드 시 바로 초기화 (스타트업에서 하는 것이 더 제어하기 좋음)

if __name__ == '__main__':
    logger.info("Testing embedding module...")
    try:
        model = get_embed_model()
        if model:
            logger.info("Embedding model loaded successfully via get_embed_model().")
            test_sentences = ["안녕하세요, 한국어 임베딩 모델 테스트입니다.", "이것은 두 번째 문장입니다."]
            embeddings = model.encode(test_sentences)
            logger.info(f"Test embeddings shape: {embeddings.shape}") # (num_sentences, embedding_dim)
            logger.info("Embedding module test successful.")
        else:
            logger.error("Failed to get embedding model for test.")
    except Exception as e:
        logger.error(f"Error during embedding module test: {e}", exc_info=True)
