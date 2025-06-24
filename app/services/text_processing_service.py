from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

# 한국어 임베딩 모델 로드
# 모델을 처음 사용하는 경우 자동으로 다운로드됩니다.
# 인터넷 연결이 필요할 수 있습니다.
# Hugging Face Hub에서 다른 한국어 모델을 선택할 수도 있습니다.
# 예: "snunlp/KR-SBERT-V40K-KoCharMean" 등
MODEL_NAME = "jhgan/ko-sroberta-multitask"
try:
    embedding_model = SentenceTransformer(MODEL_NAME)
except Exception as e:
    print(f"Error loading SentenceTransformer model '{MODEL_NAME}': {e}")
    print("Please ensure you have an internet connection for the first download,")
    print("or that the model is available locally if you've downloaded it manually.")
    embedding_model = None # 모델 로드 실패 시 None으로 설정

def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[str]:
    """
    Splits a long text into smaller, overlapping chunks using Langchain's RecursiveCharacterTextSplitter.
    This splitter tries to keep paragraphs, then sentences, then words together.
    """
    if not text:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False, # 정규식 대신 기본 구분자 사용
        separators=["\n\n", "\n", " ", ""] # 문단, 줄바꿈, 공백 순으로 분리 시도
    )
    chunks = text_splitter.split_text(text)
    print(f"Text processing: Split text into {len(chunks)} chunks (size: {chunk_size}, overlap: {chunk_overlap})")
    return chunks

def get_embeddings(text_chunks: list[str]) -> list[list[float]]:
    """
    Converts text chunks into vector embeddings using the pre-loaded SentenceTransformer model.
    Returns a list of embeddings, where each embedding is a list of floats.
    """
    if not embedding_model:
        print("Error: Embedding model is not loaded. Cannot generate embeddings.")
        # 빈 리스트 또는 적절한 오류 처리를 반환할 수 있습니다.
        # 여기서는 각 청크에 대해 빈 임베딩을 반환하여 호출하는 쪽에서 처리하도록 합니다.
        return [[] for _ in text_chunks]

    if not text_chunks:
        return []

    print(f"Text processing: Generating embeddings for {len(text_chunks)} chunks using '{MODEL_NAME}'.")
    try:
        # encode()는 numpy array를 반환하므로, tolist()를 사용하여 list of lists of floats로 변환합니다.
        embeddings = embedding_model.encode(text_chunks, convert_to_tensor=False)
        # numpy.ndarray를 list of lists of floats로 변환
        embeddings_list = [embedding.tolist() for embedding in embeddings]
        print(f"Text processing: Successfully generated {len(embeddings_list)} embeddings.")
        return embeddings_list
    except Exception as e:
        print(f"Error during embedding generation: {e}")
        return [[] for _ in text_chunks] # 오류 발생 시 빈 임베딩 반환

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
