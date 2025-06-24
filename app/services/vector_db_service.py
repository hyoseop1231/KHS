import chromadb
import os
from typing import List, Dict, Any

# ChromaDB 클라이언트 설정
# 데이터를 영구 저장하기 위해 PersistentClient를 사용합니다.
# 저장 경로는 프로젝트 루트의 vector_db_data 디렉토리입니다.
CHROMA_DATA_PATH = "vector_db_data"
COLLECTION_NAME = "pdf_documents_collection"

# ChromaDB 클라이언트 초기화
# 이 클라이언트는 서비스 모듈이 로드될 때 생성됩니다.
try:
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
except Exception as e:
    print(f"Error initializing ChromaDB PersistentClient at '{CHROMA_DATA_PATH}': {e}")
    print("Please ensure the directory is writable and chromadb is installed correctly.")
    client = None

# 컬렉션 가져오기 또는 생성
# 컬렉션은 문서와 임베딩을 저장하는 공간입니다.
try:
    if client:
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            # metadata={"hnsw:space": "cosine"} # 유사도 측정 방식 (기본값은 L2, 코사인 유사도가 텍스트에 더 적합할 수 있음)
            # ChromaDB 최신 버전에서는 get_or_create_collection 시 embedding_function을 명시하거나,
            # add 할 때 embeddings를 직접 제공해야 합니다. 우리는 후자를 사용합니다.
        )
        print(f"ChromaDB collection '{COLLECTION_NAME}' loaded/created successfully at '{CHROMA_DATA_PATH}'.")
    else:
        collection = None
        print("ChromaDB client is not available. Collection cannot be loaded/created.")
except Exception as e:
    collection = None
    print(f"Error getting or creating ChromaDB collection '{COLLECTION_NAME}': {e}")


def store_vectors(document_id: str, text_chunks: List[str], vectors: List[List[float]], metadatas: List[Dict[str, Any]] = None):
    """
    Stores text chunks, their corresponding vectors, and metadatas in the ChromaDB collection.

    Args:
        document_id (str): A unique identifier for the source document (e.g., PDF filename).
        text_chunks (List[str]): The list of text chunks (documents in ChromaDB terms).
        vectors (List[List[float]]): The list of vector embeddings corresponding to text_chunks.
        metadatas (List[Dict[str, Any]], optional): A list of dictionaries containing metadata for each chunk.
                                                    Each metadata dict should include at least {'source_document_id': document_id, 'chunk_index': i}.
                                                    Defaults to None, in which case basic metadata is generated.
    """
    if not collection:
        print("Error: ChromaDB collection is not available. Cannot store vectors.")
        return

    if not text_chunks or not vectors:
        print("Error: Text chunks or vectors are empty. Nothing to store.")
        return

    if len(text_chunks) != len(vectors):
        print("Error: The number of text chunks and vectors must be the same.")
        return

    if metadatas and len(text_chunks) != len(metadatas):
        print("Error: If metadatas are provided, their count must match text_chunks and vectors.")
        return

    ids = []
    generated_metadatas = []
    for i, chunk in enumerate(text_chunks):
        chunk_id = f"{document_id}_chunk_{i}"
        ids.append(chunk_id)
        if metadatas:
            # 사용자가 제공한 메타데이터에 기본 정보를 추가하거나 확인
            current_meta = metadatas[i]
            current_meta.setdefault('source_document_id', document_id)
            current_meta.setdefault('chunk_index', i)
            generated_metadatas.append(current_meta)
        else:
            generated_metadatas.append({
                'source_document_id': document_id,
                'chunk_index': i,
                'original_text_preview': chunk[:200] # 미리보기용 원본 텍스트 일부
            })

    try:
        collection.add(
            embeddings=vectors,
            documents=text_chunks, # 원본 텍스트 청크도 함께 저장
            metadatas=generated_metadatas,
            ids=ids
        )
        print(f"Vector DB: Stored {len(vectors)} vectors for document '{document_id}' in collection '{COLLECTION_NAME}'.")
    except Exception as e:
        print(f"Error storing vectors in ChromaDB for document '{document_id}': {e}")


def search_similar_vectors(query_vector: List[float], top_k: int = 5, filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Searches for text chunks with vectors similar to the query_vector in the ChromaDB collection.

    Args:
        query_vector (List[float]): The vector representation of the user's query.
        top_k (int, optional): The number of similar results to return. Defaults to 5.
        filter_metadata (Dict[str, Any], optional): A dictionary to filter results based on metadata.
                                                  Example: {"source_document_id": "specific_doc.pdf"}
                                                  Defaults to None (no filtering).

    Returns:
        List[Dict[str, Any]]: A list of search results. Each result is a dictionary containing
                              'id', 'text' (document content), 'metadata', and 'distance' (or 'score').
    """
    if not collection:
        print("Error: ChromaDB collection is not available. Cannot search vectors.")
        return []

    if not query_vector:
        print("Error: Query vector is empty. Cannot perform search.")
        return []

    try:
        print(f"Vector DB: Searching for {top_k} similar vectors in '{COLLECTION_NAME}'.")
        if filter_metadata:
            print(f"Applying metadata filter: {filter_metadata}")

        results = collection.query(
            query_embeddings=[query_vector], # query_embeddings는 리스트의 리스트 형태여야 함
            n_results=top_k,
            where=filter_metadata, # 메타데이터 필터링 조건
            include=['documents', 'metadatas', 'distances'] # 반환할 정보: 원본 텍스트, 메타데이터, 거리
        )

        # 결과 형식 재구성 (필요에 따라)
        # ChromaDB의 query 결과는 약간 복잡한 구조를 가질 수 있습니다.
        # 예: {'ids': [['id1', 'id2']], 'documents': [['doc1', 'doc2']], ...}
        # 이를 사용하기 쉬운 형태로 변환합니다.

        formatted_results = []
        if results and results.get('ids') and results.get('ids')[0]:
            ids = results['ids'][0]
            documents = results['documents'][0] if results.get('documents') else [None] * len(ids)
            metadatas = results['metadatas'][0] if results.get('metadatas') else [None] * len(ids)
            distances = results['distances'][0] if results.get('distances') else [None] * len(ids)

            for i in range(len(ids)):
                formatted_results.append({
                    "id": ids[i],
                    "text": documents[i],
                    "metadata": metadatas[i],
                    "distance": distances[i] # 거리가 짧을수록 유사함
                })
            print(f"Vector DB: Found {len(formatted_results)} results.")
        else:
            print("Vector DB: No results found or empty result set.")

        return formatted_results

    except Exception as e:
        print(f"Error searching vectors in ChromaDB: {e}")
        return []

if __name__ == '__main__':
    # 간단한 테스트용
    print(f"Vector DB service module loaded. Chroma client: {'Initialized' if client else 'Failed'}. Collection: {'Initialized' if collection else 'Failed'}")

    if client and collection:
        print("\n--- Testing ChromaDB Operations (Example) ---")

        # 테스트용 데이터
        test_doc_id = "test_document.pdf"
        test_chunks = [
            "이것은 첫 번째 테스트 청크입니다. ChromaDB 저장 테스트.",
            "두 번째 청크는 약간 다른 내용을 담고 있습니다.",
            "세 번째 청크는 검색 테스트를 위해 고유한 키워드를 포함합니다: '코코넛'."
        ]
        # 실제 임베딩 대신 더미 임베딩 사용 (차원 수는 실제 모델과 맞추는 것이 좋음, 예: 384)
        # SentenceTransformer 모델 로드가 필요하므로, 여기서는 간단히 처리
        # from app.services.text_processing_service import get_embeddings as get_real_embeddings
        # test_vectors = get_real_embeddings(test_chunks) # 실제로는 이렇게 해야 함

        # 임시 더미 벡터 (실제로는 text_processing_service.get_embeddings를 사용해야 함)
        # text_processing_service가 로드될 때 모델을 다운로드하므로, 의존성을 피하기 위해 여기서는 더미 사용
        dummy_embedding_dim = 384 # 사용하는 임베딩 모델의 차원 수와 일치시켜야 함
        test_vectors = [[float(i/100.0)] * dummy_embedding_dim for i in range(len(test_chunks))]


        if not all(test_vectors) or not all(len(v) == dummy_embedding_dim for v in test_vectors):
             print("Dummy vectors are not correctly generated. Skipping store/search test.")
        else:
            print(f"\nStoring {len(test_chunks)} test vectors for document '{test_doc_id}'...")
            store_vectors(test_doc_id, test_chunks, test_vectors)

            print("\nSearching for vectors similar to the first chunk's vector...")
            # 첫 번째 청크의 벡터로 검색 (자기 자신도 결과에 포함될 수 있음)
            query_vec = test_vectors[0]
            similar_results = search_similar_vectors(query_vec, top_k=2)

            if similar_results:
                print("Search results:")
                for res in similar_results:
                    print(f"  ID: {res['id']}, Distance: {res['distance']:.4f}, Text: {res['text'][:50]}...")
            else:
                print("No similar results found or error during search.")

            print("\nSearching with a specific metadata filter (if applicable)...")
            # 예시: 특정 문서 ID로 필터링하여 검색
            # 이 테스트에서는 test_doc_id만 있으므로, 필터링 결과는 위와 유사할 것임
            filtered_results = search_similar_vectors(query_vec, top_k=2, filter_metadata={'source_document_id': test_doc_id})
            if filtered_results:
                print("Filtered search results:")
                for res in filtered_results:
                     print(f"  ID: {res['id']}, Distance: {res['distance']:.4f}, Text: {res['text'][:50]}...")
            else:
                print("No results found with the specified filter.")
    else:
        print("\nSkipping ChromaDB operations test as client or collection failed to initialize.")

    print("\nNote: For real testing, ensure 'text_processing_service.embedding_model' is loaded and used for generating actual embeddings.")
