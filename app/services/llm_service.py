import requests
import json
from typing import List, Dict, Any
from app.config import settings
from app.utils.logging_config import get_logger
from app.utils.exceptions import LLMError

logger = get_logger(__name__)

def get_llm_response(prompt: str, model_name: str = None, options: Dict = None, stream: bool = False) -> str:
    """
    Sends a prompt to the Ollama LLM and gets a response.

    Args:
        prompt (str): The prompt to send to the LLM.
        model_name (str, optional): The name of the Ollama model to use.
                                    Defaults to DEFAULT_MODEL if not provided.
        options (Dict, optional): Additional Ollama model parameters (e.g., temperature).
                                  Refer to Ollama API documentation for available options.
        stream (bool, optional): Whether to stream the response. Defaults to False.
                                 If True, the function will yield chunks of the response.
                                 Currently, this function aggregates the stream for simplicity.

    Returns:
        str: The LLM's response text, or an error message if something goes wrong.
    """
    if not model_name:
        model_name = settings.OLLAMA_DEFAULT_MODEL

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": stream,
    }
    if options:
        payload["options"] = options

    logger.info(f"Sending prompt to Ollama model '{model_name}'. Prompt length: {len(prompt)} chars")
    logger.debug(f"Prompt preview: {prompt[:200]}...")

    try:
        response = requests.post(
            settings.OLLAMA_API_URL, 
            json=payload, 
            stream=stream,
            timeout=settings.OLLAMA_TIMEOUT
        )
        response.raise_for_status()

        if stream:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    try:
                        json_chunk = json.loads(decoded_line)
                        if 'response' in json_chunk:
                            full_response += json_chunk['response']
                        if json_chunk.get('done', False): # Check for the 'done' field
                            break
                    except json.JSONDecodeError:
                        logger.warning(f"Non-JSON line in stream: {decoded_line}")
            logger.info("Streamed response received and aggregated")
            return full_response.strip()
        else:
            response_data = response.json()
            logger.info("Non-streamed response received")
            return response_data.get("response", "Error: No 'response' field in LLM output.").strip()

    except requests.exceptions.ConnectionError as e:
        error_msg = f"Could not connect to Ollama at {settings.OLLAMA_API_URL}. Ensure Ollama is running."
        logger.error(f"{error_msg} Details: {e}")
        raise LLMError(error_msg, "CONNECTION_ERROR")
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP error occurred: {e}"
        logger.error(f"{error_msg}. Response: {e.response.text if e.response else 'No response body'}")
        raise LLMError(error_msg, "HTTP_ERROR")
    except requests.exceptions.Timeout as e:
        error_msg = f"Request timed out after {settings.OLLAMA_TIMEOUT} seconds"
        logger.error(f"{error_msg}: {e}")
        raise LLMError(error_msg, "TIMEOUT_ERROR")
    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"
        logger.error(error_msg)
        raise LLMError(error_msg, "UNEXPECTED_ERROR")


def construct_rag_prompt(query: str, context_chunks: List[str], lang: str = "ko") -> str:
    """
    Constructs a prompt for the LLM using the RAG pattern, tailored for Korean or English.

    Args:
        query (str): The user's original question.
        context_chunks (List[str]): A list of relevant text chunks retrieved from the vector database.
        lang (str, optional): Language of the prompt. "ko" for Korean, "en" for English. Defaults to "ko".

    Returns:
        str: The fully constructed prompt to be sent to the LLM.
    """
    context = "\n\n---\n\n".join(context_chunks) # 각 청크를 명확히 구분

    if lang == "ko":
        prompt = f"""주어진 문맥 정보를 바탕으로 다음 질문에 답변해 주세요.

[문맥 정보]
{context}
---

[질문]
{query}

[지침]
- 답변은 마크다운(Markdown) 형식으로 작성해주세요. 표, 코드, 강조, 리스트, 링크, 이미지 등 다양한 서식을 적극적으로 활용하세요.
- 각 문맥 청크별로 [출처:문서ID] 태그가 있으니, 답변에 해당 정보를 반드시 포함하여 출처를 명확히 표시하세요.
- 문맥 정보에서 질문에 대한 답을 찾을 수 있는 경우, 해당 정보를 사용하여 답변을 생성해주세요.
- 답변은 완전한 문장으로, 명확하고 친절하게, 최대한 자세하고 길게 작성해주세요.
- 예시, 표, 코드, 추가 설명이 가능하다면 포함해주세요.
- 만약 문맥 정보에서 질문에 대한 답을 찾을 수 없다면, "제공된 문맥 정보만으로는 질문에 답변하기 어렵습니다."라고 답변해주세요.
- 답변에 개인적인 의견이나 문맥 정보에 없는 내용을 추가하지 마세요.
- 한국어로 답변해주세요.

답변:
"""
    else: # Default to English prompt
        prompt = f"""Based on the provided context information, please answer the following question.

[Context Information]
{context}
---

[Question]
{query}

[Instructions]
- Write your answer in Markdown format. Use tables, code, emphasis, lists, links, images, etc. as appropriate.
- Each context chunk has a [Source:DocumentID] tag. Be sure to include this information in your answer to clearly indicate the source.
- If you can find the answer to the question in the context information, please use that information to generate your answer.
- Your answer should be a complete sentence, clear, and as detailed and long as possible.
- If possible, include examples, tables, code, or additional explanations.
- If you cannot find the answer to the question in the context information, please respond with "It is difficult to answer the question based on the provided context information alone."
- Do not add personal opinions or information not present in the context.
- Please answer in English.

Answer:
"""
    return prompt


def process_llm_chat_request(user_query: str,
                             retrieved_docs: List[Dict[str, Any]],
                             model_name: str = None,
                             lang: str = "ko") -> str:
    """
    High-level function to process a chat request using RAG.
    1. Constructs a prompt from the user query and retrieved documents.
    2. Gets a response from the LLM.

    Args:
        user_query (str): The user's question.
        retrieved_docs (List[Dict[str, Any]]): List of documents (chunks) retrieved from vector DB.
                                               Each dict should have a "text" key.
        model_name (str, optional): Ollama model name. Defaults to settings.OLLAMA_DEFAULT_MODEL.
        lang (str, optional): Language for the prompt. Defaults to "ko".

    Returns:
        str: The LLM's response.
    """
    if not retrieved_docs:
        if lang == "ko":
            return "관련 문서를 찾을 수 없어 답변을 생성할 수 없습니다."
        else:
            return "No relevant documents found to generate an answer."

    # 각 청크에 출처 태그 추가
    context_chunks = []
    for doc in retrieved_docs:
        text = doc.get("text", "")
        meta = doc.get("metadata", {})
        source = meta.get("source_document_id", "?")
        if text:
            context_chunks.append(f"{text}\n\n[출처:{source}]")
    if not context_chunks:
        if lang == "ko":
            return "검색된 문서에 내용이 없어 답변을 생성할 수 없습니다."
        else:
            return "Retrieved documents have no content to generate an answer."

    rag_prompt = construct_rag_prompt(user_query, context_chunks, lang=lang)

    # Ollama 모델 옵션 (필요에 따라 추가/수정)
    ollama_options = {
        "temperature": 0.8,  # 더 다양한 답변
        "num_predict": 2048,  # 답변 최대 토큰 수 대폭 증가
    }

    try:
        llm_response = get_llm_response(rag_prompt, model_name=model_name, options=ollama_options, stream=False)
        logger.info(f"Successfully generated LLM response for query: {user_query[:50]}...")
        return llm_response
    except LLMError as e:
        logger.error(f"LLM error in chat request: {e}")
        if lang == "ko":
            return f"답변 생성 중 오류가 발생했습니다: {e.message}"
        else:
            return f"An error occurred while generating response: {e.message}"


if __name__ == '__main__':
    print(f"LLM service module loaded. Ollama API URL: {settings.OLLAMA_API_URL}, Default Model: {settings.OLLAMA_DEFAULT_MODEL}")

    # --- Test RAG Prompt Construction ---
    print("\n--- Testing RAG Prompt Construction (Korean) ---")
    sample_query_ko = "코코넛의 효능은 무엇인가요?"
    sample_context_ko = [
        "코코넛은 다양한 건강 효능을 가지고 있습니다. 특히 코코넛 오일은 피부 보습에 좋습니다.",
        "또한, 코코넛 워터는 전해질이 풍부하여 운동 후 수분 보충에 도움을 줄 수 있습니다."
    ]
    korean_prompt = construct_rag_prompt(sample_query_ko, sample_context_ko, lang="ko")
    print(f"Generated Korean RAG Prompt:\n{korean_prompt}")

    # --- Test LLM Call (Requires Ollama server to be running with the model) ---
    # 주의: 아래 테스트는 실제 Ollama 서버와 통신을 시도합니다.
    # `DEFAULT_MODEL` (예: 'llama2')이 Ollama에 로드되어 있어야 합니다.

    # 테스트를 실행하려면 아래 주석을 해제하세요.
    # print(f"\n--- Testing LLM Call (Model: {DEFAULT_MODEL}) ---")
    # test_prompt = "대한민국의 수도는 어디인가요? 한 문장으로 답해주세요."
    # print(f"Sending test prompt: {test_prompt}")
    # response = get_llm_response(test_prompt, model_name=DEFAULT_MODEL)
    # print(f"LLM Response: {response}")

    # print(f"\n--- Testing RAG-style LLM Call (Model: {DEFAULT_MODEL}) ---")
    # # 위에서 생성한 korean_prompt 사용
    # print(f"Sending RAG prompt (first 100 chars): {korean_prompt[:100]}...")
    # rag_response = get_llm_response(korean_prompt, model_name=DEFAULT_MODEL)
    # print(f"LLM RAG Response: {rag_response}")

    # print(f"\n--- Testing process_llm_chat_request ---")
    # test_retrieved_docs = [{"text": chunk} for chunk in sample_context_ko]
    # chat_response = process_llm_chat_request(sample_query_ko, test_retrieved_docs, model_name=DEFAULT_MODEL)
    # print(f"Processed LLM Chat Response: {chat_response}")

    print("\nTo run LLM call tests, uncomment the relevant sections in `if __name__ == '__main__':`")
    print(f"Ensure your Ollama server is running and the model '{DEFAULT_MODEL}' (or your preferred model) is available.")
