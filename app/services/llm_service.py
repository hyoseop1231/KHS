import requests
import aiohttp
import asyncio
import json
from typing import List, Dict, Any, Optional, Generator
from app.config import settings
from app.utils.logging_config import get_logger
from app.utils.exceptions import LLMError

logger = get_logger(__name__)

def format_cached_response_with_llm(cached_text: str, model_name: str = None) -> str:
    """
    Format cached response text using LLM to add proper markdown structure
    """
    if not model_name:
        model_name = settings.OLLAMA_DEFAULT_MODEL
    
    # Create a formatting prompt
    formatting_prompt = f"""다음 주조 기술 텍스트를 읽기 쉽게 마크다운 형식으로 구조화해주세요. 
내용과 전문 용어는 절대 변경하지 말고, 오직 형식만 개선해주세요:

**형식 개선 규칙:**
- 제목과 소제목에 # ## ### 사용
- 중요한 내용은 **굵게** 표시
- 목록은 - 또는 1. 사용
- 주조 전문용어는 `백틱` 사용 (예: `주형`, `탕구`, `라이저`)
- 표가 있다면 마크다운 표 형식으로 정리
- 긴 문단은 적절히 나누기
- 기술적 수치나 데이터는 명확히 구분

**주조 전문 용어 유지:**
주형, 탕구, 라이저, 코어, 패턴, 사형, 금형, 결함, 수축, 응고, 용해 등의 용어는 정확히 유지하세요.

원본 텍스트:
{cached_text}

구조화된 마크다운:"""

    try:
        payload = {
            "model": model_name,
            "prompt": formatting_prompt,
            "stream": False,
            "options": {
                "num_predict": 2048,
                "temperature": 0.3,  # Low temperature for consistent formatting
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }
        
        logger.debug(f"Sending cached text formatting request to Ollama: {model_name}")
        
        response = requests.post(
            settings.OLLAMA_API_URL, 
            json=payload, 
            timeout=settings.OLLAMA_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            formatted_text = result.get('response', '').strip()
            
            if formatted_text:
                logger.info(f"Successfully formatted cached response ({len(cached_text)} -> {len(formatted_text)} chars)")
                return formatted_text
            else:
                logger.warning("LLM returned empty formatted text, using original")
                return cached_text
        else:
            logger.error(f"Ollama formatting request failed: {response.status_code} - {response.text}")
            return cached_text
            
    except Exception as e:
        logger.error(f"Error formatting cached response with LLM: {e}")
        return cached_text

def get_llm_response(prompt: str, model_name: str = None, options: Dict = None, stream: bool = False) -> str | Generator[str, None, None]:
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

    Returns:
        str: The LLM's response text if stream is False.
        Generator[str, None, None]: A generator that yields chunks of the LLM's response if stream is True.
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
            stream=True, # Always stream from requests, handle aggregation/yielding based on 'stream' arg
            timeout=settings.OLLAMA_TIMEOUT
        )
        response.raise_for_status()

        if stream:
            def generate():
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        try:
                            json_chunk = json.loads(decoded_line)
                            if 'response' in json_chunk:
                                yield json_chunk['response']
                            if json_chunk.get('done', False):
                                break
                        except json.JSONDecodeError:
                            logger.warning(f"Non-JSON line in stream: {decoded_line}")
            logger.info("Streaming response from Ollama.")
            return generate()
        else:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    try:
                        json_chunk = json.loads(decoded_line)
                        if 'response' in json_chunk:
                            full_response += json_chunk['response']
                        if json_chunk.get('done', False):
                            break
                    except json.JSONDecodeError:
                        logger.warning(f"Non-JSON line in stream: {decoded_line}")
            logger.info("Streamed response received and aggregated")
            return full_response.strip()

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

[답변 작성 지침]
- 답변은 마크다운(Markdown) 형식으로 작성해주세요. 제목(#, ##), 목록(-), 강조(**), 표 등을 활용하세요.
- **주조 전문 용어를 정확하게 사용하세요**: 주형, 탕구, 라이저, 코어, 패턴, 결함, 사형, 금형 등의 전문 용어를 정확히 사용해주세요.
- 답변 구조를 체계적으로 구성하세요 (개요 → 상세설명 → 요약).
- 예시나 구체적인 수치가 있다면 포함해주세요.
- **출처 표시는 답변 마지막에 참고문헌 형태로 한 번만 표시하세요**.

[출처 표시 규칙]
- 답변 본문에는 출처를 반복 표시하지 마세요.
- 답변 마지막에 "## 📚 참고문헌" 섹션을 만들어 출처를 정리해주세요.
- 형식: "- 📄 문서ID: [관련내용 요약]"

[품질 기준]
- 명확하고 이해하기 쉬운 설명
- 논리적이고 체계적인 구성
- 실무에 도움이 되는 구체적인 정보 제공
- 한국어로 작성

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

def construct_multimodal_rag_prompt(query: str, 
                                   context_chunks: List[str], 
                                   image_descriptions: List[str] = None,
                                   table_contents: List[str] = None,
                                   lang: str = "ko") -> str:
    """
    Constructs a multimodal prompt for the LLM using the RAG pattern.
    """
    if image_descriptions is None:
        image_descriptions = []
    if table_contents is None:
        table_contents = []
    
    # Prepare different content sections
    content_sections = []
    
    if context_chunks:
        text_context = "\n\n---\n\n".join(context_chunks)
        content_sections.append(f"텍스트 정보:\n{text_context}")
    
    # Image context temporarily disabled
    # if image_descriptions:
    #     image_context = "\n\n---\n\n".join(image_descriptions)
    #     content_sections.append(f"이미지 정보:\n{image_context}")
    
    if table_contents:
        table_context = "\n\n---\n\n".join(table_contents)
        content_sections.append(f"표 정보:\n{table_context}")
    
    all_content = "\n\n=== === ===\n\n".join(content_sections)
    
    if lang == "ko":
        prompt = f"""주어진 멀티모달 컨텐트 정보를 바탕으로 다음 질문에 답변해 주세요.

{all_content}
---

[질문]
{query}

[답변 작성 지침]
- 답변은 마크다운(Markdown) 형식으로 작성해주세요. 제목(#, ##), 목록(-), 강조(**), 표 등을 활용하세요.
- **주조 전문 용어를 정확하게 사용하세요**: 주형, 탕구, 라이저, 코어, 패턴, 결함, 사형, 금형, 쿠폴라, 용해, 응고 등의 전문 용어를 정확히 사용해주세요.
- 답변 구조를 체계적으로 구성하세요 (개요 → 상세설명 → 요약).
- 표를 참조할 때는 [표1], [표2] 형식을 사용하고, 중요한 수치나 경향을 분석해서 설명해주세요.
- **출처 표시는 답변 마지막에 참고문헌 형태로 한 번만 표시하세요**.

[출처 표시 규칙]
- 답변 본문에는 출처를 반복 표시하지 마세요.
- 답변 마지막에 "## 📚 참고문헌" 섹션을 만들어 출처를 정리해주세요.
- 텍스트 출처: "- 📄 문서ID: [관련내용 요약]"
- 표 출처: "- 📊 표N (문서ID): [표 내용 요약]"

[품질 기준]
- 명확하고 이해하기 쉬운 설명
- 논리적이고 체계적인 구성
- 실무에 도움이 되는 구체적인 정보 제공
- 한국어로 작성

답변:
"""
    else:
        prompt = f"""Please answer the following question based on the given multimodal content information.

{all_content}
---

[Question]
{query}

[Instructions]
- Write your answer in Markdown format. Use tables, code, emphasis, lists, links, etc. as appropriate.
- Each content piece has a [Source:DocumentID] tag. Be sure to include this information in your answer to clearly indicate the source.
- Integrate text, image, and table information to provide a comprehensive answer to the question.
- If images or tables are available, reference their content in your explanation using phrases like "As shown in the image above..." or "According to the data in the table..."
- Your answer should be a complete sentence, clear, and as detailed and long as possible.
- If possible, include examples, tables, code, or additional explanations.
- If you cannot find the answer to the question in the provided content, please respond with "It is difficult to answer the question based on the provided content information alone."
- Do not add personal opinions or information not present in the content.
- If there are charts or graphs in images, describe their content specifically.
- If there is table data, analyze and explain important figures or trends.
- Please answer in English.

Answer:
"""
    
    return prompt

def enhance_response_with_media_references(response: str, 
                                         images: List[Dict[str, Any]], 
                                         tables: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Enhances the LLM response with references to images and tables that can be displayed in the UI.
    
    Returns:
        Dict containing enhanced response with media references
    """
    enhanced_response = {
        "text": response,
        "referenced_images": [],
        "referenced_tables": [],
        "has_media": False
    }
    
    # Image references temporarily disabled
    # for idx, img in enumerate(images, 1):
    #     meta = img.get('metadata', {})
    #     enhanced_response["referenced_images"].append({
    #         "index": idx,
    #         "filename": meta.get('filename', ''),
    #         "path": meta.get('file_path', ''),
    #         "page": meta.get('page', ''),
    #         "source": meta.get('source_document_id', ''),
    #         "description": img.get('description', '')
    #     })
    
    # Add all available tables with index numbers for UI display
    for idx, table in enumerate(tables, 1):
        meta = table.get('metadata', {})
        enhanced_response["referenced_tables"].append({
            "index": idx,
            "filename": meta.get('filename', ''),
            "path": meta.get('file_path', ''),
            "page": meta.get('page', ''),
            "source": meta.get('source_document_id', ''),
            "content": table.get('content', ''),
            "parsed_data": table.get('parsed_data', [])
        })
    
    enhanced_response["has_media"] = bool(
        enhanced_response["referenced_tables"]
    )
    
    return enhanced_response

# 비동기 LLM 처리 함수들
async def get_llm_response_async(
    prompt: str, 
    model_name: str = None, 
    options: Dict[str, Any] = None, 
    stream: bool = False
):
    """
    비동기 LLM 응답 생성
    """
    if model_name is None:
        model_name = settings.OLLAMA_DEFAULT_MODEL

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": stream,
    }
    if options:
        payload["options"] = options

    logger.info(f"Sending async prompt to Ollama model '{model_name}'. Prompt length: {len(prompt)} chars")

    try:
        timeout = aiohttp.ClientTimeout(total=settings.OLLAMA_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(settings.OLLAMA_API_URL, json=payload) as response:
                response.raise_for_status()
                
                if stream:
                    async def generate():
                        async for line in response.content:
                            if line:
                                try:
                                    decoded_line = line.decode('utf-8').strip()
                                    if decoded_line:
                                        json_chunk = json.loads(decoded_line)
                                        if 'response' in json_chunk:
                                            yield json_chunk['response']
                                        if json_chunk.get('done', False):
                                            break
                                except json.JSONDecodeError:
                                    logger.warning(f"Non-JSON line in async stream: {decoded_line}")
                                    continue
                    return generate()
                else:
                    # 비스트리밍 모드
                    full_response = ""
                    async for line in response.content:
                        if line:
                            try:
                                decoded_line = line.decode('utf-8').strip()
                                if decoded_line:
                                    json_chunk = json.loads(decoded_line)
                                    if 'response' in json_chunk:
                                        full_response += json_chunk['response']
                                    if json_chunk.get('done', False):
                                        break
                            except json.JSONDecodeError:
                                logger.warning(f"Non-JSON line in async response: {decoded_line}")
                                continue
                    return full_response

    except aiohttp.ClientConnectorError as e:
        error_msg = f"Could not connect to Ollama at {settings.OLLAMA_API_URL}. Ensure Ollama is running."
        logger.error(f"{error_msg} Details: {e}")
        raise LLMError(error_msg, "CONNECTION_ERROR")
    except aiohttp.ClientResponseError as e:
        error_msg = f"HTTP error occurred: {e}"
        logger.error(f"{error_msg}. Status: {e.status}")
        raise LLMError(error_msg, "HTTP_ERROR")
    except asyncio.TimeoutError as e:
        error_msg = f"Async request timed out after {settings.OLLAMA_TIMEOUT} seconds"
        logger.error(f"{error_msg}: {e}")
        raise LLMError(error_msg, "TIMEOUT_ERROR")
    except Exception as e:
        error_msg = f"An unexpected async error occurred: {e}"
        logger.error(error_msg)
        raise LLMError(error_msg, "UNEXPECTED_ERROR")

async def process_multimodal_rag_async(
    query: str, 
    search_results: Dict[str, List[Dict[str, Any]]], 
    model_name: str = None, 
    lang: str = "ko"
) -> Dict[str, Any]:
    """
    비동기 멀티모달 RAG 처리
    """
    # 컨텍스트 압축
    text_chunks = [result['document'] for result in search_results.get('text', [])]
    compressed_context = compress_context_for_performance(text_chunks, max_tokens=2000)
    
    # 프롬프트 구성
    prompt = construct_rag_prompt(query, compressed_context, lang)
    
    # 비동기 LLM 호출
    response = await get_llm_response_async(prompt, model_name)
    
    # 응답 강화
    enhanced_response = enhance_response_with_metadata(
        response, search_results.get('text', []), [], search_results.get('tables', [])
    )
    
    return enhanced_response

def compress_context_for_performance(text_chunks: List[str], max_tokens: int = 2000) -> List[str]:
    """
    성능을 위한 의미 기반 컨텍스트 압축
    """
    if not text_chunks:
        return []
    
    # 토큰 수 추정 (1 토큰 ≈ 4 글자)
    total_chars = sum(len(chunk) for chunk in text_chunks)
    estimated_tokens = total_chars // 4
    
    if estimated_tokens <= max_tokens:
        return text_chunks
    
    logger.info(f"Context compression needed: {estimated_tokens} -> {max_tokens} tokens")
    
    # 의미 기반 중요도 점수 계산
    scored_chunks = []
    for i, chunk in enumerate(text_chunks):
        score = _calculate_chunk_importance(chunk, i, len(text_chunks))
        scored_chunks.append((chunk, score))
    
    # 중요도 순으로 정렬
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # 토큰 제한 내에서 가장 중요한 청크들 선택
    compressed_chunks = []
    current_tokens = 0
    max_chars = max_tokens * 4
    
    for chunk, score in scored_chunks:
        chunk_tokens = len(chunk) // 4
        
        if current_tokens + chunk_tokens <= max_tokens:
            compressed_chunks.append(chunk)
            current_tokens += chunk_tokens
        else:
            # 남은 공간이 있으면 중요한 부분만 부분 포함
            remaining_chars = max_chars - (current_tokens * 4)
            if remaining_chars > 200:  # 최소 200자 이상일 때만
                # 청크의 첫 부분과 마지막 부분 포함 (요약 효과)
                half_remaining = remaining_chars // 2
                truncated = chunk[:half_remaining] + " ... " + chunk[-half_remaining:]
                compressed_chunks.append(truncated)
            break
    
    # 원래 순서대로 재정렬 (시간 순서 유지)
    chunk_to_index = {chunk: i for i, chunk in enumerate(text_chunks)}
    compressed_chunks.sort(key=lambda x: _get_original_index(x, chunk_to_index))
    
    logger.info(f"Compressed context: {len(text_chunks)} -> {len(compressed_chunks)} chunks")
    return compressed_chunks

def _calculate_chunk_importance(chunk: str, position: int, total_chunks: int) -> float:
    """
    청크의 중요도 점수 계산
    """
    import re
    
    score = 0.0
    
    # 1. 주조 전문용어 포함 정도 (가중치: 3.0)
    foundry_keywords = [
        '주형', '탕구', '라이저', '코어', '패턴', '사형', '금형', '결함', '수축', '응고',
        '용해', '쿠폴라', '조형', '주입', '캐스팅', '몰드', '스프루', '게이트', '런너',
        '벤트', '이형제', '바인더', '규사', '점토', '벤토나이트', '기포', '균열', '핀홀',
        'casting', 'mold', 'core', 'pattern', 'defect', 'shrinkage', 'porosity'
    ]
    
    keyword_count = sum(1 for keyword in foundry_keywords if keyword.lower() in chunk.lower())
    score += keyword_count * 3.0
    
    # 2. 숫자/데이터 포함 정도 (가중치: 2.0)
    numbers = re.findall(r'\d+(?:\.\d+)?', chunk)
    score += len(numbers) * 0.5
    
    # 3. 표 참조나 이미지 참조 (가중치: 2.5)
    table_refs = re.findall(r'표\s*\d+|Table\s*\d+|\[표\s*\d+\]', chunk, re.IGNORECASE)
    image_refs = re.findall(r'그림\s*\d+|Figure\s*\d+|\[그림\s*\d+\]', chunk, re.IGNORECASE)
    score += (len(table_refs) + len(image_refs)) * 2.5
    
    # 4. 텍스트 길이 (가중치: 1.0) - 더 긴 텍스트는 더 많은 정보 포함
    score += len(chunk) / 1000.0
    
    # 5. 위치 기반 가중치 - 문서 시작/끝 부분은 중요할 가능성 높음
    position_weight = 1.0
    if position < total_chunks * 0.2:  # 앞 20%
        position_weight = 1.2
    elif position > total_chunks * 0.8:  # 뒤 20%
        position_weight = 1.1
    
    score *= position_weight
    
    # 6. 문장 완성도 (가중치: 1.5)
    sentences = re.split(r'[.!?]', chunk)
    complete_sentences = [s for s in sentences if s.strip() and len(s.strip()) > 10]
    score += len(complete_sentences) * 1.5
    
    return score

def _get_original_index(chunk: str, chunk_to_index: dict) -> int:
    """
    원래 청크 인덱스 찾기 (부분 일치 고려)
    """
    # 정확히 일치하는 경우
    if chunk in chunk_to_index:
        return chunk_to_index[chunk]
    
    # 부분 일치하는 경우 (압축된 청크)
    for original_chunk, index in chunk_to_index.items():
        if chunk.startswith(original_chunk[:100]):  # 첫 100자로 매칭
            return index
    
    return 999999  # 찾지 못한 경우 맨 뒤로

def process_llm_chat_request(user_query: str,
                             retrieved_content: Dict[str, Any],
                             model_name: str = None,
                             lang: str = "ko",
                             stream: bool = False) -> Dict[str, Any] | Generator[Dict[str, Any], None, None]:
    """
    High-level function to process a chat request using RAG, supporting both text and multimodal content.
    1. Constructs a prompt from the user query and retrieved documents/multimodal content.
    2. Gets a response from the LLM.
    3. Enhances the response with media references.

    Args:
        user_query (str): The user's question.
        retrieved_content (Dict[str, Any]): Dict containing 'text_chunks', 'images', 'tables'.
                                            For text-only, only 'text_chunks' is needed.
        model_name (str, optional): Ollama model name. Defaults to settings.OLLAMA_DEFAULT_MODEL.
        lang (str, optional): Language for the prompt. Defaults to "ko".
        stream (bool, optional): Whether to stream the response. Defaults to False.

    Returns:
        Dict[str, Any]: The LLM's response, potentially with media references, if stream is False.
        Generator[Dict[str, Any], None, None]: A generator that yields chunks of the LLM's response,
                                                potentially with media references, if stream is True.
    """
    text_chunks = retrieved_content.get('text', [])
    images = retrieved_content.get('images', [])
    tables = retrieved_content.get('tables', [])

    if not any([text_chunks, images, tables]):
        if lang == "ko":
            return {"text": "관련 컨텐트를 찾을 수 없어 답변을 생성할 수 없습니다.", "referenced_images": [], "referenced_tables": [], "has_media": False}
        else:
            return {"text": "No relevant content found to generate an answer.", "referenced_images": [], "referenced_tables": [], "has_media": False}

    # Prepare context for prompt construction
    context_chunks_for_prompt = []
    for doc in text_chunks:
        text = doc.get("text", "")
        meta = doc.get("metadata", {})
        source = meta.get("source_document_id", "?")
        chunk_index = meta.get("chunk_index", "?")
        if text:
            context_chunks_for_prompt.append(f"[출처:{source}_청크{chunk_index}]\n{text}")

    # Determine if it's a multimodal request
    is_multimodal = bool(images or tables)

    if is_multimodal:
        prompt = construct_multimodal_rag_prompt(
            user_query, 
            context_chunks_for_prompt, 
            image_descriptions=[], # Image descriptions are not used in prompt for now
            table_contents=[f"[표{idx}] [출처:{t.get('metadata', {}).get('source_document_id', '?')}_페이지{t.get('metadata', {}).get('page', '?')}]\n{t.get('content', '')}" for idx, t in enumerate(tables, 1)],
            lang=lang
        )
        ollama_options = {
            "temperature": settings.LLM_TEMPERATURE,
            "num_predict": settings.LLM_NUM_PREDICT_MULTIMODAL,
        }
    else:
        prompt = construct_rag_prompt(user_query, context_chunks_for_prompt, lang=lang)
        ollama_options = {
            "temperature": settings.LLM_TEMPERATURE,
            "num_predict": settings.LLM_NUM_PREDICT_TEXT,
        }

    try:
        if stream:
            def streaming_response_generator():
                first_chunk = True
                for chunk_text in get_llm_response(prompt, model_name=model_name, options=ollama_options, stream=True):
                    if first_chunk:
                        # Send initial metadata (referenced images/tables) with the first text chunk
                        initial_response = enhance_response_with_media_references("", images, tables)
                        initial_response["text"] = chunk_text # Add first text chunk
                        yield initial_response
                        first_chunk = False
                    else:
                        yield {"text": chunk_text, "referenced_images": [], "referenced_tables": [], "has_media": False} # Subsequent chunks only contain text
            return streaming_response_generator()
        else:
            llm_response_text = get_llm_response(prompt, model_name=model_name, options=ollama_options, stream=False)
            logger.info(f"Successfully generated LLM response for query: {user_query[:50]}...")
            
            # Enhance response with media references if multimodal
            return enhance_response_with_media_references(llm_response_text, images, tables)

    except LLMError as e:
        logger.error(f"LLM error in chat request: {e}")
        if lang == "ko":
            return {"text": f"답변 생성 중 오류가 발생했습니다: {e.message}", "referenced_images": [], "referenced_tables": [], "has_media": False}
        else:
            return {"text": f"An error occurred while generating response: {e.message}", "referenced_images": [], "referenced_tables": [], "has_media": False}


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

    # --- Test Multimodal RAG Prompt Construction ---
    print("\n--- Testing Multimodal RAG Prompt Construction (Korean) ---")
    sample_multimodal_query_ko = "이 문서의 주요 내용은 무엇이며, 표에는 어떤 데이터가 있나요?"
    sample_multimodal_text_chunks = [
        {"text": "주조 공정은 금속을 녹여 거푸집에 부어 제품을 만드는 과정입니다.", "metadata": {"source_document_id": "doc1", "chunk_index": 0}},
        {"text": "주요 결함으로는 기포, 수축, 균열 등이 있습니다.", "metadata": {"source_document_id": "doc1", "chunk_index": 1}}
    ]
    sample_multimodal_images = [] # Temporarily disabled
    sample_multimodal_tables = [
        {"content": "온도 | 강도\n1000C | 200MPa\n1200C | 150MPa", "metadata": {"source_document_id": "doc1", "page": 5}, "parsed_data": [["온도", "강도"], ["1000C", "200MPa"], ["1200C", "150MPa"]]}
    ]
    
    multimodal_content_for_test = {
        "text": sample_multimodal_text_chunks,
        "images": sample_multimodal_images,
        "tables": sample_multimodal_tables
    }

    print("\n--- Testing process_llm_chat_request (non-streaming) ---")
    multimodal_response_non_stream = process_llm_chat_request(
        sample_multimodal_query_ko,
        multimodal_content_for_test,
        model_name=settings.OLLAMA_DEFAULT_MODEL,
        lang="ko",
        stream=False
    )
    print(f"Processed LLM Chat Response (non-streaming):\n{multimodal_response_non_stream['text']}")
    if multimodal_response_non_stream['referenced_tables']:
        print(f"Referenced Tables: {multimodal_response_non_stream['referenced_tables']}")

    print("\n--- Testing process_llm_chat_request (streaming) ---")
    print("Streaming response:")
    multimodal_response_stream = process_llm_chat_request(
        sample_multimodal_query_ko,
        multimodal_content_for_test,
        model_name=settings.OLLAMA_DEFAULT_MODEL,
        lang="ko",
        stream=True
    )
    full_streamed_response = ""
    for chunk in multimodal_response_stream:
        print(chunk.get("text", ""), end='')
        full_streamed_response += chunk.get("text", "")
        if chunk.get("referenced_tables"):
            print(f"\n(Referenced Tables in stream: {chunk['referenced_tables']})\n")
    print(f"\nFull streamed response length: {len(full_streamed_response)}")

    print("\nTo run LLM call tests, ensure your Ollama server is running and the model is available.")
