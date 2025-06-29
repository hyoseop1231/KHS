import requests
import json
from typing import List, Dict, Any, Optional, Generator
from app.config import settings
from app.utils.logging_config import get_logger
from app.utils.exceptions import LLMError

logger = get_logger(__name__)

def construct_multimodal_rag_prompt(
    user_query: str,
    context_chunks: List[str],
    image_descriptions: List[str],
    table_contents: List[str],
    lang: str = "ko"
) -> str:
    """
    Construct a multimodal RAG prompt combining text, images, and tables.
    
    Args:
        user_query: User's query
        context_chunks: List of relevant text chunks
        image_descriptions: List of image descriptions
        table_contents: List of table contents
        lang: Language code (default: "ko")
    
    Returns:
        str: Constructed prompt for LLM
    """
    if lang == "ko":
        prompt_parts = [
            "다음은 주조 기술 문서에서 추출한 관련 정보입니다:",
            "",
            "=== 텍스트 정보 ===",
        ]
        
        for i, chunk in enumerate(context_chunks, 1):
            prompt_parts.append(f"텍스트 {i}: {chunk}")
            prompt_parts.append("")
        
        if image_descriptions:
            prompt_parts.append("=== 이미지 정보 ===")
            for i, desc in enumerate(image_descriptions, 1):
                if desc.strip():
                    prompt_parts.append(f"이미지 {i}: {desc}")
            prompt_parts.append("")
        
        if table_contents:
            prompt_parts.append("=== 표 정보 ===")
            for i, table in enumerate(table_contents, 1):
                if table.strip():
                    prompt_parts.append(f"표 {i}: {table}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "위의 정보를 바탕으로 다음 질문에 답해주세요:",
            f"질문: {user_query}",
            "",
            "답변 시 다음 사항을 준수해주세요:",
            "1. 제공된 정보만을 사용하여 정확하고 구체적으로 답변하세요",
            "2. 관련 이미지나 표가 있다면 해당 내용을 참조하여 설명하세요",
            "3. 정보가 부족하거나 없는 경우 솔직히 말씀해주세요",
            "4. 마크다운 형식으로 구조화하여 답변하세요",
            "5. 전문용어는 이해하기 쉽게 설명해주세요",
            "",
            "답변:"
        ])
    else:
        prompt_parts = [
            "The following is relevant information extracted from foundry technology documents:",
            "",
            "=== Text Information ===",
        ]
        
        for i, chunk in enumerate(context_chunks, 1):
            prompt_parts.append(f"Text {i}: {chunk}")
            prompt_parts.append("")
        
        if image_descriptions:
            prompt_parts.append("=== Image Information ===")
            for i, desc in enumerate(image_descriptions, 1):
                if desc.strip():
                    prompt_parts.append(f"Image {i}: {desc}")
            prompt_parts.append("")
        
        if table_contents:
            prompt_parts.append("=== Table Information ===")
            for i, table in enumerate(table_contents, 1):
                if table.strip():
                    prompt_parts.append(f"Table {i}: {table}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "Based on the above information, please answer the following question:",
            f"Question: {user_query}",
            "",
            "Please follow these guidelines when answering:",
            "1. Use only the provided information to give accurate and specific answers",
            "2. If there are related images or tables, refer to them in your explanation",
            "3. Be honest if information is insufficient or unavailable",
            "4. Structure your answer using markdown format",
            "5. Explain technical terms in an easy-to-understand way",
            "",
            "Answer:"
        ])
    
    return "\n".join(prompt_parts)

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

        if response.status_code != 200:
            error_msg = f"Ollama API request failed with status {response.status_code}: {response.text}"
            logger.error(error_msg)
            raise LLMError(error_msg)

        if stream:
            # Stream response - yield chunks as they come
            logger.debug("Streaming response from Ollama...")
            for line in response.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line.decode('utf-8'))
                        if 'response' in json_response:
                            yield json_response['response']
                        if json_response.get('done', False):
                            break
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON response line: {line}, error: {e}")
                        continue
        else:
            # Non-streaming response - collect all chunks and return as string
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line.decode('utf-8'))
                        if 'response' in json_response:
                            full_response += json_response['response']
                        if json_response.get('done', False):
                            break
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON response line: {line}, error: {e}")
                        continue
            
            logger.info(f"Received complete response from Ollama. Length: {len(full_response)} chars")
            return full_response.strip()

    except requests.exceptions.Timeout:
        error_msg = f"Ollama request timed out after {settings.OLLAMA_TIMEOUT} seconds"
        logger.error(error_msg)
        raise LLMError(error_msg)
    except requests.exceptions.ConnectionError:
        error_msg = "Failed to connect to Ollama. Please ensure Ollama is running."
        logger.error(error_msg)
        raise LLMError(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error during Ollama request: {str(e)}"
        logger.error(error_msg)
        raise LLMError(error_msg)

def get_llm_response_async(prompt: str, model_name: str = None, options: Dict = None) -> str:
    """
    Asynchronous version of get_llm_response for non-streaming requests.
    
    Args:
        prompt (str): The prompt to send to the LLM.
        model_name (str, optional): The name of the Ollama model to use.
        options (Dict, optional): Additional Ollama model parameters.
    
    Returns:
        str: The LLM's response text.
    """
    return get_llm_response(prompt, model_name, options, stream=False)