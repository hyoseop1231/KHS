"""
Multimodal LLM Service
Enhanced LLM service that handles text, images, and tables in responses.
"""

import requests
import json
from typing import List, Dict, Any
from app.config import settings
from app.utils.logging_config import get_logger
from app.utils.exceptions import LLMError
from app.services.llm_service import get_llm_response

logger = get_logger(__name__)

def process_multimodal_llm_chat_request(user_query: str,
                                      multimodal_content: Dict[str, Any],
                                      model_name: str = None,
                                      lang: str = "ko") -> str:
    """
    Enhanced function to process chat requests with multimodal content.
    
    Args:
        user_query (str): The user's question.
        multimodal_content (Dict[str, Any]): Dict containing 'text_chunks', 'images', 'tables'.
        model_name (str, optional): Ollama model name.
        lang (str, optional): Language for the prompt.

    Returns:
        str: The LLM's response.
    """
    # Check if any content is available
    text_chunks = multimodal_content.get('text_chunks', [])
    images = multimodal_content.get('images', [])
    tables = multimodal_content.get('tables', [])
    
    if not any([text_chunks, images, tables]):
        if lang == "ko":
            return "관련 컨텐트를 찾을 수 없어 답변을 생성할 수 없습니다."
        else:
            return "No relevant content found to generate an answer."
    
    # Prepare multimodal context
    context_chunks = []
    image_descriptions = []
    table_contents = []
    
    # Process text chunks
    for doc in text_chunks:
        text = doc.get("text", "")
        meta = doc.get("metadata", {})
        source = meta.get("source_document_id", "?")
        chunk_index = meta.get("chunk_index", "?")
        
        if text:
            tagged_chunk = f"[출처:{source}_청크{chunk_index}]\n{text}"
            context_chunks.append(tagged_chunk)
    
    # Image processing temporarily disabled
    # for idx, img in enumerate(images, 1):
    #     meta = img.get('metadata', {})
    #     source = meta.get('source_document_id', '?')
    #     page = meta.get('page', '?')
    #     description = img.get('description', '')
    #     filename = meta.get('filename', '')
    #     
    #     if description or filename:
    #         img_desc = f"[이미지{idx}] [출처:{source}_페이지{page}]\n파일명: {filename}\n설명: {description}"
    #         image_descriptions.append(img_desc)
    
    # Process tables with numbered references
    for idx, table in enumerate(tables, 1):
        meta = table.get('metadata', {})
        source = meta.get('source_document_id', '?')
        page = meta.get('page', '?')
        content = table.get('content', '')
        parsed_data = table.get('parsed_data', [])
        
        if content or parsed_data:
            table_content = f"[표{idx}] [출처:{source}_페이지{page}]\n{content}"
            if parsed_data:
                # Format table data nicely
                table_rows = []
                for row in parsed_data[:5]:  # Limit to first 5 rows
                    if isinstance(row, list):
                        table_rows.append(' | '.join(str(cell) for cell in row))
                if table_rows:
                    table_content += "\n\n표 데이터:\n" + '\n'.join(table_rows)
                    if len(parsed_data) > 5:
                        table_content += f"\n(... 총 {len(parsed_data)}행)"
            table_contents.append(table_content)
    
    # Construct multimodal prompt
    prompt = construct_multimodal_rag_prompt(
        user_query, 
        context_chunks, 
        image_descriptions, 
        table_contents, 
        lang
    )
    
    # Ollama 모델 옵션
    ollama_options = {
        "temperature": 0.8,
        "num_predict": 3072,  # Increased for detailed multimodal responses
    }
    
    try:
        response = get_llm_response(prompt, model_name, options=ollama_options)
        
        logger.info(f"Multimodal LLM response generated. Length: {len(response)} chars")
        logger.info(f"  - Text chunks: {len(context_chunks)}")
        logger.info(f"  - Images: {len(image_descriptions)}")
        logger.info(f"  - Tables: {len(table_contents)}")
        logger.debug(f"Response preview: {response[:200]}...")
        
        return response
        
    except LLMError as e:
        logger.error(f"Multimodal LLM error: {e}")
        if lang == "ko":
            return f"멀티모달 답변 생성 중 오류가 발생했습니다: {e.message}"
        else:
            return f"Error generating multimodal response: {e.message}"

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

[지침]
- 답변은 마크다운(Markdown) 형식으로 작성해주세요. 표, 코드, 강조, 리스트, 링크 등 다양한 서식을 적극적으로 활용하세요.
- 각 컨텐트에 [출처:문서ID] 태그가 있으니, 답변에 해당 정보를 반드시 포함하여 출처를 명확히 표시하세요.
- 표를 참조할 때는 반드시 [표1], [표2] 등의 번호를 사용하세요. 예: "[표1]에 나타난 데이터에 따르면..."
- 텍스트와 표 정보를 종합하여 질문에 대한 완전한 답변을 생성해주세요.
- 답변은 완전한 문장으로, 명확하고 친절하게, 최대한 자세하고 길게 작성해주세요.
- 예시, 표, 코드, 추가 설명이 가능하다면 포함해주세요.
- 만약 제공된 컨텐트에서 질문에 대한 답을 찾을 수 없다면, "제공된 컨텐트 정보만으로는 질문에 답변하기 어렵습니다."라고 답변해주세요.
- 답변에 개인적인 의견이나 컨텐트 정보에 없는 내용을 추가하지 마세요.
- 표 데이터가 있다면 중요한 수치나 경향을 분석해서 설명해주세요.

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