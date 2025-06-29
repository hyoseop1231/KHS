"""
Streaming service for real-time LLM response delivery
"""
from typing import Dict, Any, Generator
from app.services.llm_service import get_llm_response, construct_multimodal_rag_prompt
from app.config import settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

def process_multimodal_llm_chat_request_stream(
    user_query: str,
    multimodal_content: Dict[str, Any],
    model_name: str = None,
    lang: str = "ko",
    options: Dict = None
) -> Generator[str, None, None]:
    """
    Process multimodal chat request with streaming response.
    
    Args:
        user_query: User's question
        multimodal_content: Retrieved content (text, images, tables)
        model_name: LLM model to use
        lang: Language for response
        options: LLM options
        
    Yields:
        str: Streaming response chunks
    """
    try:
        # Extract content from multimodal data - handle both key formats for compatibility
        text_data = multimodal_content.get("text", multimodal_content.get("text_chunks", []))
        if isinstance(text_data, list) and text_data and isinstance(text_data[0], dict):
            # Format: [{"text": "...", "metadata": {...}}, ...]
            context_chunks = [chunk.get("text", "") for chunk in text_data]
        elif isinstance(text_data, list):
            # Format: ["text1", "text2", ...]
            context_chunks = text_data
        else:
            context_chunks = []
        
        images = multimodal_content.get("images", [])
        image_descriptions = [img.get("description", "") for img in images]
        
        tables = multimodal_content.get("tables", [])
        table_contents = [table.get("content", "") for table in tables]
        
        # Construct the multimodal prompt
        prompt = construct_multimodal_rag_prompt(
            user_query,
            context_chunks,
            image_descriptions,
            table_contents,
            lang
        )
        
        logger.info(f"Starting streaming response for query: {user_query[:50]}...")
        
        # Get streaming response from LLM
        stream_generator = get_llm_response(
            prompt=prompt,
            model_name=model_name,
            options=options,
            stream=True  # Enable streaming
        )
        
        # Yield each chunk
        for chunk in stream_generator:
            if chunk:  # Only yield non-empty chunks
                yield chunk
                
        logger.info("Streaming response completed")
        
    except Exception as e:
        logger.error(f"Error in streaming response: {e}")
        yield f"\n\n[오류: {str(e)}]"