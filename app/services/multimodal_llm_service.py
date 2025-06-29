from typing import Dict, Any, List, Optional

from app.services.llm_service import get_llm_response, construct_multimodal_rag_prompt
from app.utils.sanitizer import sanitize_llm_response

def process_multimodal_llm_chat_request(
    user_query: str,
    multimodal_content: Dict[str, Any],
    model_name: Optional[str] = None,
    lang: str = "ko",
    options: Dict = None
) -> str:
    """
    Processes a multimodal chat request by constructing a multimodal RAG prompt
    and querying the LLM.
    """
    # Extract text chunks - handle both key formats for compatibility
    text_data = multimodal_content.get("text_chunks", multimodal_content.get("text", []))
    if isinstance(text_data, list) and text_data and isinstance(text_data[0], dict):
        # Format: [{"text": "...", "metadata": {...}}, ...]
        context_chunks = [chunk.get("text", "") for chunk in text_data]
    elif isinstance(text_data, list):
        # Format: ["text1", "text2", ...]
        context_chunks = text_data
    else:
        context_chunks = []

    # Extract image descriptions
    images = multimodal_content.get("images", [])
    image_descriptions = [img.get("description", "") for img in images]

    # Extract table contents
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

    # Query the LLM with custom options
    response = get_llm_response(prompt, model_name=model_name, options=options)
    
    # Sanitize the response before returning
    sanitized_response = sanitize_llm_response(response, content_type="markdown")
    return sanitized_response

def enhance_response_with_media_references(
    llm_response_text: str,
    retrieved_images: List[Dict[str, Any]],
    retrieved_tables: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Enhances the LLM response by extracting media references for UI display.
    Analyzes the response text for table/image references and provides metadata.
    """
    import re
    
    referenced_images = []
    referenced_tables = []
    
    # Process tables - look for table references in response
    table_pattern = r'\[표\s*(\d+)\]|\[Table\s*(\d+)\]|표\s*(\d+)|Table\s*(\d+)'
    table_matches = re.finditer(table_pattern, llm_response_text, re.IGNORECASE)
    
    referenced_table_indices = set()
    for match in table_matches:
        # Extract table number from any of the capturing groups
        table_num = next((g for g in match.groups() if g is not None), None)
        if table_num:
            try:
                idx = int(table_num) - 1  # Convert to 0-based index
                if 0 <= idx < len(retrieved_tables):
                    referenced_table_indices.add(idx)
            except ValueError:
                continue
    
    # Add referenced tables with metadata
    for idx in referenced_table_indices:
        table = retrieved_tables[idx]
        metadata = table.get('metadata', {})
        referenced_tables.append({
            "index": idx + 1,
            "filename": metadata.get('filename', ''),
            "path": metadata.get('file_path', ''),
            "page": metadata.get('page', ''),
            "source": metadata.get('source_document_id', ''),
            "content": table.get('content', ''),
            "parsed_data": table.get('parsed_data', [])
        })
    
    # Process images - look for image references in response  
    image_pattern = r'\[이미지\s*(\d+)\]|\[Image\s*(\d+)\]|이미지\s*(\d+)|Image\s*(\d+)'
    image_matches = re.finditer(image_pattern, llm_response_text, re.IGNORECASE)
    
    referenced_image_indices = set()
    for match in image_matches:
        # Extract image number from any of the capturing groups
        image_num = next((g for g in match.groups() if g is not None), None)
        if image_num:
            try:
                idx = int(image_num) - 1  # Convert to 0-based index
                if 0 <= idx < len(retrieved_images):
                    referenced_image_indices.add(idx)
            except ValueError:
                continue
    
    # Add referenced images with metadata
    for idx in referenced_image_indices:
        image = retrieved_images[idx]
        metadata = image.get('metadata', {})
        referenced_images.append({
            "index": idx + 1,
            "filename": metadata.get('filename', ''),
            "path": metadata.get('file_path', ''),
            "page": metadata.get('page', ''),
            "source": metadata.get('source_document_id', ''),
            "description": image.get('description', '')
        })
    
    # If no explicit references found, include all available media
    if not referenced_tables and retrieved_tables:
        for idx, table in enumerate(retrieved_tables):
            metadata = table.get('metadata', {})
            referenced_tables.append({
                "index": idx + 1,
                "filename": metadata.get('filename', ''),
                "path": metadata.get('file_path', ''),
                "page": metadata.get('page', ''),
                "source": metadata.get('source_document_id', ''),
                "content": table.get('content', ''),
                "parsed_data": table.get('parsed_data', [])
            })
    
    if not referenced_images and retrieved_images:
        for idx, image in enumerate(retrieved_images):
            metadata = image.get('metadata', {})
            referenced_images.append({
                "index": idx + 1,
                "filename": metadata.get('filename', ''),
                "path": metadata.get('file_path', ''),
                "page": metadata.get('page', ''),
                "source": metadata.get('source_document_id', ''),
                "description": image.get('description', '')
            })
    
    has_media = bool(referenced_images or referenced_tables)
    
    return {
        "text": llm_response_text,
        "referenced_images": referenced_images,
        "referenced_tables": referenced_tables,
        "has_media": has_media
    }