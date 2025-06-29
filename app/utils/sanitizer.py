"""Content sanitization utilities for security"""

import re
import html
from typing import Optional
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

class ContentSanitizer:
    """Sanitizes content to prevent XSS and other injection attacks"""
    
    # Allowed HTML tags for basic formatting (very restrictive)
    ALLOWED_TAGS = {
        'p', 'br', 'strong', 'b', 'em', 'i', 'u', 'code', 'pre', 
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'ul', 'ol', 'li', 'blockquote',
        'table', 'thead', 'tbody', 'tr', 'th', 'td'
    }
    
    # Dangerous patterns to remove
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',
        r'<iframe[^>]*>.*?</iframe>',
        r'<object[^>]*>.*?</object>',
        r'<embed[^>]*>.*?</embed>',
        r'<form[^>]*>.*?</form>',
        r'<input[^>]*>',
        r'<button[^>]*>.*?</button>',
        r'javascript:',
        r'vbscript:',
        r'data:text/html',
        r'on\w+\s*=',  # Event handlers like onclick, onload, etc.
    ]
    
    @classmethod
    def sanitize_html_content(cls, content: str) -> str:
        """
        Sanitize HTML content for safe rendering
        
        Args:
            content: HTML content to sanitize
            
        Returns:
            Sanitized HTML content
        """
        if not content:
            return ""
        
        try:
            # Remove dangerous patterns
            sanitized = content
            for pattern in cls.DANGEROUS_PATTERNS:
                sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
            
            # Remove any tags not in allowed list
            # This is a simple approach - for production, consider using a proper HTML sanitizer library
            tag_pattern = r'<(/?)(\w+)[^>]*>'
            
            def replace_tag(match):
                closing = match.group(1)
                tag_name = match.group(2).lower()
                if tag_name in cls.ALLOWED_TAGS:
                    return match.group(0)
                else:
                    return ''
            
            sanitized = re.sub(tag_pattern, replace_tag, sanitized)
            
            # Escape any remaining < > characters that aren't part of allowed tags
            # This helps prevent any missed XSS attempts
            sanitized = re.sub(r'<(?!/?(p|br|strong|b|em|i|u|code|pre|h[1-6]|ul|ol|li|blockquote|table|thead|tbody|tr|th|td)(?:\s[^>]*)?>)', '&lt;', sanitized)
            
            logger.debug(f"Sanitized HTML content: {len(content)} -> {len(sanitized)} chars")
            return sanitized
            
        except Exception as e:
            logger.error(f"Error sanitizing HTML content: {e}")
            # If sanitization fails, escape everything for safety
            return html.escape(content)
    
    @classmethod
    def sanitize_text_content(cls, content: str) -> str:
        """
        Sanitize plain text content
        
        Args:
            content: Text content to sanitize
            
        Returns:
            Sanitized text content
        """
        if not content:
            return ""
        
        # For plain text, just escape HTML entities
        return html.escape(content)
    
    @classmethod
    def sanitize_markdown_content(cls, content: str) -> str:
        """
        Sanitize markdown content before parsing
        
        Args:
            content: Markdown content to sanitize
            
        Returns:
            Sanitized markdown content
        """
        if not content:
            return ""
        
        try:
            # Remove dangerous markdown patterns
            sanitized = content
            
            # Remove HTML script tags and dangerous elements
            for pattern in cls.DANGEROUS_PATTERNS:
                sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
            
            # Remove markdown link patterns that could be dangerous
            # Remove links with javascript: or data: protocols
            sanitized = re.sub(r'\[([^\]]*)\]\((?:javascript:|vbscript:|data:text/html)[^)]*\)', r'\1', sanitized, flags=re.IGNORECASE)
            
            # Remove image sources with dangerous protocols
            sanitized = re.sub(r'!\[([^\]]*)\]\((?:javascript:|vbscript:|data:text/html)[^)]*\)', r'[Image: \1]', sanitized, flags=re.IGNORECASE)
            
            logger.debug(f"Sanitized markdown content: {len(content)} -> {len(sanitized)} chars")
            return sanitized
            
        except Exception as e:
            logger.error(f"Error sanitizing markdown content: {e}")
            # If sanitization fails, escape everything for safety
            return html.escape(content)

def sanitize_llm_response(response: str, content_type: str = "markdown") -> str:
    """
    Sanitize LLM response based on content type
    
    Args:
        response: LLM response to sanitize
        content_type: Type of content ('markdown', 'html', 'text')
        
    Returns:
        Sanitized response
    """
    if not response:
        return ""
    
    if content_type == "markdown":
        return ContentSanitizer.sanitize_markdown_content(response)
    elif content_type == "html":
        return ContentSanitizer.sanitize_html_content(response)
    else:
        return ContentSanitizer.sanitize_text_content(response)