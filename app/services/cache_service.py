"""
Response caching service for improving chatbot performance
"""
import hashlib
import json
import time
from typing import Optional, Dict, Any
from app.config import settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

class ResponseCache:
    """Simple in-memory cache for chatbot responses"""
    
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.enabled = settings.ENABLE_RESPONSE_CACHE
        self.ttl = settings.CACHE_TTL_SECONDS
        self.max_size = settings.MAX_CACHE_SIZE
        
        logger.info(f"Response cache initialized: enabled={self.enabled}, ttl={self.ttl}s, max_size={self.max_size}")
    
    def _generate_cache_key(self, query: str, document_ids: list = None, model_name: str = None) -> str:
        """Generate a unique cache key for the query"""
        # Normalize inputs
        normalized_query = query.lower().strip()
        sorted_doc_ids = sorted(document_ids or [])
        
        # Create hash of query + context
        key_data = {
            "query": normalized_query,
            "document_ids": sorted_doc_ids,
            "model": model_name or settings.OLLAMA_DEFAULT_MODEL
        }
        
        key_string = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        cache_key = hashlib.md5(key_string.encode('utf-8')).hexdigest()
        
        return cache_key
    
    def _cleanup_expired(self):
        """Remove expired cache entries"""
        if not self.enabled:
            return
            
        current_time = time.time()
        expired_keys = []
        
        for key, access_time in self.access_times.items():
            if current_time - access_time > self.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _cleanup_lru(self):
        """Remove least recently used entries if cache is full"""
        if not self.enabled or len(self.cache) <= self.max_size:
            return
        
        # Sort by access time and remove oldest entries
        sorted_entries = sorted(self.access_times.items(), key=lambda x: x[1])
        entries_to_remove = len(self.cache) - self.max_size + 1
        
        for key, _ in sorted_entries[:entries_to_remove]:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)
        
        logger.debug(f"Cleaned up {entries_to_remove} LRU cache entries")
    
    def get(self, query: str, document_ids: list = None, model_name: str = None) -> Optional[Dict[str, Any]]:
        """Get cached response if available"""
        if not self.enabled:
            return None
        
        self._cleanup_expired()
        
        cache_key = self._generate_cache_key(query, document_ids, model_name)
        
        if cache_key in self.cache:
            # Update access time
            self.access_times[cache_key] = time.time()
            cached_response = self.cache[cache_key]
            
            logger.info(f"Cache HIT for query: {query[:50]}...")
            return cached_response
        
        logger.debug(f"Cache MISS for query: {query[:50]}...")
        return None
    
    def set(self, query: str, response: Dict[str, Any], document_ids: list = None, model_name: str = None):
        """Store response in cache"""
        if not self.enabled:
            return
        
        cache_key = self._generate_cache_key(query, document_ids, model_name)
        current_time = time.time()
        
        # Store response with metadata
        cache_entry = {
            **response,
            "_cached_at": current_time,
            "_cache_key": cache_key
        }
        
        self.cache[cache_key] = cache_entry
        self.access_times[cache_key] = current_time
        
        # Cleanup if necessary
        self._cleanup_lru()
        
        logger.info(f"Cached response for query: {query[:50]}...")
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.access_times.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "enabled": self.enabled,
            "total_entries": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl,
            "oldest_entry_age": min([time.time() - t for t in self.access_times.values()]) if self.access_times else 0,
            "newest_entry_age": max([time.time() - t for t in self.access_times.values()]) if self.access_times else 0
        }

# Global cache instance
response_cache = ResponseCache()

def get_cached_response(query: str, document_ids: list = None, model_name: str = None) -> Optional[Dict[str, Any]]:
    """Get cached response - convenience function"""
    return response_cache.get(query, document_ids, model_name)

def cache_response(query: str, response: Dict[str, Any], document_ids: list = None, model_name: str = None):
    """Cache response - convenience function"""
    response_cache.set(query, response, document_ids, model_name)

def clear_cache():
    """Clear cache - convenience function"""
    response_cache.clear()

def get_cache_stats() -> Dict[str, Any]:
    """Get cache stats - convenience function"""
    return response_cache.get_stats()