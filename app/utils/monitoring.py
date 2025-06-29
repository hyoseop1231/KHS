"""Application monitoring and health check utilities"""

import time
import psutil
from typing import Dict, Any
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

class PerformanceMonitor:
    """Monitor application performance metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.processing_times = []
    
    def record_request(self, processing_time: float, error: bool = False):
        """Record a request with its processing time"""
        self.request_count += 1
        if error:
            self.error_count += 1
        
        self.processing_times.append(processing_time)
        
        # Keep only last 1000 processing times for memory efficiency
        if len(self.processing_times) > 1000:
            self.processing_times = self.processing_times[-1000:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        uptime = time.time() - self.start_time
        
        # Calculate average processing time
        avg_processing_time = 0
        if self.processing_times:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        
        # Get system metrics
        system_stats = self.get_system_stats()
        
        return {
            "uptime_seconds": round(uptime, 2),
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": round(self.error_count / max(self.request_count, 1) * 100, 2),
            "avg_processing_time": round(avg_processing_time, 3),
            "recent_requests": len(self.processing_times),
            **system_stats
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system resource usage statistics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_usage_percent": round(cpu_percent, 1),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_usage_percent": round(memory.percent, 1),
                "disk_total_gb": round(disk.total / (1024**3), 2),
                "disk_used_gb": round(disk.used / (1024**3), 2),
                "disk_usage_percent": round(disk.percent, 1)
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {
                "cpu_usage_percent": 0,
                "memory_total_gb": 0,
                "memory_used_gb": 0,
                "memory_usage_percent": 0,
                "disk_total_gb": 0,
                "disk_used_gb": 0,
                "disk_usage_percent": 0
            }
    
    def check_health(self) -> Dict[str, Any]:
        """Perform health checks"""
        health_status = {
            "status": "healthy",
            "checks": {
                "database": self._check_database(),
                "memory": self._check_memory(),
                "disk": self._check_disk(),
                "error_rate": self._check_error_rate()
            }
        }
        
        # Overall status based on individual checks
        failed_checks = [check for check, status in health_status["checks"].items() if not status["healthy"]]
        if failed_checks:
            health_status["status"] = "unhealthy"
            health_status["failed_checks"] = failed_checks
        
        return health_status
    
    def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            # For ChromaDB, we'll check if we can import and create a client
            import chromadb
            from app.config import settings
            
            # Try to create a client - this will fail if ChromaDB is not accessible
            client = chromadb.PersistentClient(path=settings.CHROMA_DATA_PATH)
            collections = client.list_collections()
            
            return {
                "healthy": True,
                "message": f"Database accessible, {len(collections)} collections found"
            }
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Database check failed: {str(e)}"
            }
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                return {
                    "healthy": False,
                    "message": f"High memory usage: {memory.percent}%"
                }
            return {
                "healthy": True,
                "message": f"Memory usage: {memory.percent}%"
            }
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Memory check failed: {str(e)}"
            }
    
    def _check_disk(self) -> Dict[str, Any]:
        """Check disk usage"""
        try:
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                return {
                    "healthy": False,
                    "message": f"High disk usage: {disk.percent}%"
                }
            return {
                "healthy": True,
                "message": f"Disk usage: {disk.percent}%"
            }
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Disk check failed: {str(e)}"
            }
    
    def _check_error_rate(self) -> Dict[str, Any]:
        """Check error rate"""
        if self.request_count == 0:
            return {
                "healthy": True,
                "message": "No requests processed yet"
            }
        
        error_rate = (self.error_count / self.request_count) * 100
        if error_rate > 20:  # More than 20% error rate is concerning
            return {
                "healthy": False,
                "message": f"High error rate: {error_rate:.1f}%"
            }
        
        return {
            "healthy": True,
            "message": f"Error rate: {error_rate:.1f}%"
        }

# Global monitor instance
monitor = PerformanceMonitor()

def get_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    return monitor