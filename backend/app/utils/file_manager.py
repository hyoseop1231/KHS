"""File management utilities for document operations"""

import os
import glob
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.config import settings
from app.utils.logging_config import get_logger
from app.utils.exceptions import FileProcessingError

logger = get_logger(__name__)

class DocumentFileManager:
    """Manages uploaded document files and their lifecycle"""
    
    @staticmethod
    def get_uploaded_files() -> List[Dict[str, Any]]:
        """
        업로드 디렉토리의 모든 파일 목록을 반환합니다.
        
        Returns:
            List[Dict]: 파일 정보 리스트 (filename, size, modified_time 등)
        """
        try:
            upload_path = Path(settings.UPLOAD_DIR)
            if not upload_path.exists():
                logger.warning(f"Upload directory does not exist: {settings.UPLOAD_DIR}")
                return []
            
            files = []
            for file_path in upload_path.glob("*.pdf"):
                try:
                    stat = file_path.stat()
                    # 파일명에서 document_id 추출 (document_id_originalname.pdf 형식)
                    filename = file_path.name
                    if "_" in filename:
                        document_id = filename.split("_")[0]
                    else:
                        document_id = filename.replace(".pdf", "")
                    
                    files.append({
                        "filename": filename,
                        "document_id": document_id,
                        "file_path": str(file_path),
                        "size": stat.st_size,
                        "modified_time": stat.st_mtime,
                        "size_mb": round(stat.st_size / (1024 * 1024), 2)
                    })
                except Exception as e:
                    logger.error(f"Error reading file info for {file_path}: {e}")
                    continue
            
            # 수정 시간 기준 내림차순 정렬
            files.sort(key=lambda x: x["modified_time"], reverse=True)
            logger.info(f"Found {len(files)} uploaded files")
            return files
            
        except Exception as e:
            logger.error(f"Error listing uploaded files: {e}")
            raise FileProcessingError(f"Failed to list files: {e}", "LIST_FILES_ERROR")
    
    @staticmethod
    def delete_file_by_document_id(document_id: str) -> bool:
        """
        Document ID로 해당 파일을 삭제합니다.
        
        Args:
            document_id (str): 삭제할 문서의 ID
            
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            upload_path = Path(settings.UPLOAD_DIR)
            # document_id로 시작하는 파일 찾기
            pattern = f"{document_id}_*.pdf"
            matching_files = list(upload_path.glob(pattern))
            
            if not matching_files:
                logger.warning(f"No files found for document_id: {document_id}")
                return False
            
            deleted_count = 0
            for file_path in matching_files:
                try:
                    file_path.unlink()
                    logger.info(f"Deleted file: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {e}")
            
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting file for document_id {document_id}: {e}")
            raise FileProcessingError(f"Failed to delete file: {e}", "DELETE_FILE_ERROR")
    
    @staticmethod
    def delete_all_files() -> int:
        """
        업로드 디렉토리의 모든 PDF 파일을 삭제합니다.
        
        Returns:
            int: 삭제된 파일의 개수
        """
        try:
            upload_path = Path(settings.UPLOAD_DIR)
            if not upload_path.exists():
                logger.warning(f"Upload directory does not exist: {settings.UPLOAD_DIR}")
                return 0
            
            pdf_files = list(upload_path.glob("*.pdf"))
            deleted_count = 0
            
            for file_path in pdf_files:
                try:
                    file_path.unlink()
                    logger.debug(f"Deleted file: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {e}")
            
            logger.info(f"Deleted {deleted_count} PDF files from upload directory")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting all files: {e}")
            raise FileProcessingError(f"Failed to delete all files: {e}", "DELETE_ALL_FILES_ERROR")
    
    @staticmethod
    def get_file_info(document_id: str) -> Optional[Dict[str, Any]]:
        """
        특정 document_id에 해당하는 파일 정보를 반환합니다.
        
        Args:
            document_id (str): 조회할 문서의 ID
            
        Returns:
            Optional[Dict]: 파일 정보 또는 None
        """
        try:
            upload_path = Path(settings.UPLOAD_DIR)
            pattern = f"{document_id}_*.pdf"
            matching_files = list(upload_path.glob(pattern))
            
            if not matching_files:
                return None
            
            file_path = matching_files[0]  # 첫 번째 매칭 파일
            stat = file_path.stat()
            
            return {
                "filename": file_path.name,
                "document_id": document_id,
                "file_path": str(file_path),
                "size": stat.st_size,
                "modified_time": stat.st_mtime,
                "size_mb": round(stat.st_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting file info for document_id {document_id}: {e}")
            raise FileProcessingError(f"Failed to get file info: {e}", "GET_FILE_INFO_ERROR")
    
    @staticmethod
    def cleanup_orphaned_files(valid_document_ids: List[str]) -> int:
        """
        벡터 DB에 존재하지 않는 고아 파일들을 정리합니다.
        
        Args:
            valid_document_ids (List[str]): 유효한 document_id 목록
            
        Returns:
            int: 삭제된 고아 파일의 개수
        """
        try:
            upload_path = Path(settings.UPLOAD_DIR)
            if not upload_path.exists():
                return 0
            
            pdf_files = list(upload_path.glob("*.pdf"))
            orphaned_count = 0
            
            for file_path in pdf_files:
                # 파일명에서 document_id 추출
                filename = file_path.name
                if "_" in filename:
                    file_document_id = filename.split("_")[0]
                else:
                    file_document_id = filename.replace(".pdf", "")
                
                # 유효한 document_id 목록에 없으면 고아 파일
                if file_document_id not in valid_document_ids:
                    try:
                        file_path.unlink()
                        logger.info(f"Deleted orphaned file: {file_path}")
                        orphaned_count += 1
                    except Exception as e:
                        logger.error(f"Error deleting orphaned file {file_path}: {e}")
            
            logger.info(f"Cleaned up {orphaned_count} orphaned files")
            return orphaned_count
            
        except Exception as e:
            logger.error(f"Error during orphaned file cleanup: {e}")
            raise FileProcessingError(f"Failed to cleanup orphaned files: {e}", "CLEANUP_ERROR")
    
    @staticmethod
    def get_storage_stats() -> Dict[str, Any]:
        """
        업로드 디렉토리의 저장소 통계를 반환합니다.
        
        Returns:
            Dict: 저장소 통계 정보
        """
        try:
            upload_path = Path(settings.UPLOAD_DIR)
            if not upload_path.exists():
                return {
                    "total_files": 0,
                    "total_size_bytes": 0,
                    "total_size_mb": 0,
                    "directory_exists": False
                }
            
            pdf_files = list(upload_path.glob("*.pdf"))
            total_size = sum(f.stat().st_size for f in pdf_files if f.exists())
            
            return {
                "total_files": len(pdf_files),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "directory_exists": True,
                "directory_path": str(upload_path)
            }
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            raise FileProcessingError(f"Failed to get storage stats: {e}", "STORAGE_STATS_ERROR")