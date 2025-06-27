from pydantic import BaseModel
from typing import Optional

class UploadResponse(BaseModel):
    task_id: str
    filename: Optional[str] = None
    document_id: Optional[str] = None
    message: Optional[str] = "File upload accepted, processing started."

# WebSocket 통신을 위한 모델 (선택 사항, 현재는 사용 안 함)
# class WebSocketMessage(BaseModel):
#     type: str # 'question', 'answer_chunk', 'error', 'status'
#     payload: dict | str

# class QuestionMessage(WebSocketMessage):
#     type: str = "question"
#     payload: str # The user's question

# class AnswerChunkMessage(WebSocketMessage):
#     type: str = "answer_chunk"
#     payload: str # A chunk of the RAG answer

# class ErrorMessage(WebSocketMessage):
#     type: str = "error"
#     payload: dict # {"code": ..., "message": ...}
