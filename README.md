# 멀티모달 RAG 챗봇 (Multimodal RAG Chatbot)

이 프로젝트는 PDF 파일에서 텍스트, 이미지, 표를 추출하고, 멀티모달 콘텐츠를 기반으로 질문에 답변하는 고급 RAG (Retrieval Augmented Generation) 챗봇입니다. FastAPI를 백엔드 프레임워크로 사용하며, 로컬 Ollama 인스턴스를 LLM으로 활용합니다.

## ✨ 주요 기능

### 📄 멀티모달 문서 처리
*   **PDF 파일 업로드** - 백그라운드 비동기 처리 지원
*   **텍스트 추출** - PyMuPDF + Pytesseract OCR (한국어/영어 지원)
*   **이미지 추출** - PDF에서 이미지 자동 추출 및 메타데이터 저장
*   **표 추출** - OpenCV를 이용한 표 구조 인식 및 데이터 파싱
*   **텍스트 청킹** - LangChain을 이용한 지능적 텍스트 분할

### 🔍 지능형 검색 시스템
*   **벡터 임베딩** - Sentence Transformers (`jhgan/ko-sroberta-multitask`)
*   **멀티모달 저장** - ChromaDB에 텍스트, 이미지, 표 분리 저장
*   **문서별 필터링** - 특정 문서나 여러 문서에서 선택적 검색
*   **하이브리드 검색** - 텍스트 벡터 유사도 + 이미지/표 메타데이터 검색

### 🤖 고급 답변 생성
*   **멀티모달 RAG** - 텍스트, 이미지, 표를 통합한 맥락적 답변
*   **Ollama LLM 연동** - 로컬 LLM 모델 선택 가능
*   **출처 표시** - 답변에 참조한 문서, 페이지, 이미지, 표 정보 포함
*   **마크다운 지원** - 풍부한 형식의 답변 렌더링

### 🎨 현대적 웹 인터페이스
*   **반응형 디자인** - 모바일/태블릿 최적화
*   **실시간 진행률** - 문서 처리 단계별 진행 상황 표시
*   **멀티모달 답변 표시** - 참조된 이미지와 표를 답변과 함께 표시
*   **문서 관리** - 업로드된 문서 목록, 삭제, 통계 기능

## 프로젝트 구조

```
ocr-llm-chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI 애플리케이션 정의
│   ├── api/                # API 엔드포인트 (upload, chat)
│   ├── services/           # OCR, 텍스트 처리, DB, LLM 서비스 로직
│   ├── models/             # Pydantic 모델 (ChatRequest, ChatResponse)
│   ├── static/             # CSS, JavaScript
│   └── templates/          # HTML 템플릿 (index.html)
├── uploads/                # 업로드된 PDF 임시 저장
├── vector_db_data/         # ChromaDB 데이터 영구 저장
├── requirements.txt        # Python 의존성 목록
└── README.md
```

## 새로운 기능 및 개선사항

### ✨ 주요 개선사항
- **환경 설정 관리**: 모든 설정값을 환경변수로 분리하여 운영환경별 설정 가능
- **강화된 보안**: 파일 업로드 검증, MIME 타입 체크, 입력값 검증 및 sanitization
- **고급 에러 핸들링**: 세분화된 예외 처리와 구조화된 로깅 시스템
- **성능 최적화**: 임베딩 모델 싱글톤 캐싱, 배치 처리, LRU 캐시 적용
- **포괄적 테스트**: 단위 테스트 및 API 테스트 포함

### 🔧 기술 스택 개선
- **보안**: python-magic을 이용한 MIME 타입 검증, CORS 및 Trusted Host 미들웨어
- **로깅**: 구조화된 로깅 시스템 (콘솔, 파일, 에러 분리)
- **테스트**: pytest 기반 테스트 스위트 (70% 커버리지 목표)
- **설정**: 환경변수 기반 설정 관리

## 설정 및 실행

### 1. 사전 요구 사항

*   **Python:** 3.8 이상 권장
*   **Tesseract OCR 엔진:**
    *   시스템에 Tesseract OCR이 설치되어 있어야 하며, **한국어 데이터 파일 (`kor.traineddata`)**이 포함되어야 합니다.
    *   **Ubuntu/Debian:**
        ```bash
        sudo apt-get update
        sudo apt-get install tesseract-ocr tesseract-ocr-kor
        ```
    *   **macOS (Homebrew 사용):**
        ```bash
        brew install tesseract tesseract-lang
        # tesseract-lang은 모든 언어팩을 설치합니다. 특정 언어만 필요시 tesseract --list-langs 확인 후 설치.
        # 한국어팩만 설치하려면, brew install tesseract -> 이후 tessdata_fast 저장소에서 kor.traineddata 수동 다운로드 및 TESSDATA_PREFIX 설정 필요할 수 있음.
        # 일반적으로 tesseract-lang 설치가 간편합니다.
        ```
    *   **Windows:**
        1.  Tesseract [공식 설치 프로그램](https://github.com/UB-Mannheim/tesseract/wiki) (예: `tesseract-ocr-w64-setup-v5.x.x.exe`)을 다운로드하여 설치합니다.
        2.  설치 시 "Additional language data"에서 "Korean"을 선택합니다.
        3.  Tesseract 설치 경로를 시스템 PATH 환경 변수에 추가합니다 (예: `C:\Program Files\Tesseract-OCR`).
        4.  (선택 사항) `app/services/ocr_service.py` 파일 상단에 `pytesseract.pytesseract.tesseract_cmd` 경로를 직접 지정할 수 있습니다.
*   **Ollama:**
    *   [Ollama 공식 웹사이트](https://ollama.ai/)의 지침에 따라 Ollama를 설치합니다.
    *   애플리케이션에서 사용할 LLM 모델을 다운로드합니다. 기본 모델은 `llama2`로 설정되어 있습니다 (`app/services/llm_service.py`의 `DEFAULT_MODEL` 변수).
        ```bash
        ollama pull llama2
        ```
        다른 모델(예: 한국어 특화 모델 `mistral:latest` 또는 `eeve-korean-10.8b:latest` 등)을 사용하려면 해당 모델을 pull 받고, `DEFAULT_MODEL` 값을 변경하거나 채팅 요청 시 `model_name` 파라미터로 지정합니다.

### 2. 프로젝트 설정

1.  **저장소 복제 (해당하는 경우):**
    ```bash
    git clone <repository-url>
    cd ocr-llm-chatbot
    ```

2.  **Python 가상 환경 생성 및 활성화:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```

3.  **환경 설정:**
    ```bash
    # .env.example을 복사하여 .env 파일 생성
    cp .env.example .env
    # 필요에 따라 .env 파일을 편집하여 설정 조정
    ```

4.  **의존성 설치:**
    *   `requirements.txt` 파일에 명시된 모든 라이브러리를 설치합니다.
    *   Sentence Transformer 모델 (예: `jhgan/ko-sroberta-multitask`)은 처음 실행 시 인터넷을 통해 자동으로 다운로드됩니다.
    ```bash
    pip install -r requirements.txt
    ```

### 3. 애플리케이션 실행

*   Uvicorn ASGI 서버를 사용하여 FastAPI 애플리케이션을 실행합니다.
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
    `--reload` 옵션은 코드 변경 시 서버를 자동으로 재시작합니다 (개발 시 유용).

*   실행 후 웹 브라우저에서 `http://localhost:8000`으로 접속합니다.

### 4. 테스트 실행

*   단위 테스트 및 API 테스트를 실행합니다:
    ```bash
    # 모든 테스트 실행
    pytest
    
    # 커버리지 포함 테스트 실행
    pytest --cov=app --cov-report=html
    
    # 특정 테스트 파일만 실행
    pytest tests/test_security.py
    ```

## 사용 방법

1.  **PDF 업로드:** 웹 페이지의 "PDF 파일 업로드" 섹션에서 PDF 파일을 선택하고 "업로드 및 처리" 버튼을 클릭합니다.
    *   파일이 서버로 업로드되고 백그라운드에서 처리 (OCR, 청킹, 임베딩, DB 저장)가 시작됩니다.
    *   업로드 성공 시 `document_id`가 반환되며, 이는 채팅 시 특정 문서 문맥을 사용하는 데 활용될 수 있습니다 (현재 UI에서는 자동 활용 미구현, API 레벨에서는 지원).
2.  **챗봇과 대화:** "챗봇과 대화하기" 섹션의 입력창에 PDF 내용과 관련된 질문을 입력하고 "전송" 버튼을 클릭하거나 Enter 키를 누릅니다.
    *   챗봇은 업로드된 모든 문서의 내용을 기반으로 답변하거나, (구현된 경우) 특정 `document_id`의 문맥 내에서 답변을 생성하려고 시도합니다.

## 주의사항 및 추가 정보

*   **백그라운드 처리:** PDF 처리는 시간이 소요될 수 있는 작업이므로 백그라운드에서 수행됩니다. 업로드 직후에는 해당 PDF의 내용이 아직 검색되지 않을 수 있습니다. 처리 완료까지 기다려야 합니다. (실시간 처리 상태 알림 기능은 현재 미구현)
*   **Ollama 서버:** LLM 기능을 사용하려면 Ollama 서버가 로컬에서 실행 중이어야 합니다.
*   **Sentence Transformer 모델:** 첫 임베딩 생성 시 `sentence-transformers` 라이브러리가 지정된 모델 (`jhgan/ko-sroberta-multitask`)을 Hugging Face Hub에서 다운로드합니다. 이 과정에는 인터넷 연결이 필요하며 다소 시간이 소요될 수 있습니다. 이후에는 캐시된 모델을 사용합니다.
*   **Tesseract 언어 데이터:** 한국어 OCR을 위해서는 `kor.traineddata` 파일이 Tesseract의 `tessdata` 디렉토리에 올바르게 설치되어 있어야 합니다.

## 향후 개선 가능성

*   실시간 PDF 처리 상태 알림 (WebSocket, SSE 등)
*   특정 `document_id`를 UI에서 선택하여 해당 문서에 대해서만 질문하는 기능
*   더 정교한 오류 처리 및 사용자 피드백
*   모델 설정 (Sentence Transformer, Ollama) 외부화 (예: 설정 파일, 환경 변수)
*   Docker를 사용한 패키징 및 배포 간소화 (다음 단계)
```
