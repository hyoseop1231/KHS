# 🏭 KITECH RAG 챗봇 (KHS-2)

한국생산기술연구원(KITECH) 주조 기술 전문 RAG(Retrieval Augmented Generation) 챗봇 시스템

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 목차

- [🎯 프로젝트 개요](#-프로젝트-개요)
- [✨ 주요 기능](#-주요-기능)
- [🏗️ 시스템 아키텍처](#️-시스템-아키텍처)
- [⚡ 빠른 시작](#-빠른-시작)
- [🐳 Docker 설치](#-docker-설치)
- [📝 사용법](#-사용법)
- [⚙️ 설정](#️-설정)
- [🔧 개발](#-개발)
- [📚 API 문서](#-api-문서)
- [🚀 배포](#-배포)
- [🧪 테스트](#-테스트)
- [📊 모니터링](#-모니터링)
- [🛠️ 문제 해결](#️-문제-해결)

## 🎯 프로젝트 개요

KITECH RAG 챗봇은 주조 기술 분야의 전문 지식을 학습하여 사용자의 질문에 정확하고 상세한 답변을 제공하는 AI 시스템입니다.

### 🌟 핵심 특징

- **🔍 멀티모달 RAG**: 텍스트, 이미지, 표 데이터를 통합 분석
- **🇰🇷 한국어 최적화**: 한국어 전용 임베딩 모델 (jhgan/ko-sroberta-multitask)
- **📊 실시간 처리**: 스트리밍 응답으로 빠른 사용자 경험
- **🔒 보안 강화**: 파일 검증, 입력 검증, XSS 방지
- **📈 성능 최적화**: 병렬 처리, 배치 임베딩, 메모리 관리
- **🐳 Docker 지원**: 컨테이너화된 배포 환경

## ✨ 주요 기능

### 📄 문서 처리
- **PDF 업로드 및 분석**: 텍스트, 이미지, 표 자동 추출
- **고급 OCR**: Tesseract + OpenCV를 활용한 한국어 텍스트 인식
- **멀티모달 콘텐츠**: 이미지 설명 및 표 구조 분석
- **실시간 진행률**: 문서 처리 상태 실시간 모니터링

### 🤖 AI 대화
- **스트리밍 응답**: 실시간 답변 생성 및 표시
- **컨텍스트 인식**: 문서 내용 기반 정확한 답변
- **마크다운 렌더링**: 구조화된 답변 형식
- **참조 정보**: 답변 근거가 되는 문서 정보 제공

### 🎛️ 시스템 관리
- **실시간 대시보드**: 시스템 상태, 문서 통계
- **모델 관리**: Ollama 모델 선택 및 상태 확인
- **설정 최적화**: OCR 교정, LLM 교정 토글
- **자동 새로고침**: 상태 정보 자동 업데이트

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   External      │
│   (HTML/JS)     │    │   (FastAPI)     │    │   Services      │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • 파일 업로드    │◄──►│ • API 엔드포인트 │◄──►│ • Ollama LLM    │
│ • 실시간 채팅    │    │ • 멀티모달 처리  │    │ • ChromaDB      │
│ • 진행률 표시    │    │ • 스트리밍 응답  │    │ • Tesseract OCR │
│ • 상태 모니터링  │    │ • 보안 & 검증   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │      Core Services        │
                    ├───────────────────────────┤
                    │ • PDF 처리 (PyMuPDF)      │
                    │ • OCR 서비스 (Tesseract)   │
                    │ • 텍스트 처리 & 청킹       │
                    │ • 벡터 DB (ChromaDB)      │
                    │ • 임베딩 (SentenceTransf) │
                    │ • 실시간 처리 관리         │
                    └───────────────────────────┘
```

### 📂 프로젝트 구조

```
KHS-2/
├── app/
│   ├── api/
│   │   └── endpoints.py          # API 라우터 및 엔드포인트
│   ├── services/
│   │   ├── llm_service.py        # LLM 통신 및 프롬프트 생성
│   │   ├── multimodal_llm_service.py  # 멀티모달 LLM 처리
│   │   ├── streaming_service.py  # 스트리밍 응답 처리
│   │   ├── ocr_service.py        # OCR 및 문서 처리
│   │   ├── text_processing_service.py  # 텍스트 처리 및 임베딩
│   │   ├── vector_db_service.py  # 벡터 DB 관리
│   │   └── model_info_service.py # 모델 정보 캐시
│   ├── utils/
│   │   ├── logging_config.py     # 로깅 설정
│   │   ├── security.py           # 보안 유틸리티
│   │   ├── sanitizer.py          # 입력 검증
│   │   ├── monitoring.py         # 성능 모니터링
│   │   └── file_manager.py       # 파일 관리
│   ├── templates/
│   │   └── index.html            # 메인 웹 인터페이스
│   ├── static/
│   │   └── style.css             # CSS 스타일
│   ├── config.py                 # 애플리케이션 설정
│   └── main.py                   # FastAPI 애플리케이션
├── uploads/                      # 업로드된 파일 저장소
├── vector_db_data/               # ChromaDB 데이터
├── requirements.txt              # Python 의존성
├── Dockerfile                    # Docker 이미지 빌드
├── docker-compose.yml            # Docker Compose 설정
└── README.md                     # 프로젝트 문서
```

## ⚡ 빠른 시작

### 📋 시스템 요구사항

- **Python**: 3.11+
- **메모리**: 최소 8GB (권장 16GB+)
- **디스크**: 최소 10GB 여유 공간
- **Ollama**: 설치 및 실행 중이어야 함

### 🔧 로컬 설치

1. **저장소 클론**
```bash
git clone <repository-url>
cd KHS-2
```

2. **가상환경 생성 및 활성화**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

3. **의존성 설치**
```bash
pip install -r requirements.txt
```

4. **Ollama 설치 및 모델 다운로드**
```bash
# Ollama 설치 (https://ollama.com/download)
ollama pull gemma2:9b
# 또는 다른 선호 모델
```

5. **환경 변수 설정**
```bash
cp .env.example .env
# .env 파일을 편집하여 설정 조정
```

6. **애플리케이션 실행**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

7. **브라우저에서 접속**
```
http://localhost:8000
```

## 🐳 Docker 설치

### 기본 실행

```bash
# 이미지 빌드 및 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f kitech-app
```

### 프로덕션 배포

```bash
# 프로덕션 모드 (Nginx 포함)
docker-compose --profile production up -d

# 모니터링 포함
docker-compose --profile production --profile monitoring up -d
```

### 환경 변수 설정

`.env` 파일 생성:

```bash
# 보안 설정
SECRET_KEY=your-super-secret-key-here
CORS_ORIGINS=http://localhost:8000,https://yourdomain.com

# LLM 설정
OLLAMA_API_URL=http://host.docker.internal:11434/api/generate
OLLAMA_DEFAULT_MODEL=gemma2:9b

# 파일 처리 설정
MAX_FILE_SIZE=200
OCR_LANGUAGES=kor+eng
OCR_MAX_WORKERS=4

# 성능 설정
EMBEDDING_BATCH_SIZE=32
CHUNK_SIZE=1000
```

## 📝 사용법

### 1. 문서 업로드

1. **파일 선택**: PDF 파일을 드래그&드롭 또는 클릭하여 선택
2. **OCR 설정**: OCR 교정 및 LLM 교정 옵션 선택
3. **업로드 시작**: "업로드 및 처리" 버튼 클릭
4. **진행률 확인**: 실시간 처리 상태 모니터링

### 2. AI 채팅

1. **질문 입력**: 하단 채팅창에 질문 입력
2. **모델 선택**: 상단에서 사용할 AI 모델 선택
3. **답변 확인**: 스트리밍으로 실시간 답변 생성
4. **참조 정보**: 답변 하단의 출처 문서 확인

### 3. 시스템 모니터링

- **대시보드**: 실시간 시스템 상태 확인
- **문서 관리**: 업로드된 문서 목록 및 삭제
- **모델 상태**: Ollama 연결 상태 및 모델 정보
- **성능 지표**: 메모리 사용량, 처리 속도 등

## ⚙️ 설정

### 주요 설정 파일

#### `app/config.py`

```python
# 서버 설정
HOST = "0.0.0.0"
PORT = 8000
DEBUG = False

# 파일 처리
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = [".pdf"]

# LLM 설정
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_DEFAULT_MODEL = "gemma2:9b"
LLM_TEMPERATURE = 0.7

# OCR 설정
OCR_LANGUAGES = "kor+eng"
OCR_DPI = 300
OCR_CORRECTION_ENABLED = True

# 성능 최적화
EMBEDDING_BATCH_SIZE = 32
OCR_MAX_WORKERS = 8
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
```

### 환경 변수

| 변수명 | 기본값 | 설명 |
|--------|--------|------|
| `HOST` | 0.0.0.0 | 서버 호스트 |
| `PORT` | 8000 | 서버 포트 |
| `DEBUG` | False | 디버그 모드 |
| `SECRET_KEY` | - | JWT 암호화 키 |
| `OLLAMA_API_URL` | http://localhost:11434/api/generate | Ollama API URL |
| `OLLAMA_DEFAULT_MODEL` | gemma2:9b | 기본 LLM 모델 |
| `MAX_FILE_SIZE` | 100 | 최대 파일 크기 (MB) |
| `OCR_LANGUAGES` | kor+eng | OCR 언어 설정 |
| `EMBEDDING_BATCH_SIZE` | 32 | 임베딩 배치 크기 |

## 🔧 개발

### 개발 환경 설정

```bash
# 개발용 의존성 설치
pip install -r requirements.txt

# 개발 서버 실행 (핫 리로드)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 새로운 서비스 추가

1. `app/services/` 디렉토리에 새 서비스 파일 생성
2. `app/api/endpoints.py`에 API 엔드포인트 추가
3. 필요시 `app/config.py`에 설정 추가
4. 테스트 작성 (`tests/` 디렉토리)

## 📚 API 문서

개발 모드에서 자동 생성되는 API 문서:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 주요 엔드포인트

#### 문서 관리

```http
POST /api/upload_pdf/
GET /api/documents
DELETE /api/documents/{document_id}
GET /api/upload_status/{document_id}
```

#### AI 채팅

```http
POST /api/chat/stream
POST /api/chat
```

#### 시스템 정보

```http
GET /api/ollama/status
GET /api/ollama/models
GET /api/storage/stats
GET /api/health
```

## 🚀 배포

### 프로덕션 배포 체크리스트

- [ ] 환경 변수 설정 (`SECRET_KEY`, `CORS_ORIGINS` 등)
- [ ] HTTPS 설정 (Nginx + SSL 인증서)
- [ ] 로그 설정 및 로테이션
- [ ] 백업 전략 (DB, 업로드 파일)
- [ ] 모니터링 설정 (Prometheus + Grafana)
- [ ] 보안 검토 (방화벽, 접근 제어)

### Nginx 설정 예시

```nginx
server {
    listen 80;
    server_name yourdomain.com;
    
    client_max_body_size 200M;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket 지원
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## 🧪 테스트

### 단위 테스트

```bash
# 전체 테스트 실행
pytest

# 커버리지 포함
pytest --cov=app tests/

# 특정 테스트 실행
pytest tests/test_llm_service.py -v
```

## 📊 모니터링

### 기본 모니터링

시스템 대시보드에서 실시간 확인 가능:

- **시스템 상태**: CPU, 메모리, 디스크 사용률
- **문서 통계**: 업로드된 문서 수, 총 청크 수
- **모델 상태**: Ollama 연결 상태, 활성 모델
- **처리 성능**: 응답 시간, 처리량

### 고급 모니터링 (선택사항)

Docker Compose 모니터링 프로필 사용:

```bash
# Prometheus + Grafana 실행
docker-compose --profile monitoring up -d

# 접속
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin123)
```

## 🛠️ 문제 해결

### 일반적인 문제

#### 1. Ollama 연결 실패

```bash
# Ollama 상태 확인
ollama list

# Ollama 재시작
killall ollama
ollama serve

# 방화벽 확인
curl http://localhost:11434/api/tags
```

#### 2. OCR 오류

```bash
# Tesseract 설치 확인
tesseract --version

# 언어 팩 설치
sudo apt-get install tesseract-ocr-kor
```

#### 3. 메모리 부족

```python
# config.py에서 설정 조정
EMBEDDING_BATCH_SIZE = 16  # 기본값: 32
OCR_MAX_WORKERS = 4        # 기본값: 8
```

#### 4. 파일 업로드 실패

- 파일 크기 확인 (기본값: 100MB)
- 파일 권한 확인
- 디스크 공간 확인

### 디버깅

```bash
# 디버그 모드 실행
export DEBUG=true
uvicorn app.main:app --reload --log-level debug

# 상세 로그 확인
export LOG_LEVEL=DEBUG
```

### 성능 최적화

```python
# 대용량 파일 처리 시
OCR_BATCH_SIZE = 4          # 배치 크기 감소
OCR_MAX_WORKERS = 4         # 워커 수 감소
EMBEDDING_BATCH_SIZE = 16   # 임베딩 배치 크기 감소
```

## 📞 지원

- **이슈 리포트**: [GitHub Issues](링크)
- **문서**: [Wiki](링크)
- **이메일**: support@kitech.re.kr

## 📄 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE) 하에 배포됩니다.

---

**한국생산기술연구원(KITECH)** © 2024