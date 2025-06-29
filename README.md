# 🏭 KITECH: 한국 주조기술 전문 멀티모달 RAG 챗봇

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Security](https://img.shields.io/badge/Security-A+-brightgreen.svg)

**KITECH**는 한국 주조기술 전문용어에 특화된 엔터프라이즈급 멀티모달 RAG (Retrieval Augmented Generation) 챗봇 시스템입니다. PDF 문서에서 텍스트, 이미지, 표를 지능적으로 추출하고, 첨단 AI 기술로 전문적인 답변을 제공합니다.

## ✨ 핵심 기능

### 📄 **고급 멀티모달 문서 처리**
- **🔍 지능형 PDF 분석**: PyMuPDF + Pytesseract OCR로 한국어/영어 동시 지원
- **🖼️ 이미지 추출**: 고해상도 이미지 추출 및 메타데이터 자동 태깅
- **📊 표 구조 인식**: OpenCV 기반 표 검출 및 구조화된 데이터 파싱
- **⚡ 비동기 처리**: 대용량 파일도 백그라운드에서 효율적 처리
- **🧠 OCR 교정**: LLM 기반 주조 전문용어 인식률 향상

### 🚀 **첨단 RAG 검색 시스템**
- **🔗 벡터 임베딩**: `jhgan/ko-sroberta-multitask` 한국어 특화 모델
- **🗄️ 멀티모달 저장**: ChromaDB 기반 텍스트/이미지/표 분리 저장
- **🎯 스마트 검색**: 의미 기반 검색 + 주조 전문용어 가중치 적용
- **📁 문서별 필터링**: 특정 문서 또는 다중 문서 선택적 검색
- **💾 지능형 캐시**: LRU 기반 응답 캐싱으로 성능 최적화

### 🤖 **전문가급 AI 답변 생성**
- **🔗 Ollama LLM 연동**: 로컬 LLM 모델 자유 선택
- **📋 멀티모달 RAG**: 텍스트, 이미지, 표를 통합한 맥락적 답변
- **📍 정확한 출처 표시**: 문서, 페이지, 이미지, 표 참조 정보 자동 태깅
- **🎨 마크다운 렌더링**: 구조화된 전문적 답변 형식
- **🌊 실시간 스트리밍**: Server-Sent Events로 즉시 응답 확인

### 🎨 **현대적 사용자 인터페이스**
- **📱 반응형 디자인**: 모바일/태블릿/데스크탑 최적화
- **📊 실시간 대시보드**: OCR/LLM 상태, 시스템 리소스 모니터링
- **📈 진행률 시각화**: 문서 처리 단계별 상세 진행 상황
- **🎛️ 직관적 제어**: OCR 교정, LLM 교정 토글 설정
- **🔍 멀티모달 표시**: 참조 이미지/표를 답변과 함께 시각화

## 🏗️ 시스템 아키텍처

```
KITECH/
├── 🚀 app/
│   ├── main.py                 # FastAPI 앱 + 보안 미들웨어
│   ├── config.py              # 환경변수 기반 설정 관리
│   ├── 🌐 api/
│   │   └── endpoints.py       # REST API + 스트리밍 엔드포인트
│   ├── 🔧 services/
│   │   ├── ocr_service.py     # 멀티모달 OCR 처리
│   │   ├── text_processing_service.py # 텍스트 청킹 & 임베딩
│   │   ├── vector_db_service.py       # ChromaDB 벡터 관리
│   │   ├── llm_service.py            # Ollama LLM 연동
│   │   ├── multimodal_llm_service.py # 멀티모달 RAG 로직
│   │   ├── cache_service.py          # 응답 캐시 관리
│   │   ├── streaming_service.py      # 실시간 스트리밍
│   │   ├── ocr_correction_service.py # OCR 교정 시스템
│   │   └── term_correction_service.py # 주조 전문용어 교정
│   ├── 🛡️ utils/
│   │   ├── security.py        # 파일 검증 & 보안
│   │   ├── sanitizer.py       # XSS 방지 & 콘텐츠 정화
│   │   ├── monitoring.py      # 성능 모니터링 & 헬스체크
│   │   ├── logging_config.py  # 구조화된 로깅
│   │   └── exceptions.py      # 커스텀 예외 처리
│   ├── 🎨 templates/
│   │   └── index.html         # 반응형 웹 인터페이스
│   ├── 📁 static/
│   │   └── style.css          # 현대적 UI 스타일
│   └── 📊 data/
│       └── foundry_terminology.json # 주조 전문용어 사전
├── 📋 tests/                   # 포괄적 테스트 스위트
├── 🗂️ uploads/                # 업로드 파일 임시 저장
├── 🗄️ vector_db_data/        # ChromaDB 영구 저장
└── 📝 logs/                   # 시스템 로그 파일
```

## 🔒 보안 & 성능 특징

### 🛡️ **엔터프라이즈급 보안**
- **XSS 방지**: 포괄적인 콘텐츠 Sanitization 시스템
- **파일 검증**: MIME 타입, 파일 시그니처, 크기 검증
- **보안 헤더**: CSP, XSS Protection, HSTS 적용
- **입력 검증**: 모든 사용자 입력에 대한 엄격한 검증
- **프로덕션 강화**: SECRET_KEY 강제, CORS 제한

### ⚡ **최적화된 성능**
- **메모리 효율성**: psutil 기반 동적 배치 크기 조정
- **비동기 처리**: FastAPI + 백그라운드 작업 처리
- **스마트 캐시**: LRU 캐시 + TTL 기반 응답 캐싱
- **배치 처리**: 대용량 문서 효율적 처리
- **리소스 모니터링**: 실시간 시스템 리소스 추적

### 📊 **모니터링 & 관리**
- **헬스체크**: `/api/health` 엔드포인트
- **성능 메트릭**: `/api/metrics` 실시간 통계
- **구조화된 로깅**: 레벨별 로그 분리 저장
- **에러 추적**: 상세한 스택 트레이스 기록

## 🚀 빠른 시작

### 1️⃣ **시스템 요구사항**

- **Python**: 3.8+ (권장: 3.11+)
- **메모리**: 최소 4GB RAM (권장: 8GB+)
- **디스크**: 최소 2GB 여유 공간
- **OS**: Windows 10+, macOS 10.15+, Ubuntu 18.04+

### 2️⃣ **Tesseract OCR 설치**

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-kor tesseract-ocr-eng
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Windows:**
1. [Tesseract 설치 프로그램](https://github.com/UB-Mannheim/tesseract/wiki) 다운로드
2. 설치 시 "Korean" 언어팩 선택
3. PATH 환경변수에 Tesseract 경로 추가

### 3️⃣ **Ollama LLM 설치**

```bash
# Ollama 설치 (공식 웹사이트 참조)
curl -fsSL https://ollama.ai/install.sh | sh

# 한국어 특화 모델 다운로드 (예시)
ollama pull gemma2:9b
ollama pull qwen2:7b
```

### 4️⃣ **프로젝트 설정**

```bash
# 저장소 클론
git clone <repository-url>
cd KITECH

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 또는 venv\\Scripts\\activate  # Windows

# 환경설정 파일 생성
cp .env.example .env
# .env 파일을 편집하여 필요한 설정 조정

# 의존성 설치
pip install -r requirements.txt
```

### 5️⃣ **애플리케이션 실행**

**개발 환경:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**프로덕션 환경:**
```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

**또는 간편 실행:**
```bash
python start_server.py
```

### 6️⃣ **웹 인터페이스 접속**

브라우저에서 `http://localhost:8000` 접속

## 🧪 테스트 실행

```bash
# 전체 테스트 실행
pytest

# 커버리지 포함 테스트
pytest --cov=app --cov-report=html

# 특정 테스트만 실행
pytest tests/test_security.py -v

# 성능 테스트
pytest tests/test_performance.py -v
```

## 📖 사용 방법

### 📁 **문서 업로드**
1. 웹 페이지에서 PDF 파일 선택
2. OCR/LLM 교정 옵션 설정
3. "업로드 및 처리" 클릭
4. 실시간 진행률 확인

### 💬 **AI 챗봇 대화**
1. 업로드된 문서 선택 (옵션)
2. 주조기술 관련 질문 입력
3. 실시간 스트리밍 답변 확인
4. 참조 이미지/표 함께 검토

### 📊 **시스템 모니터링**
- 대시보드에서 실시간 상태 확인
- `/api/health` - 시스템 헬스체크
- `/api/metrics` - 성능 지표 조회

## 🔧 고급 설정

### 📋 **주요 환경변수**

```bash
# 서버 설정
HOST=0.0.0.0
PORT=8000
DEBUG=False

# 보안 설정
SECRET_KEY=your-super-secure-secret-key
CORS_ORIGINS=https://yourdomain.com

# LLM 설정
OLLAMA_DEFAULT_MODEL=gemma2:9b
LLM_TEMPERATURE=0.7
LLM_NUM_PREDICT_MULTIMODAL=2048

# 성능 최적화
OCR_MAX_WORKERS=8
EMBEDDING_BATCH_SIZE=32
CACHE_TTL_SECONDS=3600

# 파일 처리
MAX_FILE_SIZE=200  # MB
ALLOWED_EXTENSIONS=.pdf,.docx
```

### 🔄 **Docker 배포 (권장)**

#### 🚀 **빠른 Docker 배포**
```bash
# 1. 자동 설정 스크립트 사용 (권장)
./scripts/docker-setup.sh

# 2. 개발 환경
./scripts/docker-setup.sh --dev

# 3. 모니터링 포함 프로덕션
./scripts/docker-setup.sh --monitoring

# 4. 수동 배포
docker-compose up --build -d
```

#### 📋 **주요 Docker 구성**
- **🐳 Multi-stage Dockerfile**: 최적화된 이미지 크기
- **📦 Docker Compose**: 개발/프로덕션 환경 분리
- **🔒 보안 설정**: 비root 사용자, 헬스체크 포함
- **📊 모니터링**: Prometheus + Grafana (선택사항)
- **🌐 Nginx 프록시**: 프로덕션 로드밸런싱

#### 🔧 **빠른 커맨드**
```bash
# 서비스 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f kitech-app

# 헬스체크
curl http://localhost:8000/api/health

# 데이터 백업
docker run --rm -v kitech_vector_data:/data ubuntu tar czf backup.tar.gz -C /data .
```

> 💡 **자세한 Docker 가이드**: [docs/DOCKER.md](docs/DOCKER.md) 참조

## 🎯 성능 최적화 팁

### 🚀 **메모리 최적화**
- 큰 PDF 파일은 배치 크기 조정
- OCR_MAX_WORKERS를 CPU 코어 수에 맞게 설정
- 임베딩 모델 캐싱 활용

### ⚡ **응답 속도 향상**
- 자주 묻는 질문은 캐시 활용
- 불필요한 이미지/표 제외
- 컨텍스트 압축 설정 조정

### 📊 **확장성 고려사항**
- Redis 캐시 도입 검토
- 분산 처리를 위한 Celery 적용
- 로드 밸런싱 구성

## 🐛 문제 해결

### ❓ **자주 묻는 질문**

**Q: OCR 한국어 인식률이 낮아요**
A: Tesseract 한국어 데이터 파일 확인 및 OCR_DPI 설정 조정

**Q: LLM 응답이 느려요**
A: 모델 크기 조정 또는 GPU 사용 검토

**Q: 메모리 부족 오류가 발생해요**
A: OCR_BATCH_SIZE 및 EMBEDDING_BATCH_SIZE 감소

### 🔍 **로그 확인**
```bash
# 실시간 로그 모니터링
tail -f logs/app.log

# 에러 로그만 확인
tail -f logs/error.log
```

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 🙏 감사의 말

- [FastAPI](https://fastapi.tiangolo.com/) - 현대적인 웹 프레임워크
- [ChromaDB](https://www.trychroma.com/) - 벡터 데이터베이스
- [Ollama](https://ollama.ai/) - 로컬 LLM 플랫폼
- [Tesseract](https://tesseract-ocr.github.io/) - OCR 엔진

---

**KITECH으로 한국 주조기술의 디지털 혁신을 경험하세요! 🏭✨**