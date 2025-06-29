# 🐳 KITECH Docker 배포 가이드

이 문서는 KITECH 한국 주조기술 RAG 챗봇을 Docker를 사용하여 배포하는 방법을 설명합니다.

## 📋 목차

- [시스템 요구사항](#시스템-요구사항)
- [빠른 시작](#빠른-시작)
- [개발 환경 설정](#개발-환경-설정)
- [프로덕션 배포](#프로덕션-배포)
- [모니터링 설정](#모니터링-설정)
- [환경 설정](#환경-설정)
- [문제 해결](#문제-해결)

## 🖥️ 시스템 요구사항

### 최소 요구사항
- **Docker**: 20.10.0+
- **Docker Compose**: 2.0.0+
- **메모리**: 4GB RAM
- **디스크**: 10GB 여유 공간
- **CPU**: 2 코어

### 권장 사양
- **메모리**: 8GB+ RAM
- **디스크**: 20GB+ 여유 공간
- **CPU**: 4+ 코어
- **OS**: Ubuntu 20.04+, macOS 11+, Windows 10+

## 🚀 빠른 시작

### 1️⃣ 자동 설정 스크립트 사용

```bash
# 저장소 클론
git clone <repository-url>
cd KITECH

# 자동 설정 스크립트 실행
./scripts/docker-setup.sh

# 개발 환경으로 설정
./scripts/docker-setup.sh --dev

# 모니터링 포함 프로덕션 환경
./scripts/docker-setup.sh --monitoring
```

### 2️⃣ 수동 설정

```bash
# 환경 설정 파일 생성
cp .env.example .env

# 환경 변수 편집 (중요!)
nano .env

# 컨테이너 빌드 및 실행
docker-compose up --build -d

# 서비스 상태 확인
docker-compose ps
```

### 3️⃣ 접속 확인

- **웹 인터페이스**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs
- **헬스체크**: http://localhost:8000/api/health

## 🛠️ 개발 환경 설정

개발 중에는 코드 변경 시 자동 리로드가 되는 개발용 설정을 사용하세요.

```bash
# 개발용 docker-compose 사용
docker-compose -f docker-compose.dev.yml up --build -d

# 또는 setup 스크립트 사용
./scripts/docker-setup.sh --dev

# 로그 실시간 확인
docker-compose -f docker-compose.dev.yml logs -f kitech-app-dev

# 개발 도구 컨테이너 실행 (선택사항)
docker-compose -f docker-compose.dev.yml --profile tools up -d
```

### 개발 환경 특징
- 🔄 **핫 리로드**: 코드 변경 시 자동 재시작
- 📊 **상세 로깅**: DEBUG 레벨 로깅
- 🔧 **Redis Commander**: Redis 브라우저 (http://localhost:8081)
- ⚡ **빠른 재시작**: 캐시된 의존성

## 🏭 프로덕션 배포

### 기본 프로덕션 설정

```bash
# 프로덕션용 환경 변수 설정
cp .env.example .env
# SECRET_KEY, CORS_ORIGINS 등 프로덕션 값으로 변경

# 프로덕션 배포
docker-compose up --build -d

# 상태 확인
docker-compose ps
docker-compose logs kitech-app
```

### Nginx 프록시 포함 배포

```bash
# Nginx 포함 배포
docker-compose --profile production up --build -d

# SSL 인증서 설정 (필요시)
# nginx/ssl/ 디렉토리에 cert.pem, key.pem 배치
```

### 스케일링

```bash
# 앱 인스턴스 스케일링
docker-compose up --scale kitech-app=3 -d

# 로드 밸런서 설정 확인
curl -H "Host: your-domain.com" http://localhost
```

## 📊 모니터링 설정

모니터링 스택(Prometheus + Grafana)을 포함한 배포:

```bash
# 모니터링 포함 배포
docker-compose --profile monitoring up --build -d

# 또는 setup 스크립트 사용
./scripts/docker-setup.sh --monitoring
```

### 모니터링 접속 정보
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **KITECH 메트릭**: http://localhost:8000/api/metrics

### 주요 메트릭
- 🎯 **성능**: 응답 시간, 처리량, 에러율
- 💾 **리소스**: CPU, 메모리, 디스크 사용량
- 🤖 **AI**: LLM 응답 시간, OCR 성공률
- 📊 **비즈니스**: 문서 처리 건수, 사용자 질문 수

## ⚙️ 환경 설정

### 주요 환경 변수

```bash
# .env 파일 예시

# 🔒 보안 설정 (필수!)
SECRET_KEY=your-super-secure-secret-key-here
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# 🚀 서버 설정
HOST=0.0.0.0
PORT=8000
DEBUG=false

# 📄 파일 처리
MAX_FILE_SIZE=200
ALLOWED_EXTENSIONS=.pdf,.docx

# 🤖 LLM 설정
OLLAMA_API_URL=http://host.docker.internal:11434/api/generate
OLLAMA_DEFAULT_MODEL=gemma2:9b
LLM_TEMPERATURE=0.7
LLM_NUM_PREDICT_MULTIMODAL=2048

# 🔍 OCR 설정
OCR_LANGUAGES=kor+eng
OCR_DPI=300
OCR_CORRECTION_ENABLED=true
OCR_MAX_WORKERS=4

# ⚡ 성능 최적화
EMBEDDING_BATCH_SIZE=32
CHUNK_SIZE=1000
CHUNK_OVERLAP=150

# 💾 캐시 설정
ENABLE_RESPONSE_CACHE=true
CACHE_TTL_SECONDS=3600
MAX_CACHE_SIZE=100

# 📝 로깅
LOG_LEVEL=INFO
```

### Ollama 설정

#### 호스트에서 Ollama 실행 (권장)
```bash
# Ollama 설치 및 실행
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve

# 한국어 모델 다운로드
ollama pull gemma2:9b
ollama pull qwen2:7b

# Docker에서 호스트 Ollama 접근
# .env 파일에서:
OLLAMA_API_URL=http://host.docker.internal:11434/api/generate
```

#### Docker에서 Ollama 실행
```bash
# docker-compose.yml에 Ollama 서비스 추가
services:
  ollama:
    image: ollama/ollama:latest
    container_name: kitech-ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped

# .env 파일에서:
OLLAMA_API_URL=http://ollama:11434/api/generate
```

## 🔧 유용한 명령어

### 컨테이너 관리
```bash
# 서비스 시작
docker-compose up -d

# 서비스 중지
docker-compose down

# 특정 서비스 재시작
docker-compose restart kitech-app

# 로그 확인
docker-compose logs -f kitech-app

# 컨테이너 접속
docker-compose exec kitech-app bash

# 리소스 사용량 확인
docker stats
```

### 데이터 관리
```bash
# 볼륨 목록 확인
docker volume ls

# 볼륨 백업
docker run --rm -v kitech_vector_data:/data -v $(pwd):/backup ubuntu tar czf /backup/vector_backup.tar.gz -C /data .

# 볼륨 복원
docker run --rm -v kitech_vector_data:/data -v $(pwd):/backup ubuntu tar xzf /backup/vector_backup.tar.gz -C /data

# 데이터 초기화 (주의!)
docker-compose down -v
```

### 성능 최적화
```bash
# 사용하지 않는 이미지 정리
docker image prune -a

# 시스템 전체 정리
docker system prune -a --volumes

# 빌드 캐시 정리
docker builder prune
```

## 🐛 문제 해결

### 일반적인 문제들

#### 1. 컨테이너가 시작되지 않음
```bash
# 로그 확인
docker-compose logs kitech-app

# 컨테이너 상태 확인
docker-compose ps

# 헬스체크 확인
docker-compose exec kitech-app curl http://localhost:8000/api/health
```

#### 2. Ollama 연결 실패
```bash
# 호스트에서 Ollama 상태 확인
curl http://localhost:11434/api/tags

# Docker에서 호스트 접근 테스트
docker-compose exec kitech-app curl http://host.docker.internal:11434/api/tags

# 방화벽 확인 (Linux)
sudo ufw status
```

#### 3. 메모리 부족
```bash
# 메모리 사용량 확인
docker stats

# 컨테이너 메모리 제한 조정 (docker-compose.yml)
deploy:
  resources:
    limits:
      memory: 6G
```

#### 4. 권한 문제
```bash
# 볼륨 권한 확인
docker-compose exec kitech-app ls -la /app/uploads

# 권한 수정
docker-compose exec --user root kitech-app chown -R appuser:appuser /app/uploads
```

### 성능 문제 해결

#### OCR 처리 속도 개선
```bash
# OCR 워커 수 증가
OCR_MAX_WORKERS=8

# 배치 크기 조정
OCR_BATCH_SIZE=8
```

#### 임베딩 생성 속도 개선
```bash
# 배치 크기 조정
EMBEDDING_BATCH_SIZE=64

# GPU 사용 (CUDA 컨테이너 필요)
runtime: nvidia
```

### 로그 분석
```bash
# 에러 로그만 필터링
docker-compose logs kitech-app 2>&1 | grep ERROR

# 특정 시간대 로그
docker-compose logs --since="2024-01-01T00:00:00" kitech-app

# 실시간 로그 모니터링
docker-compose logs -f --tail=100 kitech-app
```

## 🔄 업데이트 및 백업

### 애플리케이션 업데이트
```bash
# 코드 업데이트
git pull

# 이미지 재빌드
docker-compose build --no-cache

# 서비스 재시작
docker-compose up -d
```

### 데이터 백업
```bash
# 전체 데이터 백업 스크립트
#!/bin/bash
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# 벡터 DB 백업
docker run --rm -v kitech_vector_data:/data -v $(pwd)/$BACKUP_DIR:/backup ubuntu tar czf /backup/vector_data.tar.gz -C /data .

# 업로드 파일 백업
docker run --rm -v kitech_uploads:/data -v $(pwd)/$BACKUP_DIR:/backup ubuntu tar czf /backup/uploads.tar.gz -C /data .

# 로그 백업
docker run --rm -v kitech_logs:/data -v $(pwd)/$BACKUP_DIR:/backup ubuntu tar czf /backup/logs.tar.gz -C /data .

echo "Backup completed: $BACKUP_DIR"
```

## 🌐 프로덕션 배포 체크리스트

- [ ] 🔒 SECRET_KEY 변경
- [ ] 🌍 CORS_ORIGINS 설정
- [ ] 🔥 DEBUG=false 설정
- [ ] 📊 모니터링 설정
- [ ] 🛡️ 방화벽 설정
- [ ] 🔐 SSL 인증서 설정
- [ ] 💾 데이터 백업 스크립트 설정
- [ ] 📝 로그 로테이션 설정
- [ ] 🚨 알림 설정
- [ ] 🔄 자동 업데이트 스크립트

---

**Docker로 KITECH을 효율적으로 배포하고 관리하세요! 🐳✨**