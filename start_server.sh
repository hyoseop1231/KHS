#!/bin/bash

# KITECH RAG Chatbot 서버 시작 스크립트
# 기존 8000포트 프로세스 종료 후 새 서버 실행

echo "🚀 KITECH RAG Chatbot 서버 시작 중..."

# 가상환경 자동 활성화
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "🔧 가상환경을 활성화합니다..."
    source venv/bin/activate
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        echo "❌ 가상환경 활성화에 실패했습니다."
        echo "venv 디렉토리가 존재하는지 확인해주세요."
        exit 1
    fi
fi

echo "✅ 가상환경 활성화 확인: $VIRTUAL_ENV"

# 기존 8000포트 사용 프로세스 확인 및 종료
echo "🔍 포트 8000 사용 중인 프로세스 확인 중..."

# macOS/Linux에서 포트 8000을 사용하는 프로세스 찾기
PID=$(lsof -ti:8000)

if [ ! -z "$PID" ]; then
    echo "🛑 포트 8000을 사용 중인 프로세스 발견 (PID: $PID)"
    echo "프로세스를 종료합니다..."
    kill -9 $PID
    sleep 2
    
    # 종료 확인
    NEW_PID=$(lsof -ti:8000)
    if [ -z "$NEW_PID" ]; then
        echo "✅ 프로세스가 성공적으로 종료되었습니다."
    else
        echo "❌ 프로세스 종료에 실패했습니다. 수동으로 종료해주세요."
        exit 1
    fi
else
    echo "✅ 포트 8000이 사용 가능합니다."
fi

# 현재 디렉토리 확인
if [ ! -f "app/main.py" ]; then
    echo "❌ app/main.py 파일을 찾을 수 없습니다."
    echo "프로젝트 루트 디렉토리에서 실행해주세요."
    exit 1
fi

echo "📂 현재 디렉토리: $(pwd)"

# 서버 실행
echo "🔥 KITECH RAG Chatbot 서버를 시작합니다..."
echo "📍 접속 주소: http://localhost:8000"
echo "⏹️  서버 중지: Ctrl+C"
echo ""

# uvicorn 서버 실행 (개발 모드)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000