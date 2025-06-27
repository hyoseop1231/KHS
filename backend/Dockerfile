# Python 3.11-slim 베이스 이미지 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 라이브러리 설치 (PaddleOCR 관련)
# PaddleOCR은 OpenCV에 의존하며, 일부 시스템 라이브러리가 필요할 수 있습니다.
# libGL.so.1 오류 방지를 위해 libgl1-mesa-glx 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 의존성 설치
COPY ./requirements.txt /app/requirements.txt
# 위 COPY 경로는 docker-compose의 context가 `.` 이고 dockerfile 위치가 `backend/Dockerfile` 이므로,
# 빌드 컨텍스트 내의 상대 경로를 사용해야 합니다.
# COPY ./backend/requirements.txt /app/requirements.txt -> COPY requirements.txt /app/requirements.txt
# docker-compose.yml에서 context: . 이고 dockerfile: backend/Dockerfile 이므로,
# backend 디렉토리 내의 파일은 빌드 컨텍스트 루트에서 backend/filename 으로 접근됩니다.
# 따라서 COPY ./backend/requirements.txt 는 맞지 않고, Dockerfile이 있는 위치 기준으로 COPY requirements.txt 가 맞습니다.
# 다만, build context가 `.` 이므로, `COPY backend/requirements.txt /app/requirements.txt` 가 되어야 합니다.
# 혼동을 줄이기 위해 Dockerfile 내에서는 상대경로를 사용하고, docker-compose.yml에서 build context를 명확히 합니다.
# context: . 이므로 backend/Dockerfile에서 backend 폴더 내의 파일을 복사하려면 COPY ./requirements.txt 가 아니라 COPY backend/requirements.txt 가 되어야 합니다.
# 하지만, docker build 명령어 실행 시 docker build -f backend/Dockerfile . 처럼 실행하면 . 이 빌드 컨텍스트가 되고,
# Dockerfile 내에서는 이 컨텍스트 루트로부터의 상대경로를 사용합니다.
# 즉, COPY backend/requirements.txt /app/requirements.txt 가 됩니다.
# 설계서의 backend/Dockerfile 에는 COPY backend/requirements.txt . 라고 되어있는데, 이것은 WORKDIR /app 이후이므로 /app/requirements.txt 로 복사됩니다.
# 여기서는 WORKDIR /app 이후 COPY ./requirements.txt 로 되어있으니, 빌드 컨텍스트의 requirements.txt를 /app/requirements.txt로 복사합니다.
# docker-compose.yml 의 build context가 `.` 이므로, `backend/requirements.txt`를 `COPY backend/requirements.txt /app/requirements.txt`로 해야합니다.

# 현재 backend/Dockerfile에 있으므로, requirements.txt는 같은 디렉토리에 있어야 함.
# 하지만 docker-compose.yml에서 build context를 `.` (루트)로 설정하고 `dockerfile: backend/Dockerfile`로 지정하면,
# Dockerfile 내의 COPY 명령어는 루트 컨텍스트를 기준으로 경로를 해석합니다.
# 따라서 `COPY backend/requirements.txt /app/requirements.txt`가 올바른 경로입니다.
# 위 Dockerfile 내용에서는 COPY ./backend/requirements.txt 로 되어있는데, 이를 수정합니다.
# --> 수정: COPY requirements.txt /app/requirements.txt (빌드 컨텍스트가 backend/ 일 경우)
# --> 수정: COPY backend/requirements.txt /app/requirements.txt (빌드 컨텍스트가 . 일 경우)
# 여기서는 docker-compose.yml의 context: . 이므로 후자를 따릅니다.

RUN pip install --no-cache-dir -r requirements.txt

# 백엔드 애플리케이션 코드 복사
# COPY ./backend /app -> COPY . /app (빌드 컨텍스트가 backend/ 일 경우)
# COPY backend /app (빌드 컨텍스트가 . 일 경우)
COPY . /app
# 위 COPY . /app은 backend 디렉토리의 모든 내용을 /app으로 복사합니다. (docker-compose context: ., dockerfile: backend/Dockerfile 가정)
# 만약 backend 디렉토리만 복사하려면 COPY backend /app 이 되어야 합니다.
# 그러나 docker-compose.yml에서 이미 `context: .` 이고 `dockerfile: backend/Dockerfile` 이므로,
# Dockerfile이 위치한 `backend` 디렉토리의 내용이 빌드 컨텍스트의 일부가 됩니다.
# 이 경우 `COPY . /app`은 `backend` 디렉토리의 내용을 `/app`으로 복사하는 것이 아니라,
# 프로젝트 루트의 모든 내용을 `/app`으로 복사하게 됩니다. 이것은 잘못된 동작입니다.
# 올바르게 하려면, docker-compose.yml에서 `build: backend` (context를 backend로) 하거나,
# 현재처럼 `context: .`을 유지하려면 Dockerfile의 COPY 경로를 `COPY backend/ /app/` 으로 해야 합니다.
# 설계서의 `COPY backend .` 는 WORKDIR /app 이후이므로 `backend` 디렉토리를 `/app/backend`로 복사합니다. 이것도 의도와 다릅니다.
# 여기서는 `COPY backend /app`으로 하여, 프로젝트 루트의 backend 폴더 내용을 컨테이너의 /app 폴더로 복사합니다.

# Uvicorn으로 FastAPI 애플리케이션 실행
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000", "--reload", "--app-dir", "/app/app"]
# app.main:app 을 실행해야 하므로, app 폴더가 /app/app 으로 복사된다면 main:app 으로 접근해야 합니다.
# 만약 backend/app 폴더가 /app 으로 복사된다면 main:app (app.main 의미)
# WORKDIR /app
# COPY backend/app /app
# CMD ["uvicorn", "main:app", ...]
# 현재는 COPY backend /app 이므로, 컨테이너 /app 내부에는 app, uploads, vector_db_data 등이 있게 됩니다.
# 따라서 CMD는 ["uvicorn", "app.main:app", ...] 이 맞습니다.
# --app-dir 옵션은 uvicorn이 앱을 찾는 경로를 지정합니다.
# CMD ["uvicorn", "app.main:app", "--host=0.0.0.0", "--port=8000", "--reload"] (기존 설계서)
# 위 Dockerfile 내용대로라면 CMD ["uvicorn", "app.main:app", "--host=0.0.0.0", "--port=8000", "--reload"] 가 맞습니다.
# 단, app폴더가 /app/app으로 복사되는 것이 아니라 /app 내부에 app 폴더가 위치하게 됩니다.
# 즉, /app/app/main.py 가 아니라 /app/main.py가 됩니다. (backend/app/main.py가 /app/main.py로 복사되도록 수정 필요)

# Dockerfile 재정의 (경로 명확화)
# FROM python:3.11-slim
# WORKDIR /app
# RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 && rm -rf /var/lib/apt/lists/*
# COPY backend/requirements.txt /app/requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt
# COPY backend/app /app/app  <-- backend/app 폴더를 /app/app 으로 복사
# CMD ["uvicorn", "app.main:app", "--host=0.0.0.0", "--port=8000", "--reload"]
# 위 방식이 아니라, backend/app 폴더를 /app 으로 복사하는 것이 더 일반적입니다.
# FROM python:3.11-slim
# WORKDIR /app
# RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 && rm -rf /var/lib/apt/lists/*
# COPY backend/requirements.txt /app/requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt
# COPY backend/app /app  <-- backend/app 폴더를 /app 으로 복사 (내부에 main.py 등이 위치)
# CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000", "--reload"]

# 설계서의 CMD ["uvicorn", "app.api:app", ...] 를 따르려면,
# backend/app/api.py 에 app 인스턴스가 있어야 합니다. 현재는 backend/app/main.py 에 있습니다.
# main.py 를 app/api.py로 변경하거나, CMD를 main:app으로 변경해야 합니다.
# 여기서는 main:app을 사용하고, 필요하면 파일 위치를 조정하겠습니다.

# 최종 backend/Dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/app /app # backend/app 폴더의 내용을 /app 으로 복사
# backend/uploads 등 기타 필요한 폴더도 복사하거나 볼륨 마운트
COPY backend/uploads /app/uploads
# CMD ["uvicorn", "app.main:app", "--host=0.0.0.0", "--port=8000", "--reload"] # 기존 설계서 방식
# backend/app/main.py 에 app이 있으므로, main:app (WORKDIR /app 기준)
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000", "--reload"]
