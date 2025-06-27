# Stage 1: Build the React application
FROM node:20-alpine AS builder

WORKDIR /app

# package.json과 package-lock.json (또는 yarn.lock)을 먼저 복사하여 의존성 캐싱 활용
# frontend 디렉토리의 파일들을 복사해야 합니다.
# COPY ./frontend/package.json ./frontend/package-lock.json* ./
# docker-compose.yml 에서 context: . 이고 dockerfile: frontend/Dockerfile 이므로,
# frontend 디렉토리 내의 파일은 빌드 컨텍스트 루트에서 frontend/filename 으로 접근됩니다.
COPY frontend/package.json frontend/package-lock.json* ./
# 만약 yarn을 사용한다면 아래 줄의 주석을 해제합니다.
# COPY frontend/yarn.lock ./

# npm install 또는 yarn install
RUN npm install
# RUN yarn install

# 나머지 프론트엔드 소스 코드 복사
COPY frontend/. .

# 애플리케이션 빌드 (vite 사용)
RUN npm run build
# RUN yarn build

# Stage 2: Serve the static files using Nginx
FROM nginx:alpine

# Nginx 기본 설정 파일 삭제 (필요시)
# RUN rm /etc/nginx/conf.d/default.conf

# 커스텀 Nginx 설정 파일 복사 (필요하다면, 지금은 기본 설정 사용)
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# 빌드된 정적 파일들을 Nginx 웹 루트로 복사
COPY --from=builder /app/dist /usr/share/nginx/html

# Nginx가 80번 포트에서 수신 대기하도록 노출
EXPOSE 80

# Nginx 실행
CMD ["nginx", "-g", "daemon off;"]
