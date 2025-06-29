#!/usr/bin/env python3
"""
KITECH RAG Chatbot 서버 시작 스크립트
기존 8000포트 프로세스 종료 후 새 서버 실행
"""

import os
import sys
import subprocess
import signal
import time
import psutil

def find_process_by_port(port):
    """지정된 포트를 사용하는 프로세스 찾기"""
    for proc in psutil.process_iter(['pid', 'name', 'connections']):
        try:
            for conn in proc.info['connections'] or []:
                if conn.laddr.port == port:
                    return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return None

def kill_process_by_port(port):
    """포트를 사용하는 프로세스 종료"""
    pid = find_process_by_port(port)
    if pid:
        try:
            print(f"🛑 포트 {port}을 사용 중인 프로세스 발견 (PID: {pid})")
            print("프로세스를 종료합니다...")
            
            # 프로세스 종료
            process = psutil.Process(pid)
            process.terminate()  # SIGTERM 먼저 시도
            
            # 3초 대기 후 강제 종료
            time.sleep(3)
            if process.is_running():
                process.kill()  # SIGKILL로 강제 종료
            
            time.sleep(1)
            print("✅ 프로세스가 성공적으로 종료되었습니다.")
            return True
        except psutil.NoSuchProcess:
            print("✅ 프로세스가 이미 종료되었습니다.")
            return True
        except psutil.AccessDenied:
            print("❌ 프로세스 종료 권한이 없습니다. sudo로 실행해보세요.")
            return False
        except Exception as e:
            print(f"❌ 프로세스 종료 중 오류 발생: {e}")
            return False
    else:
        print(f"✅ 포트 {port}이 사용 가능합니다.")
        return True

def check_virtual_env():
    """가상환경 활성화 확인"""
    if not os.environ.get('VIRTUAL_ENV'):
        print("⚠️  가상환경이 활성화되지 않았습니다.")
        print("다음 명령어로 가상환경을 활성화해주세요:")
        print("source venv/bin/activate")
        return False
    print(f"✅ 가상환경 활성화 확인: {os.environ.get('VIRTUAL_ENV')}")
    return True

def check_project_structure():
    """프로젝트 구조 확인"""
    if not os.path.exists("app/main.py"):
        print("❌ app/main.py 파일을 찾을 수 없습니다.")
        print("프로젝트 루트 디렉토리에서 실행해주세요.")
        return False
    print(f"📂 현재 디렉토리: {os.getcwd()}")
    return True

def start_server():
    """서버 시작"""
    print("🔥 KITECH RAG Chatbot 서버를 시작합니다...")
    print("📍 접속 주소: http://localhost:8000")
    print("⏹️  서버 중지: Ctrl+C")
    print("")
    
    try:
        # uvicorn 서버 실행
        subprocess.run([
            "uvicorn", "app.main:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 서버 실행 중 오류 발생: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 서버가 중지되었습니다.")
        sys.exit(0)

def main():
    print("🚀 KITECH RAG Chatbot 서버 시작 중...")
    
    # 1. 가상환경 확인
    if not check_virtual_env():
        sys.exit(1)
    
    # 2. 프로젝트 구조 확인
    if not check_project_structure():
        sys.exit(1)
    
    # 3. 기존 포트 8000 프로세스 종료
    print("🔍 포트 8000 사용 중인 프로세스 확인 중...")
    if not kill_process_by_port(8000):
        sys.exit(1)
    
    # 4. 서버 시작
    start_server()

if __name__ == "__main__":
    main()