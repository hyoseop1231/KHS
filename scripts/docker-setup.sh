#!/bin/bash

# KITECH Docker Setup Script
# This script helps set up the KITECH Korean Foundry RAG Chatbot using Docker

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Docker installation
check_docker() {
    print_status "Checking Docker installation..."
    
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker first."
        echo "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! command_exists docker-compose; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        echo "Visit: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are properly installed and running"
}

# Function to check system resources
check_system_resources() {
    print_status "Checking system resources..."
    
    # Check available memory (in GB)
    available_memory=$(free -g | awk '/^Mem:/{print $7}')
    if [ "$available_memory" -lt 2 ]; then
        print_warning "Available memory is less than 2GB. KITECH may run slowly."
    else
        print_success "Sufficient memory available: ${available_memory}GB"
    fi
    
    # Check available disk space (in GB)
    available_disk=$(df . | awk 'NR==2{print int($4/1024/1024)}')
    if [ "$available_disk" -lt 5 ]; then
        print_warning "Available disk space is less than 5GB. May not be sufficient for large models."
    else
        print_success "Sufficient disk space available: ${available_disk}GB"
    fi
}

# Function to check Ollama installation
check_ollama() {
    print_status "Checking Ollama installation..."
    
    if ! command_exists ollama; then
        print_warning "Ollama is not installed on the host system."
        print_status "You can either:"
        echo "  1. Install Ollama locally: https://ollama.ai/download"
        echo "  2. Use Ollama in Docker (will be configured automatically)"
        
        read -p "Do you want to continue without local Ollama? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_error "Please install Ollama first or choose to continue."
            exit 1
        fi
    else
        print_success "Ollama is installed"
        
        # Check if Ollama is running
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            print_success "Ollama is running and accessible"
            
            # List available models
            print_status "Available Ollama models:"
            ollama list 2>/dev/null || echo "  No models found"
        else
            print_warning "Ollama is installed but not running. Starting Ollama..."
            ollama serve &
            sleep 5
        fi
    fi
}

# Function to create environment file
create_env_file() {
    print_status "Creating environment configuration..."
    
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
            print_success "Created .env file from .env.example"
        else
            # Create a basic .env file
            cat > .env << EOF
# KITECH Docker Configuration

# Basic Settings
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Security (IMPORTANT: Change in production!)
SECRET_KEY=change-this-in-production-$(openssl rand -hex 32)
CORS_ORIGINS=http://localhost:8000

# File Processing
MAX_FILE_SIZE=200
ALLOWED_EXTENSIONS=.pdf

# OCR Settings
OCR_LANGUAGES=kor+eng
OCR_DPI=300
OCR_CORRECTION_ENABLED=true
OCR_MAX_WORKERS=4

# LLM Settings
OLLAMA_API_URL=http://host.docker.internal:11434/api/generate
OLLAMA_DEFAULT_MODEL=gemma2:9b
LLM_TEMPERATURE=0.7

# Performance
EMBEDDING_BATCH_SIZE=32
CACHE_TTL_SECONDS=3600

# Logging
LOG_LEVEL=INFO
EOF
            print_success "Created basic .env file"
        fi
    else
        print_status ".env file already exists, skipping creation"
    fi
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p uploads
    mkdir -p vector_db_data
    mkdir -p logs
    mkdir -p monitoring/prometheus
    mkdir -p monitoring/grafana/dashboards
    mkdir -p monitoring/grafana/datasources
    mkdir -p nginx
    
    print_success "Directories created"
}

# Function to pull/build images
build_images() {
    print_status "Building KITECH Docker images..."
    
    docker-compose build --no-cache
    
    print_success "Docker images built successfully"
}

# Function to start services
start_services() {
    local environment=$1
    
    print_status "Starting KITECH services in $environment mode..."
    
    if [ "$environment" = "development" ]; then
        docker-compose -f docker-compose.dev.yml up -d
    else
        docker-compose up -d
    fi
    
    print_success "Services started successfully"
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 10
    
    # Check health
    if curl -s http://localhost:8000/api/health >/dev/null 2>&1; then
        print_success "KITECH is running and healthy!"
        echo
        echo "🎉 Setup complete! You can now access KITECH at:"
        echo "   Web Interface: http://localhost:8000"
        echo "   Health Check:  http://localhost:8000/api/health"
        echo "   API Docs:      http://localhost:8000/docs"
    else
        print_warning "Services are starting but not yet ready. This may take a few more minutes."
        echo "   You can check the logs with: docker-compose logs -f"
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --dev, -d       Set up for development (with hot reload)"
    echo "  --prod, -p      Set up for production (default)"
    echo "  --monitoring    Include monitoring stack (Prometheus + Grafana)"
    echo "  --check-only    Only check requirements, don't start services"
    echo "  --help, -h      Show this help message"
    echo
    echo "Examples:"
    echo "  $0                 # Production setup"
    echo "  $0 --dev          # Development setup"
    echo "  $0 --monitoring   # Production with monitoring"
    echo "  $0 --check-only   # Check requirements only"
}

# Main function
main() {
    local environment="production"
    local monitoring=false
    local check_only=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dev|-d)
                environment="development"
                shift
                ;;
            --prod|-p)
                environment="production"
                shift
                ;;
            --monitoring)
                monitoring=true
                shift
                ;;
            --check-only)
                check_only=true
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    echo "🏭 KITECH Korean Foundry RAG Chatbot Docker Setup"
    echo "=================================================="
    echo
    
    # Run checks
    check_docker
    check_system_resources
    check_ollama
    
    if [ "$check_only" = true ]; then
        print_success "All requirements check passed!"
        exit 0
    fi
    
    # Setup
    create_env_file
    create_directories
    build_images
    
    # Start services based on environment
    if [ "$monitoring" = true ]; then
        export COMPOSE_PROFILES=monitoring
        print_status "Including monitoring stack..."
    fi
    
    start_services "$environment"
    
    echo
    echo "📚 Next steps:"
    echo "  1. Upload PDF documents via the web interface"
    echo "  2. Chat with your documents using the AI assistant"
    echo "  3. Monitor system health at /api/health"
    
    if [ "$monitoring" = true ]; then
        echo "  4. Check monitoring dashboard at http://localhost:3000 (Grafana)"
    fi
    
    echo
    echo "🛠️ Useful commands:"
    echo "  docker-compose logs -f           # View logs"
    echo "  docker-compose down             # Stop services"
    echo "  docker-compose restart kitech-app # Restart main app"
    
    print_success "Setup completed successfully! 🎉"
}

# Check if script is being run directly
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi