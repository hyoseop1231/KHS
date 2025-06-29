from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
from app.config import settings
from app.utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI app instance
app = FastAPI(
    title="KITECH RAG Chatbot",
    description="PDF-based RAG chatbot for foundry technology",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# Security middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Add security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    # Add security headers to prevent XSS and other attacks
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'"
    )
    return response

# Trust only specific hosts in production
if not settings.DEBUG:
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=["localhost", "127.0.0.1", settings.HOST]
    )

# Mount static files directory (for CSS, JS)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Mount uploads directory for serving extracted images and tables
import os
if os.path.exists("uploads"):
    app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

# Pre-load embedding model to prevent delays on first query
@app.on_event("startup")
async def startup_event():
    """
    Application startup event to pre-load embedding model
    """
    try:
        logger.info("Pre-loading embedding model to prevent first-query delays...")
        from app.services.text_processing_service import model_manager
        model_manager.get_model()
        logger.info("Embedding model pre-loaded successfully")
    except Exception as e:
        logger.error(f"Failed to pre-load embedding model: {e}")

# Placeholder for API router (will be added later)
from app.api import endpoints
app.include_router(endpoints.router, prefix="/api") # Added prefix for API routes

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Root endpoint to serve the main HTML page.
    """
    return templates.TemplateResponse("index.html", {"request": request, "title": "KITECH RAG Chatbot"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
