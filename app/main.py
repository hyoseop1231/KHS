from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
from app.config import settings
from app.utils.logging_config import setup_logging

# Setup logging
setup_logging()

# Create FastAPI app instance
app = FastAPI(
    title="OCR LLM Chatbot",
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

# Placeholder for API router (will be added later)
from app.api import endpoints
app.include_router(endpoints.router, prefix="/api") # Added prefix for API routes

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Root endpoint to serve the main HTML page.
    """
    return templates.TemplateResponse("index.html", {"request": request, "title": "OCR LLM Chatbot"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
