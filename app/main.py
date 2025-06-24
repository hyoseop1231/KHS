from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn

# Create FastAPI app instance
app = FastAPI(title="OCR LLM Chatbot")

# Mount static files directory (for CSS, JS)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

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
