"""
Main entry point for the Document Copilot API.

Run locally:

uv run uvicorn app.main:app --reload
"""

from fastapi import FastAPI

from app.config import settings


app = FastAPI(
    title="Document Copilot",
    version="0.1.0",
    description="Internal AI chatbot for querying company documents securely.",
)


@app.get("/")
async def root():
    return {
        "message": "Welcome to Document Copilot",
        "environment": "development",
        "status": "running",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
    }
