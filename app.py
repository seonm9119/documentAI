import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.classification.classify_document import router as classify_document_router

SERVICE_NAME = "document-ai"
DATA_DIR = Path(os.environ.get("DOCUMENT_AI_DATA_DIR", "/app/data"))


app = FastAPI(title="documentAI", version="0.1.0")
DATA_DIR.mkdir(parents=True, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://192.168.0.11:3000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(classify_document_router, prefix="/api", tags=["classify-document"])
app.mount("/document-ai-data", StaticFiles(directory=str(DATA_DIR)), name="document-ai-data")


@app.get("/health")
def health_check():
    return {"status": "ok", "service": SERVICE_NAME}
