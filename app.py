import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.classification.classify_document import (
    get_cached_key_embedding_graph_payload,
    router as classify_document_router,
)

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


@app.get("/api/key-embedding-graph")
def key_embedding_graph(response: Response):
    response.headers["Cache-Control"] = "public, max-age=86400"
    try:
        return get_cached_key_embedding_graph_payload()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="key_embedding_graph.json을 찾을 수 없습니다.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health_check():
    return {"status": "ok", "service": SERVICE_NAME}
