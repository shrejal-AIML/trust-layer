import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.api.schemas import AskRequest, AskResponse, build_response
from src.graph.rag_graph import get_graph, run
from src.ingestion.embed_store import get_collection, get_embedder
from src.retrieval import bm25
from src.retrieval.reranker import get_reranker
from src.verification.verifier import nli_scores
from src.config import NLI_ENABLED

logger = logging.getLogger("trust_layer.api")
_pipeline_lock = threading.Lock()    # one question at a time: the models share the CPU


def warmup() -> None:
    """Load every model once at startup so the first request isn't slow."""
    get_embedder()
    get_collection()
    bm25.search("warmup", 1)
    get_reranker()
    if NLI_ENABLED:
        nli_scores("warmup", [{"chunk_id": "warmup", "text": "A short warmup passage."}])
    get_graph()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.getenv("WARMUP", "1") == "1":
        warmup()
    yield


app = FastAPI(title="Trust Layer", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    with _pipeline_lock:
        try:
            out = run(request.question, mode=request.mode)
        except Exception as exc:
            logger.exception("pipeline failed")
            raise HTTPException(status_code=502, detail=f"Pipeline failed: {type(exc).__name__}") from exc
    return build_response(out)