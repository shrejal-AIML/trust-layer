import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CHROMA_DIR = ROOT / "data" / "chroma"
LOG_DIR = ROOT / "logs"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
REGISTRY_PATH = PROCESSED_DIR / "doc_registry.json"
QUERY_LOG_PATH = LOG_DIR / "queries.jsonl"

# Embeddings / chunking
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CHUNK_TOKENS = 500
CHUNK_OVERLAP = 50
COLLECTION = "trust_layer_chunks"
TOP_K = 5
# Retrieval
POOL_K = 30          # hits fetched from EACH retriever before fusion
CANDIDATES_K = 20    # fused candidates passed to the reranker
RRF_K = 60           # standard RRF constant
RERANK_MODEL = "BAAI/bge-reranker-base"

# LLM
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "") or "none"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")