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
# Verification
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
NLI_ENTAIL_THRESHOLD = 0.90   # at or above this, skip the LLM judge
NLI_WINDOW_SENTENCES = 3      # evidence is split into windows of 3 sentences
NLI_WINDOW_STRIDE = 2
NLI_MAX_WINDOWS = 4           # NLI only checks the 4 evidence windows most similar to the claim
NLI_MAX_LENGTH = 256          # tokens per (evidence window, claim) pair
JUDGE_CONFIDENCE = 0.80       # confidence assigned when the LLM judge says SUPPORTED
NLI_ENABLED = os.getenv("NLI_ENABLED", "1") == "1"   # set NLI_ENABLED=0 to send every claim to the LLM judge
# Decision loop
MAX_LOOPS = 2                # re-retrieve/rewrite attempts after the first draft
MAX_CONTEXT_PASSAGES = 8     # cap on passages kept across loops
MIN_KEPT_FRACTION = 0.5      # if fewer claims than this survive verification, abstain
# LLM runtime
LLM_TIMEOUT_S = 60
LLM_CACHE_PATH = ROOT / "data" / "cache" / "llm_cache.jsonl"