import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from src.config import EMBED_MODEL, CHROMA_DIR, COLLECTION

_model = None


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def get_collection(reset: bool = False):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def index_chunks(chunks: list[dict], reset: bool = True, batch: int = 64) -> None:
    col = get_collection(reset=reset)
    model = get_embedder()
    for i in tqdm(range(0, len(chunks), batch), desc="Embedding"):
        b = chunks[i:i + batch]
        emb = model.encode([c["text"] for c in b], normalize_embeddings=True,
                           batch_size=batch).tolist()
        col.upsert(
            ids=[c["chunk_id"] for c in b],
            embeddings=emb,
            documents=[c["text"] for c in b],
            metadatas=[{"doc_id": c["doc_id"], "source": c["source"], "page": c["page"],
                        "chunk_index": c["chunk_index"], "n_tokens": c["n_tokens"]} for c in b],
        )