import json
from tqdm import tqdm
from src.config import RAW_DIR, PROCESSED_DIR, CHUNKS_PATH
from src.ingestion.loader import load_pdf
from src.ingestion.cleaner import clean_pages
from src.ingestion.chunker import chunk_document, get_doc_id
from src.ingestion.embed_store import index_chunks


def run() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {RAW_DIR}")

    all_chunks = []
    for pdf in tqdm(pdfs, desc="Parsing PDFs"):
        pages = clean_pages(load_pdf(pdf))
        doc_id = get_doc_id(pdf.name)
        chunks = chunk_document(pages, doc_id, pdf.name)
        print(f"  {pdf.name} -> {doc_id}: {len(pages)} pages, {len(chunks)} chunks")
        all_chunks.extend(chunks)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_chunks)} chunks to {CHUNKS_PATH}")

    index_chunks(all_chunks, reset=True)
    print("Indexed in Chroma. Done.")


if __name__ == "__main__":
    run()