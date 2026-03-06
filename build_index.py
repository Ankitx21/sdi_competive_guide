import argparse
import glob
import os
import re
import sys
import time
import warnings
from getpass import getpass

import numpy as np
from pypdf import PdfReader

warnings.filterwarnings("ignore", category=FutureWarning)
import google.generativeai as genai


EMBED_MODEL_CANDIDATES = [
    "models/text-embedding-004",
    "models/gemini-embedding-001",
    "models/embedding-001",
]


def read_pdf_text(path: str):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        txt = " ".join(txt.split())
        if txt:
            pages.append((i, txt))
    return pages


def chunk_text(pages, source_name, chunk_size=1200, overlap=200):
    chunks = []
    for page_num, text in pages:
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunks.append({"source": source_name, "page": page_num, "text": text[start:end]})
            if end == len(text):
                break
            start = max(0, end - overlap)
    return chunks


def _retry_wait_seconds(err_text: str, default_wait=30):
    match = re.search(r"retry in ([0-9.]+)s", err_text, flags=re.IGNORECASE)
    if match:
        try:
            return max(default_wait, int(float(match.group(1))) + 1)
        except Exception:
            return default_wait
    return default_wait


def embed_with_retry(model_name, content, task_type=None, max_retries=6):
    current_task = task_type
    for attempt in range(max_retries):
        try:
            kwargs = {"model": model_name, "content": content}
            if current_task:
                kwargs["task_type"] = current_task
            return genai.embed_content(**kwargs)
        except Exception as err:
            msg = str(err)
            if "task_type" in msg.lower() and current_task is not None:
                current_task = None
                continue
            if "429" in msg or "ResourceExhausted" in msg:
                wait_s = _retry_wait_seconds(msg, default_wait=30)
                print(f"Rate limit hit while embedding. Waiting {wait_s}s and retrying...")
                time.sleep(wait_s)
                continue
            if attempt == max_retries - 1:
                raise
            time.sleep(2)
    raise RuntimeError("Embedding failed after retries.")


def resolve_embed_model():
    for model_name in EMBED_MODEL_CANDIDATES:
        try:
            embed_with_retry(model_name, "ping", task_type="retrieval_query", max_retries=1)
            return model_name
        except Exception:
            pass

    try:
        for model in genai.list_models():
            methods = getattr(model, "supported_generation_methods", []) or []
            if "embedContent" in methods:
                return model.name
    except Exception:
        pass
    return None


def embed_texts(model_name, texts):
    vectors = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = embed_with_retry(model_name, batch, task_type="retrieval_document")
        for emb in result["embedding"]:
            vectors.append(np.array(emb, dtype=np.float32))
    mat = np.vstack(vectors)
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return mat


def main():
    parser = argparse.ArgumentParser(description="Build and save multi-PDF embedding index.")
    parser.add_argument("--pdfs", nargs="+", help="PDF paths. If omitted, uses *.pdf in current folder.")
    parser.add_argument("--out", default="doc_index.npz", help="Output index file path.")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Chunk size in characters.")
    parser.add_argument("--overlap", type=int, default=200, help="Chunk overlap in characters.")
    parser.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY", ""), help="Gemini API key.")
    args, _ = parser.parse_known_args()

    api_key = args.api_key or getpass("Enter GEMINI API key: ").strip()
    if not api_key:
        print("Error: API key missing.")
        sys.exit(1)
    genai.configure(api_key=api_key)

    pdf_paths = args.pdfs or sorted(glob.glob("*.pdf"))
    if not pdf_paths:
        print("Error: No PDF files found. Provide --pdfs or place PDFs in current folder.")
        sys.exit(1)
    missing = [p for p in pdf_paths if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"Error: PDF not found: {p}")
        sys.exit(1)

    embed_model = resolve_embed_model()
    if not embed_model:
        print("Error: No embedding model available for this API key/project.")
        sys.exit(1)
    print(f"Using embedding model: {embed_model}")

    chunks = []
    total_pages = 0
    for pdf in pdf_paths:
        pages = read_pdf_text(pdf)
        total_pages += len(pages)
        chunks.extend(
            chunk_text(
                pages,
                source_name=os.path.basename(pdf),
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )
        )

    if not chunks:
        print("Error: No extractable text found in provided PDFs.")
        sys.exit(1)

    print(f"Embedding {len(chunks)} chunks from {len(pdf_paths)} PDF(s)...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(embed_model, texts)

    np.savez_compressed(
        args.out,
        embeddings=embeddings.astype(np.float32),
        texts=np.array(texts, dtype=object),
        pages=np.array([c["page"] for c in chunks], dtype=np.int32),
        sources=np.array([c["source"] for c in chunks], dtype=object),
        embed_model=np.array([embed_model], dtype=object),
    )
    print(
        f"Saved index: {args.out} | pages={total_pages}, chunks={len(chunks)}, dim={embeddings.shape[1]}"
    )


if __name__ == "__main__":
    main()
