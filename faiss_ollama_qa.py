# faiss_ollama_qa.py
# Query an existing FAISS index (index.faiss) + meta.jsonl and optionally generate an answer via Ollama.
# Works with your structure: <index_dir>\index.faiss and <index_dir>\meta.jsonl (per year folder).

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import faiss  # pip install faiss-cpu
except Exception as e:
    print("faiss is required. Install with: pip install faiss-cpu")
    raise

# Embedding (default: multilingual-e5-small). You can change via --embed_model.
try:
    from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
except Exception:
    SentenceTransformer = None

# Ollama (optional; you already installed `ollama`)
try:
    from ollama import chat
except Exception:
    chat = None


def load_meta_jsonl(meta_path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"Failed parsing JSONL at line {line_no}: {e}")
    return items


def embed_query(query: str, model_name: str) -> np.ndarray:
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is required. Install with: pip install sentence-transformers"
        )

    model = SentenceTransformer(model_name)

    # E5-style models prefer "query: ..." prefix, but it's harmless for many models.
    q = query
    if "e5" in model_name.lower() and not query.lower().startswith("query:"):
        q = "query: " + query

    vec = model.encode([q], normalize_embeddings=True, convert_to_numpy=True)
    return vec.astype("float32")  # shape (1, dim)


def search_faiss(index_path: str, qvec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    index = faiss.read_index(index_path)

    # Dimension check
    dim_index = index.d
    dim_query = qvec.shape[1]
    if dim_index != dim_query:
        raise RuntimeError(
            f"Embedding dim mismatch: index.d={dim_index} but query embedding dim={dim_query}.\n"
            f"Fix: run with the SAME embedding model used to build the index.\n"
            f"Try another --embed_model, or tell me what model you used when building the FAISS index."
        )

    distances, ids = index.search(qvec, top_k)
    return distances[0], ids[0]  # 1D arrays


def build_context(items: List[Dict[str, Any]], ids: np.ndarray, distances: np.ndarray, max_chars: int) -> Tuple[str, List[Dict[str, Any]]]:
    selected: List[Dict[str, Any]] = []
    blocks: List[str] = []

    for rank, (idx, dist) in enumerate(zip(ids.tolist(), distances.tolist()), start=1):
        if idx < 0 or idx >= len(items):
            continue
        it = items[idx]
        text = (it.get("text") or "").strip()
        file_path = it.get("file_path", "")
        chunk_id = it.get("chunk_id", "")
        court = it.get("court", "")
        year = it.get("year", "")

        selected.append(
            {
                "rank": rank,
                "score": float(dist),
                "court": court,
                "year": year,
                "file_path": file_path,
                "chunk_id": chunk_id,
                "text": text,
            }
        )

        # Keep context compact but meaningful
        snippet = text
        if len(snippet) > max_chars:
            snippet = snippet[: max_chars].rstrip() + " …"

        blocks.append(
            f"[{rank}] score={dist:.4f}\n"
            f"court={court} year={year}\n"
            f"file={file_path}\n"
            f"chunk_id={chunk_id}\n"
            f"text:\n{snippet}\n"
        )

    context = "\n---\n".join(blocks)
    return context, selected


def ollama_answer(model: str, query: str, context: str) -> str:
    if chat is None:
        raise RuntimeError("Ollama python package not available. Install with: pip install ollama")

    system = (
        "You are a legal research assistant. Answer in Greek.\n"
        "Use ONLY the provided excerpts as sources. If the excerpts are insufficient, say so.\n"
        "Cite sources using [1], [2], etc matching the excerpt numbers.\n"
        "Be professional and concise."
    )

    user = (
        f"Ερώτημα:\n{query}\n\n"
        f"Αποσπάσματα (sources):\n{context}\n\n"
        f"Δώσε τελική απάντηση στα Ελληνικά με παραπομπές [1],[2]..."
    )

    resp = chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    # `resp` can be dict-like or object-like depending on version
    if isinstance(resp, dict):
        return resp.get("message", {}).get("content", "")
    return getattr(resp.message, "content", "")


def save_report(out_path: str, query: str, index_dir: str, embed_model: str, llm_model: str, results: List[Dict[str, Any]], answer: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    lines: List[str] = []
    lines.append(f"# FAISS QA Report")
    lines.append("")
    lines.append(f"- Time: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Index dir: `{index_dir}`")
    lines.append(f"- Embedding model: `{embed_model}`")
    lines.append(f"- LLM model: `{llm_model}`" if llm_model else "- LLM: (disabled)")
    lines.append("")
    lines.append(f"## Query")
    lines.append(query)
    lines.append("")
    lines.append("## Answer")
    lines.append(answer.strip() if answer else "(no answer)")
    lines.append("")
    lines.append("## Top Matches")
    for r in results:
        lines.append(
            f"### [{r['rank']}] score={r['score']:.4f}\n"
            f"- court: {r.get('court','')}\n"
            f"- year: {r.get('year','')}\n"
            f"- file_path: `{r.get('file_path','')}`\n"
            f"- chunk_id: {r.get('chunk_id','')}\n"
        )
        lines.append("```")
        lines.append((r.get("text") or "").strip())
        lines.append("```")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index_dir", required=True, help="Folder containing index.faiss and meta.jsonl")
    p.add_argument("--query", required=True, help="Your question")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--max_chars", type=int, default=1800, help="Max chars per chunk included in context")
    p.add_argument("--embed_model", default="intfloat/multilingual-e5-small")
    p.add_argument("--no_llm", action="store_true", help="Only retrieve passages; do not call Ollama")
    p.add_argument("--llm_model", default="llama3.1:8b", help="Ollama model name")
    p.add_argument("--out", default="", help="Output md path (default: results/<stamp>_faiss_qa.md)")

    args = p.parse_args()

    index_path = os.path.join(args.index_dir, "index.faiss")
    meta_path = os.path.join(args.index_dir, "meta.jsonl")

    if not os.path.exists(index_path):
        print(f"[ERROR] Missing: {index_path}")
        sys.exit(1)
    if not os.path.exists(meta_path):
        print(f"[ERROR] Missing: {meta_path}")
        sys.exit(1)

    items = load_meta_jsonl(meta_path)
    if not items:
        print("[ERROR] meta.jsonl is empty")
        sys.exit(1)

    qvec = embed_query(args.query, args.embed_model)
    distances, ids = search_faiss(index_path, qvec, args.top_k)
    context, selected = build_context(items, ids, distances, args.max_chars)

    print("\n[INFO] Top matches:")
    for r in selected:
        print(f"  [{r['rank']}] score={r['score']:.4f} | {r.get('year','')} | {r.get('file_path','')} | chunk={r.get('chunk_id','')}")

    answer = ""
    if args.no_llm:
        answer = "(LLM disabled; see retrieved excerpts below.)"
    else:
        answer = ollama_answer(args.llm_model, args.query, context)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or os.path.join("results", f"{stamp}_faiss_qa.md")
    save_report(out_path, args.query, args.index_dir, args.embed_model, ("" if args.no_llm else args.llm_model), selected, answer)

    print(f"\n[INFO] Saved report to: {out_path}")


if __name__ == "__main__":
    main()
