#!/usr/bin/env python3
"""Score a sentence-transformers bi-encoder against this repo's eval set.

Mirrors agent/cmd/evalrun's scoring exactly (see score() in
agent/cmd/evalrun/main.go): a query with ideal_doc == null is only a hit
if the model abstains (we approximate abstention with a similarity floor);
otherwise hit == top-1 prediction equals ideal_doc, acceptable == top-1 is
ideal_doc or any of acceptable_docs. Also reports top-3/top-5 recall,
which a pure top-1 bi-encoder score understates when it's used as the
"retrieve" half of retrieve-then-rerank.

Usage:
    python3 eval_embed.py --model all-MiniLM-L6-v2
    python3 eval_embed.py --model ./models/finetuned-minilm --label "fine-tuned"
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent


def load_corpus(corpus_dir: Path) -> dict[str, str]:
    docs = {}
    for p in corpus_dir.rglob("*.md"):
        rel = p.relative_to(corpus_dir).as_posix()
        docs[rel] = p.read_text(errors="ignore")
    return docs


def paths_equal(a: str, b: str) -> bool:
    return os.path.normpath(a.removeprefix("./")) == os.path.normpath(b.removeprefix("./"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model name or local path")
    ap.add_argument("--label", default=None)
    ap.add_argument("--corpus", default=str(ROOT / "corpus/deno-docs"))
    ap.add_argument("--queries", default=str(ROOT / "eval/queries.json"))
    ap.add_argument("--ground-truth", default=str(ROOT / "eval/ground_truth.json"))
    ap.add_argument("--excerpt-chars", type=int, default=2000, help="doc chars embedded (title+lead, matches what a short doc chunk would look like)")
    ap.add_argument("--query-prefix", default="", help="e.g. 'query: ' for e5-family models, which are trained to expect it")
    ap.add_argument("--doc-prefix", default="", help="e.g. 'passage: ' for e5-family models")
    args = ap.parse_args()

    label = args.label or args.model
    corpus_dir = Path(args.corpus)
    docs = load_corpus(corpus_dir)
    doc_paths = sorted(docs)
    print(f"loaded {len(doc_paths)} docs from {corpus_dir}")

    queries = json.loads(Path(args.queries).read_text())
    gts = {g["id"]: g for g in json.loads(Path(args.ground_truth).read_text())}

    model = SentenceTransformer(args.model, device="cuda" if _has_cuda() else "cpu")

    doc_texts = [args.doc_prefix + docs[p][: args.excerpt_chars] for p in doc_paths]
    doc_vecs = model.encode(doc_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    query_texts = [args.query_prefix + q["query"] for q in queries]
    query_vecs = model.encode(query_texts, batch_size=32, normalize_embeddings=True)

    sims = query_vecs @ doc_vecs.T  # (num_queries, num_docs) cosine similarity

    hits = acceptable = none_count = 0
    recall_at = {1: 0, 3: 0, 5: 0}
    rows = []
    for i, q in enumerate(queries):
        gt = gts.get(q["id"])
        order = np.argsort(-sims[i])
        ranked = [doc_paths[j] for j in order[:10]]
        top1 = ranked[0]

        if gt is None:
            continue
        ideal = gt.get("ideal_doc")
        acceptable_docs = gt.get("acceptable_docs") or []

        if ideal is None:
            none_count += 1
            # a bi-encoder always returns *something* — no clean way to
            # abstain without a calibrated threshold, so these count
            # against hit/acceptable the same way evalrun counts an agent
            # that guessed instead of abstaining.
            hit = acc = False
        else:
            hit = paths_equal(top1, ideal)
            acc = hit or any(paths_equal(top1, a) for a in acceptable_docs)
            for k in recall_at:
                if any(paths_equal(r, ideal) for r in ranked[:k]):
                    recall_at[k] += 1

        hits += hit
        acceptable += acc
        rows.append({"id": q["id"], "query": q["query"], "top1": top1, "hit": hit, "acceptable": acc, "top5": ranked[:5]})

    total = len(queries)
    scored = total - none_count
    print(f"\n=== {label} ===")
    print(f"{total} queries ({none_count} coverage:none, {scored} scored)")
    print(f"exact hit@1:      {hits}/{total} ({100*hits/total:.1f}%)")
    print(f"acceptable hit@1: {acceptable}/{total} ({100*acceptable/total:.1f}%)")
    for k in (1, 3, 5):
        print(f"recall@{k} (of {scored} with a known doc): {recall_at[k]}/{scored} ({100*recall_at[k]/scored:.1f}%)")

    out_path = ROOT / f"results/embed_{label.replace('/', '_')}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"wrote per-query results to {out_path}")


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
