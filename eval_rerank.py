#!/usr/bin/env python3
"""Score the retrieve-then-rerank pipeline from the field guide's §04:
a bi-encoder narrows the corpus to a shortlist, a cross-encoder reorders
just that shortlist. Reuses eval_embed.py's bi-encoder retrieval step,
then reranks the top --shortlist candidates with a cross-encoder and
rescoring hit@1/acceptable@1 on the reranked order.

Usage:
    python3 eval_rerank.py --biencoder ./models/finetuned-minilm --shortlist 10
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

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
    ap.add_argument("--biencoder", required=True, help="bi-encoder model name or local path (the 'retrieve' stage)")
    ap.add_argument("--cross-encoder", default="cross-encoder/ms-marco-MiniLM-L6-v2", help="cross-encoder model (the 'rerank' stage)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--shortlist", type=int, default=10, help="how many bi-encoder candidates the cross-encoder reranks")
    ap.add_argument("--corpus", default=str(ROOT / "corpus/deno-docs"))
    ap.add_argument("--queries", default=str(ROOT / "eval/queries.json"))
    ap.add_argument("--ground-truth", default=str(ROOT / "eval/ground_truth.json"))
    ap.add_argument("--excerpt-chars", type=int, default=2000)
    args = ap.parse_args()

    label = args.label or f"rerank({Path(args.biencoder).name})"
    corpus_dir = Path(args.corpus)
    docs = load_corpus(corpus_dir)
    doc_paths = sorted(docs)
    print(f"loaded {len(doc_paths)} docs from {corpus_dir}")

    queries = json.loads(Path(args.queries).read_text())
    gts = {g["id"]: g for g in json.loads(Path(args.ground_truth).read_text())}

    device = "cuda" if _has_cuda() else "cpu"
    bi = SentenceTransformer(args.biencoder, device=device)
    ce = CrossEncoder(args.cross_encoder, device=device)

    doc_texts = [docs[p][: args.excerpt_chars] for p in doc_paths]
    doc_vecs = bi.encode(doc_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    query_texts = [q["query"] for q in queries]
    query_vecs = bi.encode(query_texts, batch_size=32, normalize_embeddings=True)
    sims = query_vecs @ doc_vecs.T

    hits_before = acc_before = 0
    hits_after = acc_after = 0
    none_count = 0
    rows = []
    for i, q in enumerate(queries):
        gt = gts.get(q["id"])
        if gt is None:
            continue
        ideal = gt.get("ideal_doc")
        acceptable_docs = gt.get("acceptable_docs") or []

        order = np.argsort(-sims[i])[: args.shortlist]
        shortlist = [doc_paths[j] for j in order]
        top1_before = shortlist[0]

        ce_pairs = [[q["query"], docs[p][: args.excerpt_chars]] for p in shortlist]
        ce_scores = ce.predict(ce_pairs)
        reranked = [p for _, p in sorted(zip(ce_scores, shortlist), key=lambda x: -x[0])]
        top1_after = reranked[0]

        if ideal is None:
            # matches eval_embed.py / evalrun's score(): no doc in the
            # corpus answers this query, so any confident top-1 guess
            # (which a bi-/cross-encoder always produces) counts as a
            # miss rather than being excluded from the denominator.
            none_count += 1
            hb = ab = ha = aa = False
        else:
            hb = paths_equal(top1_before, ideal)
            ab = hb or any(paths_equal(top1_before, a) for a in acceptable_docs)
            ha = paths_equal(top1_after, ideal)
            aa = ha or any(paths_equal(top1_after, a) for a in acceptable_docs)

        hits_before += hb
        acc_before += ab
        hits_after += ha
        acc_after += aa
        rows.append({
            "id": q["id"], "query": q["query"], "ideal": ideal,
            "top1_before_rerank": top1_before, "top1_after_rerank": top1_after,
            "hit_before": hb, "hit_after": ha,
        })

    total = len(queries)
    print(f"\n=== {label}: bi-encoder retrieve (top {args.shortlist}) -> cross-encoder rerank ===")
    print(f"{total} queries ({none_count} coverage:none, counted against hit/acceptable per evalrun's score())")
    print(f"before rerank — exact hit@1: {hits_before}/{total} ({100*hits_before/total:.1f}%)  acceptable@1: {acc_before}/{total} ({100*acc_before/total:.1f}%)")
    print(f"after  rerank — exact hit@1: {hits_after}/{total} ({100*hits_after/total:.1f}%)  acceptable@1: {acc_after}/{total} ({100*acc_after/total:.1f}%)")

    out_path = ROOT / f"results/rerank_{Path(args.biencoder).name.replace('/', '_')}.json"
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
