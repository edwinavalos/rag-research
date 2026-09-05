#!/usr/bin/env python3
"""Fine-tune a small bi-encoder on train/pairs.json (synthetic (query, doc)
pairs from agent/cmd/gentrainpairs), using in-batch negatives.

MultipleNegativesRankingLoss treats every other doc in the batch as a free
negative for a given query, so batch_size directly sets how many negatives
each example trains against — see internal/reference for the InfoNCE
write-up this mirrors.

Usage:
    python3 train_biencoder.py
    python3 train_biencoder.py --base bge-small-en-v1.5 --epochs 4
"""
import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="all-MiniLM-L6-v2")
    ap.add_argument("--pairs", default=str(ROOT / "train/pairs.json"))
    ap.add_argument("--corpus", default=str(ROOT / "corpus/deno-docs"))
    ap.add_argument("--out", default=str(ROOT / "models/finetuned-minilm"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32, help="also sets in-batch negative count (batch_size - 1 negatives per example)")
    ap.add_argument("--excerpt-chars", type=int, default=2000)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--query-prefix", default="", help="e.g. 'query: ' for e5-family models, which are trained to expect it")
    ap.add_argument("--doc-prefix", default="", help="e.g. 'passage: ' for e5-family models")
    args = ap.parse_args()

    corpus_dir = Path(args.corpus)
    pairs = json.loads(Path(args.pairs).read_text())
    print(f"loaded {len(pairs)} training pairs")

    doc_cache: dict[str, str] = {}

    def doc_text(rel_path: str) -> str:
        if rel_path not in doc_cache:
            doc_cache[rel_path] = args.doc_prefix + (corpus_dir / rel_path).read_text(errors="ignore")[: args.excerpt_chars]
        return doc_cache[rel_path]

    examples = [InputExample(texts=[args.query_prefix + p["query"], doc_text(p["doc"])]) for p in pairs]

    model = SentenceTransformer(args.base, device="cuda")
    train_loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_loader) * args.epochs * args.warmup_ratio)
    print(f"training {args.base} for {args.epochs} epochs, batch_size={args.batch_size} ({args.batch_size - 1} in-batch negatives/example), warmup_steps={warmup_steps}")

    model.fit(
        train_objectives=[(train_loader, loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        output_path=args.out,
    )
    print(f"saved fine-tuned model to {args.out}")


if __name__ == "__main__":
    main()
