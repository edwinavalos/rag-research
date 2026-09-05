# rag-research

A controlled experiment repo for figuring out why an LLM agent misses the
right document when searching a docs corpus, and how far you can get
without any vectors at all.

The corpus is [denoland/docs](https://github.com/denoland/docs) (457
markdown files), public, real, and not something I authored, so there's no
bias in how it happens to be structured. Two subagents built the eval set
independently: one wrote 45 realistic user queries without reading the
corpus, the other found each query's actual ground-truth answer doc (or
marked it `coverage: none` if nothing answers it). Five exploration methods
were then run against that fixed eval set and scored against ground truth.

See ["I Think I Reinvented RAG With Grep"](https://n0tls.com/2026-09-03-i-think-i-reinvented-rag-with-grep.html)
for the full writeup and results table.

The follow-up, ["RAG Research Continued - Fine Tuning Three Bi-Encoders"](https://n0tls.com/2026-09-05-fine-tuning-a-tiny-retriever.html),
fine-tunes and evaluates bi-encoder retrievers against the same eval set.

## Layout

```
agent/          Go module, all the actual code
  cmd/evalrun/     batch harness: runs the eval set through an exploration
                   method, scores against ground truth, writes results/*.json
  cmd/buildindex/  builds the corpus catalog (out-of-band, checked in)
  internal/
    explorer/        graph method: search <-> judge loop (google/adk-go)
    exploreagent/     single-shot method: one thorough parallel-tool-call turn
    doctools/        the tools both methods call: glob_docs, grep_docs,
                     read_doc, rank_docs
    indexer/         corpus catalog load/save/format
    llmprovider/     picks the model.LLM adapter (openai or anthropic)
    anthropicmodel/  hand-rolled Anthropic adapter (adk-go ships none)
eval/           queries.json + ground_truth.json (the fixed eval set)
index/          corpus_index.json, the built catalog (457 one-line summaries)
corpus/         gitignored; clone denoland/docs here to run anything
results/        evalrun output, gitignored except .gitkeep
```

## Setup

```
git clone https://github.com/denoland/docs corpus/deno-docs
cd agent
go build ./...
```

Requires `OPENAI_API_KEY` (default provider) or `ANTHROPIC_API_KEY` /
`ANTHROPIC_AUTH_TOKEN` for `-provider anthropic`.

## Running an eval

From `agent/`:

```
go run ./cmd/evalrun -method single-shot -limit 5   # cheap smoke test
go run ./cmd/evalrun -method graph                  # full run, graph method
go run ./cmd/evalrun -method single-shot            # full run, single-shot method
```

Key flags (see `-h` for the full list): `-method` (`graph` or
`single-shot`), `-provider`, `-model`, `-limit`/`-id` to restrict which
queries run, `-index` to point at a catalog (single-shot only; omit or
point at a missing file to run without one).

## Rebuilding the catalog

```
go run ./cmd/buildindex
```

Summarizes every doc in the corpus into one concrete sentence, batched
through an LLM, and writes `index/corpus_index.json`. This is a separate,
deliberate step from `evalrun`: the catalog is built once (or whenever the
corpus changes) and checked in, not regenerated inside the eval loop.

## Adding eval cases

The eval set is two flat JSON arrays, matched by `id`:

**`eval/queries.json`**, one entry per query:

```json
{
  "id": "q046",
  "persona": "someone setting up a monorepo",
  "query": "how do I use deno workspaces with multiple packages",
  "category": "workspaces",
  "difficulty": "medium"
}
```

`persona` and `category`/`difficulty` aren't read by the scoring code. They're
just there to keep the eval set legible and let you group results afterward.
The one rule that matters for a clean eval: write the query without reading
the corpus first, so the phrasing doesn't accidentally echo a doc heading.

**`eval/ground_truth.json`**, one entry per query `id`:

```json
{
  "id": "q046",
  "query": "how do I use deno workspaces with multiple packages",
  "ideal_doc": "runtime/fundamentals/workspaces.md",
  "acceptable_docs": ["runtime/fundamentals/modules.md"],
  "rationale": "This page documents deno.json workspaces and npm-style monorepo layouts directly.",
  "coverage": "direct"
}
```

- `ideal_doc` is the doc path (relative to `corpus/deno-docs/`) that best
  answers the query, or `null` if nothing in the corpus does.
- `acceptable_docs` are other doc paths that would count as a correct-enough
  answer (used for the "acceptable hit" scoring bucket); `[]` if none.
- `coverage` is `"direct"` (a page answers it squarely), `"partial"` (only
  covered indirectly/incompletely), or `"none"` (nothing in the corpus
  answers it, `ideal_doc` must be `null` in this case; a correct agent
  should come back with no confident prediction, see `score()` in
  `cmd/evalrun/main.go`).

To find the actual ground truth for a new query, explore the corpus
yourself (`grep`/read through `corpus/deno-docs/`). Don't guess from the
query's phrasing alone, since the whole point of this eval set is that
queries and ground truth were derived independently.

New cases run automatically with the rest of the eval set; use `-id q046`
while iterating on a single one, or `-limit N` to sanity-check a batch
before committing to a full (paid) run.
