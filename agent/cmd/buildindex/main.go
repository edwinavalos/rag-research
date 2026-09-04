// Command buildindex generates the corpus catalog (internal/indexer) used
// by the single-shot exploration agent. This is deliberately a separate,
// out-of-band step from cmd/evalrun: the catalog is built once (or
// whenever the corpus changes), checked in, and loaded at agent-build
// time — not regenerated inside the eval loop, which would confound
// "does having a catalog help" with "how good is catalog generation
// today" on every run.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/workflow"

	"raggraph/internal/doctools"
	"raggraph/internal/indexer"
	"raggraph/internal/llmprovider"
)

const summarizeInstruction = `You are building a search index over a documentation corpus. You will be given a batch of documents, each as a path and a content excerpt.

For EACH document, in the same order as given, produce:
- "path": copied exactly as given — never alter it.
- "title": a short title for the page, drawn from its own heading if it has one.
- "summary": ONE sentence, at most 140 characters, describing SPECIFICALLY and CONCISELY what unique information this page provides. Be concrete about the exact topic, API, error, or task it covers — someone will scan hundreds of these summaries to judge whether a page answers a specific question, so avoid generic phrasing like "This page describes..." or "This document covers...". Name the actual thing.

Respond with ONLY a JSON object of the shape {"entries": [...]}, one array element per input document in the same order as given — no prose, no markdown fences.`

func main() {
	var (
		corpusDir    = flag.String("corpus", "../corpus/deno-docs", "path to the markdown corpus root")
		outPath      = flag.String("out", "../index/corpus_index.json", "path to write the catalog JSON")
		provider     = flag.String("provider", "openai", "which model provider to use: \"openai\" or \"anthropic\"")
		modelName    = flag.String("model", "", "model name; defaults to the smallest current model for the chosen provider")
		batchSize    = flag.Int("batch-size", 15, "how many docs to summarize per LLM call")
		excerptChars = flag.Int("excerpt-chars", 800, "how many characters of each doc's content to send")
		batchDelay   = flag.Duration("batch-delay", 1*time.Second, "pause between batches")
		requestDelay = flag.Duration("request-delay", 3*time.Second, "anthropic provider only; see llmprovider")
	)
	flag.Parse()

	root, err := filepath.Abs(*corpusDir)
	if err != nil {
		log.Fatalf("resolve corpus dir: %v", err)
	}
	paths, err := listMarkdownFiles(root)
	if err != nil {
		log.Fatalf("list corpus files: %v", err)
	}
	sort.Strings(paths)
	fmt.Printf("found %d markdown files under %s\n", len(paths), root)

	ctx := context.Background()
	m, err := llmprovider.Build(ctx, *provider, *modelName, *requestDelay)
	if err != nil {
		log.Fatalf("build model: %v", err)
	}
	summarizeAgent, err := buildSummarizeAgent(m)
	if err != nil {
		log.Fatalf("build summarize agent: %v", err)
	}

	var entries []indexer.Entry
	for start := 0; start < len(paths); start += *batchSize {
		end := min(start+*batchSize, len(paths))
		batch := paths[start:end]
		fmt.Printf("[%d/%d] summarizing batch of %d\n", start/(*batchSize)+1, (len(paths)+*batchSize-1)/(*batchSize), len(batch))

		got, err := summarizeBatch(ctx, summarizeAgent, root, batch, *excerptChars)
		if err != nil {
			log.Printf("    batch failed (%v), falling back to heuristic titles for this batch", err)
			got = nil
		}
		entries = append(entries, fillGaps(root, batch, got)...)

		if end < len(paths) && *batchDelay > 0 {
			time.Sleep(*batchDelay)
		}
	}

	if err := indexer.Save(*outPath, entries); err != nil {
		log.Fatalf("write catalog: %v", err)
	}
	fmt.Printf("wrote %d catalog entries to %s\n", len(entries), *outPath)
}

func listMarkdownFiles(root string) ([]string, error) {
	var paths []string
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if d.Name() == ".git" || d.Name() == "node_modules" {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(d.Name(), ".md") {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		paths = append(paths, filepath.ToSlash(rel))
		return nil
	})
	return paths, err
}

func buildSummarizeAgent(m model.LLM) (agent.Agent, error) {
	sa, err := llmagent.New(llmagent.Config{
		Name:         "summarize_batch",
		Description:  "Summarizes a batch of docs into short catalog entries.",
		Model:        m,
		Instruction:  summarizeInstruction,
		OutputSchema: batchSchema(),
	})
	if err != nil {
		return nil, err
	}
	node, err := workflow.NewAgentNodeTyped[string, batchResult](sa, workflow.NodeConfig{
		RetryConfig: workflow.DefaultRetryConfig(),
	})
	if err != nil {
		return nil, err
	}
	// Trailing identity FunctionNode: without it, a workflow that ends on
	// an AgentNode surfaces Output as nil to an external event-stream
	// consumer (see internal/exploreagent's New for the full explanation).
	finalizeNode := workflow.NewFunctionNode("finalize", func(_ agent.Context, r batchResult) (batchResult, error) {
		return r, nil
	}, workflow.NodeConfig{})
	edges := workflow.Chain(workflow.Start, node, finalizeNode)
	return workflowagent.New(workflowagent.Config{
		Name:        "batch_summarizer",
		Description: "Summarizes one batch of corpus docs into catalog entries.",
		Edges:       edges,
		SubAgents:   []agent.Agent{sa},
	})
}

// batchResult wraps the entry array in a root object: OpenAI's structured
// outputs (response_format) reject a top-level array schema — "schema
// must be a JSON Schema of 'type: object'" — so the batch response has to
// be an object with an array field, not a bare array.
type batchResult struct {
	Entries []indexer.Entry `json:"entries"`
}

func batchSchema() *genai.Schema {
	return &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"entries": {
				Type: genai.TypeArray,
				Items: &genai.Schema{
					Type: genai.TypeObject,
					Properties: map[string]*genai.Schema{
						"path":    {Type: genai.TypeString},
						"title":   {Type: genai.TypeString},
						"summary": {Type: genai.TypeString},
					},
					Required: []string{"path", "title", "summary"},
				},
			},
		},
		Required: []string{"entries"},
	}
}

func summarizeBatch(ctx context.Context, a agent.Agent, root string, batch []string, excerptChars int) ([]indexer.Entry, error) {
	var sb strings.Builder
	for _, p := range batch {
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(p)))
		if err != nil {
			continue
		}
		excerpt := string(data)
		if len(excerpt) > excerptChars {
			excerpt = excerpt[:excerptChars]
		}
		fmt.Fprintf(&sb, "### %s\n%s\n\n", p, excerpt)
	}

	r, err := runner.NewInMemory("buildindex", a)
	if err != nil {
		return nil, err
	}
	msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{genai.NewPartFromText(sb.String())}}
	sessionID := fmt.Sprintf("s-%d", time.Now().UnixNano())

	var result []indexer.Entry
	for ev, err := range r.Run(ctx, "buildindex", sessionID, msg, agent.RunConfig{}) {
		if err != nil {
			return nil, err
		}
		if ev == nil || ev.Output == nil {
			continue
		}
		if entries, ok := coerceEntries(ev.Output); ok {
			result = entries
		}
	}
	if result == nil {
		return nil, fmt.Errorf("no output observed")
	}
	return result, nil
}

// coerceEntries accepts either batchResult directly (a FunctionNode's
// typed return value) or map[string]any (schema-validated JSON text
// parsed via `var parsed any` — see the AgentNode Output-nil note in
// internal/exploreagent). Both carry the same {"entries": [...]} shape.
func coerceEntries(out any) ([]indexer.Entry, bool) {
	switch v := out.(type) {
	case batchResult:
		return v.Entries, true
	case map[string]any:
		data, err := json.Marshal(v)
		if err != nil {
			return nil, false
		}
		var r batchResult
		if err := json.Unmarshal(data, &r); err != nil {
			return nil, false
		}
		return r.Entries, true
	default:
		return nil, false
	}
}

// fillGaps ensures every path in batch has an entry: it keeps whatever
// the model returned for paths it covered, and heuristically fills in
// (from the doc's own first heading) any path the model dropped or
// mismatched — so a flaky batch degrades the catalog's quality for a few
// entries, not its completeness.
func fillGaps(root string, batch []string, got []indexer.Entry) []indexer.Entry {
	byPath := make(map[string]indexer.Entry, len(got))
	for _, e := range got {
		byPath[e.Path] = e
	}
	out := make([]indexer.Entry, 0, len(batch))
	for _, p := range batch {
		if e, ok := byPath[p]; ok && e.Summary != "" {
			out = append(out, e)
			continue
		}
		title := ""
		if data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(p))); err == nil {
			title = doctools.DocTitle(string(data))
		}
		out = append(out, indexer.Entry{Path: p, Title: title, Summary: title})
	}
	return out
}
