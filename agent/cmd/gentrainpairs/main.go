// Command gentrainpairs generates synthetic (query, doc) training pairs
// for fine-tuning a bi-encoder retriever on this corpus. Modeled directly
// on cmd/buildindex: batch the corpus through an LLM, ask it to invent
// realistic user questions each doc would answer. This is deliberately a
// separate, out-of-band step from training itself — the pairs are
// generated once, checked in, and reused across training runs.
//
// The eval set (eval/queries.json) must stay untouched by training, or
// scoring against it just measures memorization instead of retrieval
// quality. Since queries here are generated fresh from doc content (never
// shown the real eval questions), leakage would only happen by
// coincidence — dropLeakedPairs() catches and reports any training query
// that ended up a near-duplicate of a real eval query anyway.
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

	"raggraph/internal/llmprovider"
)

const genInstruction = `You are building training data for a document retrieval model. You will be given a batch of documents, each as a path and its full (or excerpted) content.

For EACH document, in the same order as given, invent %d realistic questions a real user might type into a search box, such that THIS document is the correct answer. Vary the phrasing and specificity across the %d questions for a given doc (e.g. one terse keyword-style query, one full-sentence question, one from a beginner's phrasing) — don't just reword the doc's own heading.

Respond with ONLY a JSON object of the shape {"entries": [...]}, one array element per input document in the same order as given, each with "path" (copy the value after "PATH:" exactly) and "queries" (an array of %d strings). No prose, no markdown fences.`

type Pair struct {
	Query string `json:"query"`
	Doc   string `json:"doc"`
}

func main() {
	var (
		corpusDir       = flag.String("corpus", "../corpus/deno-docs", "path to the markdown corpus root")
		outPath         = flag.String("out", "../train/pairs.json", "path to write the training pairs JSON")
		evalQueriesPath = flag.String("eval-queries", "../eval/queries.json", "eval queries file to check for leakage against")
		provider        = flag.String("provider", "openai", "which model provider to use: \"openai\" or \"anthropic\"")
		modelName       = flag.String("model", "", "model name; defaults to the smallest current model for the chosen provider")
		queriesPerDoc   = flag.Int("queries-per-doc", 3, "how many synthetic queries to generate per document")
		batchSize       = flag.Int("batch-size", 8, "how many docs to send per LLM call")
		excerptChars    = flag.Int("excerpt-chars", 2500, "how many characters of each doc's content to send")
		batchDelay      = flag.Duration("batch-delay", 1*time.Second, "pause between batches")
		requestDelay    = flag.Duration("request-delay", 3*time.Second, "anthropic provider only; see llmprovider")
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
	genAgent, err := buildGenAgent(m, *queriesPerDoc)
	if err != nil {
		log.Fatalf("build gen agent: %v", err)
	}

	var pairs []Pair
	numBatches := (len(paths) + *batchSize - 1) / *batchSize
	for start := 0; start < len(paths); start += *batchSize {
		end := min(start+*batchSize, len(paths))
		batch := paths[start:end]
		fmt.Printf("[%d/%d] generating queries for batch of %d\n", start/(*batchSize)+1, numBatches, len(batch))

		got, err := genBatch(ctx, genAgent, root, batch, *excerptChars)
		if err != nil {
			log.Printf("    batch failed (%v), skipping this batch", err)
			got = nil
		}
		for _, e := range got {
			for _, q := range e.Queries {
				q = strings.TrimSpace(q)
				if q != "" {
					pairs = append(pairs, Pair{Query: q, Doc: e.Path})
				}
			}
		}

		if end < len(paths) && *batchDelay > 0 {
			time.Sleep(*batchDelay)
		}
	}
	fmt.Printf("generated %d raw pairs\n", len(pairs))

	evalQueries, err := loadEvalQueryTexts(*evalQueriesPath)
	if err != nil {
		log.Printf("warning: could not load eval queries for leakage check (%v) — skipping check", err)
	} else {
		before := len(pairs)
		pairs = dropLeakedPairs(pairs, evalQueries)
		if dropped := before - len(pairs); dropped > 0 {
			fmt.Printf("dropped %d pairs as near-duplicates of eval queries\n", dropped)
		} else {
			fmt.Printf("no leakage detected against %d eval queries\n", len(evalQueries))
		}
	}

	if err := savePairs(*outPath, pairs); err != nil {
		log.Fatalf("write pairs: %v", err)
	}
	fmt.Printf("wrote %d training pairs to %s\n", len(pairs), *outPath)
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

func buildGenAgent(m model.LLM, queriesPerDoc int) (agent.Agent, error) {
	instruction := fmt.Sprintf(genInstruction, queriesPerDoc, queriesPerDoc, queriesPerDoc)
	ga, err := llmagent.New(llmagent.Config{
		Name:                  "generate_train_queries",
		Description:           "Generates synthetic training queries for a batch of docs.",
		Model:                 m,
		Instruction:           instruction,
		GenerateContentConfig: &genai.GenerateContentConfig{MaxOutputTokens: 8192},
		OutputSchema:          batchSchema(),
	})
	if err != nil {
		return nil, err
	}
	node, err := workflow.NewAgentNodeTyped[string, batchResult](ga, workflow.NodeConfig{
		RetryConfig: workflow.DefaultRetryConfig(),
	})
	if err != nil {
		return nil, err
	}
	finalizeNode := workflow.NewFunctionNode("finalize", func(_ agent.Context, r batchResult) (batchResult, error) {
		return r, nil
	}, workflow.NodeConfig{})
	edges := workflow.Chain(workflow.Start, node, finalizeNode)
	return workflowagent.New(workflowagent.Config{
		Name:        "batch_query_generator",
		Description: "Generates synthetic training queries for one batch of corpus docs.",
		Edges:       edges,
		SubAgents:   []agent.Agent{ga},
	})
}

type docQueries struct {
	Path    string   `json:"path"`
	Queries []string `json:"queries"`
}

type batchResult struct {
	Entries []docQueries `json:"entries"`
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
						"path": {Type: genai.TypeString},
						"queries": {
							Type:  genai.TypeArray,
							Items: &genai.Schema{Type: genai.TypeString},
						},
					},
					Required: []string{"path", "queries"},
				},
			},
		},
		Required: []string{"entries"},
	}
}

func genBatch(ctx context.Context, a agent.Agent, root string, batch []string, excerptChars int) ([]docQueries, error) {
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
		fmt.Fprintf(&sb, "--- BEGIN DOC ---\nPATH: %s\nCONTENT:\n%s\n--- END DOC ---\n\n", p, excerpt)
	}

	r, err := runner.NewInMemory("gentrainpairs", a)
	if err != nil {
		return nil, err
	}
	msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{genai.NewPartFromText(sb.String())}}
	sessionID := fmt.Sprintf("s-%d", time.Now().UnixNano())

	var result []docQueries
	for ev, err := range r.Run(ctx, "gentrainpairs", sessionID, msg, agent.RunConfig{}) {
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

func coerceEntries(out any) ([]docQueries, bool) {
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

func loadEvalQueryTexts(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var entries []struct {
		Query string `json:"query"`
	}
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil, err
	}
	texts := make([]string, len(entries))
	for i, e := range entries {
		texts[i] = e.Query
	}
	return texts, nil
}

// dropLeakedPairs removes any training pair whose query is a
// near-duplicate (token Jaccard similarity > 0.7) of a real eval query,
// so a training run can never get credit for having memorized the exact
// phrasing the eval set will later score it on.
func dropLeakedPairs(pairs []Pair, evalQueries []string) []Pair {
	evalTokenSets := make([][]string, len(evalQueries))
	for i, q := range evalQueries {
		evalTokenSets[i] = tokenize(q)
	}
	out := make([]Pair, 0, len(pairs))
	for _, p := range pairs {
		pTokens := tokenize(p.Query)
		leaked := false
		for _, evalTokens := range evalTokenSets {
			if jaccard(pTokens, evalTokens) > 0.7 {
				leaked = true
				break
			}
		}
		if !leaked {
			out = append(out, p)
		}
	}
	return out
}

func tokenize(s string) []string {
	fields := strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9')
	})
	return fields
}

func jaccard(a, b []string) float64 {
	setA := make(map[string]bool, len(a))
	for _, t := range a {
		setA[t] = true
	}
	setB := make(map[string]bool, len(b))
	for _, t := range b {
		setB[t] = true
	}
	if len(setA) == 0 || len(setB) == 0 {
		return 0
	}
	inter := 0
	for t := range setA {
		if setB[t] {
			inter++
		}
	}
	union := len(setA) + len(setB) - inter
	return float64(inter) / float64(union)
}

func savePairs(path string, pairs []Pair) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(pairs, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}
