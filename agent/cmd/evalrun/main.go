// Command evalrun runs the graph exploration agent over the eval query
// set and scores it against ground truth. It is a batch harness, not the
// interactive ADK launcher: each query gets its own fresh in-memory
// session, run headlessly, with the terminal "finalize" node's output
// captured as the agent's answer.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/tool"

	"raggraph/internal/doctools"
	"raggraph/internal/exploreagent"
	"raggraph/internal/explorer"
	"raggraph/internal/indexer"
	"raggraph/internal/llmprovider"
)

type query struct {
	ID         string `json:"id"`
	Persona    string `json:"persona"`
	Query      string `json:"query"`
	Category   string `json:"category"`
	Difficulty string `json:"difficulty"`
}

type groundTruth struct {
	ID             string   `json:"id"`
	Query          string   `json:"query"`
	IdealDoc       *string  `json:"ideal_doc"`
	AcceptableDocs []string `json:"acceptable_docs"`
	Rationale      string   `json:"rationale"`
	Coverage       string   `json:"coverage"`
}

type evalRecord struct {
	ID             string  `json:"id"`
	Query          string  `json:"query"`
	Category       string  `json:"category"`
	Difficulty     string  `json:"difficulty"`
	Coverage       string  `json:"coverage"`
	IdealDoc       *string `json:"ideal_doc"`
	PredictedDoc   string  `json:"predicted_doc"`
	Confidence     float64 `json:"confidence"`
	Iterations     int64   `json:"iterations"`
	Reasoning      string  `json:"reasoning"`
	Hit            bool    `json:"hit"`
	Acceptable     bool    `json:"acceptable_hit"`
	ErrorMsg       string  `json:"error,omitempty"`
	DurationMillis int64   `json:"duration_ms"`
}

func main() {
	var (
		queriesPath  = flag.String("queries", "../eval/queries.json", "path to queries.json")
		gtPath       = flag.String("ground_truth", "../eval/ground_truth.json", "path to ground_truth.json")
		corpusDir    = flag.String("corpus", "../corpus/deno-docs", "path to the markdown corpus root")
		outPath      = flag.String("out", "../results/results.json", "path to write results JSON")
		method       = flag.String("method", "graph", "exploration method: \"graph\" (search<->judge loop) or \"single-shot\" (Claude Code Explore-agent style, one thorough parallel-tool-call turn)")
		provider     = flag.String("provider", "openai", "which model provider to use: \"openai\" or \"anthropic\"")
		modelName    = flag.String("model", "", "model name; defaults to the smallest current model for the chosen provider")
		limit        = flag.Int("limit", 0, "if >0, only run the first N queries (for cheap smoke tests)")
		onlyID       = flag.String("id", "", "if set, only run the query with this id")
		requestDelay = flag.Duration("request-delay", 3*time.Second, "fixed pause before every Anthropic API call (anthropic provider only; proactive rate-limit spacing)")
		queryDelay   = flag.Duration("query-delay", 5*time.Second, "fixed pause between queries, on top of request-delay")
		indexPath    = flag.String("index", "../index/corpus_index.json", "path to the corpus catalog built by cmd/buildindex (single-shot method only); missing file just means no catalog")
	)
	flag.Parse()

	queries, err := loadQueries(*queriesPath)
	if err != nil {
		log.Fatalf("load queries: %v", err)
	}
	gtByID, err := loadGroundTruth(*gtPath)
	if err != nil {
		log.Fatalf("load ground truth: %v", err)
	}

	if *onlyID != "" {
		var filtered []query
		for _, q := range queries {
			if q.ID == *onlyID {
				filtered = append(filtered, q)
			}
		}
		queries = filtered
	} else if *limit > 0 && *limit < len(queries) {
		queries = queries[:*limit]
	}

	ctx := context.Background()
	m, err := llmprovider.Build(ctx, *provider, *modelName, *requestDelay)
	if err != nil {
		log.Fatalf("build model: %v", err)
	}

	tools, err := doctools.New(*corpusDir)
	if err != nil {
		log.Fatalf("build doc tools: %v", err)
	}

	catalogEntries, err := indexer.Load(*indexPath)
	if err != nil {
		log.Fatalf("load catalog: %v", err)
	}
	catalog := indexer.FormatCatalog(catalogEntries)
	if catalog == "" {
		fmt.Println("(no catalog found at", *indexPath, "- running without one)")
	} else {
		fmt.Printf("loaded catalog: %d entries\n", len(catalogEntries))
	}

	explorerAgent, err := buildAgent(*method, m, tools, catalog)
	if err != nil {
		log.Fatalf("build agent: %v", err)
	}

	var results []evalRecord
	var hits, acceptableHits, errs, none int

	for i, q := range queries {
		gt := gtByID[q.ID]
		fmt.Printf("[%d/%d] %s: %q\n", i+1, len(queries), q.ID, q.Query)

		r, err := runner.NewInMemory("ragexp", explorerAgent)
		if err != nil {
			log.Fatalf("build runner: %v", err)
		}

		start := time.Now()
		result, runErr := runOne(ctx, r, q.Query)
		elapsed := time.Since(start)

		rec := evalRecord{
			ID:             q.ID,
			Query:          q.Query,
			Category:       q.Category,
			Difficulty:     q.Difficulty,
			DurationMillis: elapsed.Milliseconds(),
		}
		if gt != nil {
			rec.Coverage = gt.Coverage
			rec.IdealDoc = gt.IdealDoc
		}

		if runErr != nil {
			rec.ErrorMsg = runErr.Error()
			errs++
			fmt.Printf("    ERROR: %v\n", runErr)
		} else {
			rec.PredictedDoc = result.DocPath
			rec.Confidence = result.Confidence
			rec.Iterations = result.Iterations
			rec.Reasoning = result.Reasoning

			if gt != nil {
				rec.Hit, rec.Acceptable = score(result.DocPath, gt)
				if rec.Hit {
					hits++
				}
				if rec.Acceptable {
					acceptableHits++
				}
				if gt.IdealDoc == nil {
					none++
				}
			}
			fmt.Printf("    -> %q (confidence %.2f, %d rounds) hit=%v\n", result.DocPath, result.Confidence, result.Iterations, rec.Hit)
		}

		results = append(results, rec)

		if i < len(queries)-1 && *queryDelay > 0 {
			time.Sleep(*queryDelay)
		}
	}

	if err := writeResults(*outPath, results); err != nil {
		log.Fatalf("write results: %v", err)
	}

	total := len(results)
	fmt.Printf("\n=== %d queries: %d exact hits (%.1f%%), %d acceptable hits (%.1f%%), %d errors, %d unanswerable-in-gt ===\n",
		total, hits, pct(hits, total), acceptableHits, pct(acceptableHits, total), errs, none)
	fmt.Printf("results written to %s\n", *outPath)
}

// buildAgent selects the exploration method. Both return an agent.Agent
// with the same contract — Start receives the raw query string, the
// terminal node emits explorer.Result — so runOne and scoring are
// identical regardless of which one is chosen.
func buildAgent(method string, m model.LLM, tools []tool.Tool, catalog string) (agent.Agent, error) {
	switch method {
	case "graph":
		return explorer.New(m, tools)
	case "single-shot":
		return exploreagent.New(m, tools, catalog)
	default:
		return nil, fmt.Errorf("unknown method %q (want \"graph\" or \"single-shot\")", method)
	}
}

// coerceResult accepts an event.Output of either the concrete
// explorer.Result type — as produced by a Go FunctionNode return value
// (the graph method's "finalize" node) — or a generic map[string]any —
// as produced when a terminal AgentNode's schema-validated JSON text is
// parsed via `var parsed any` (the single-shot method, whose terminal
// node is the LLM agent itself, not a FunctionNode). Both shapes carry
// the same fields; this normalizes either into explorer.Result via a
// JSON round-trip for the map case.
func coerceResult(out any) (explorer.Result, bool) {
	switch v := out.(type) {
	case explorer.Result:
		return v, true
	case map[string]any:
		data, err := json.Marshal(v)
		if err != nil {
			return explorer.Result{}, false
		}
		var res explorer.Result
		if err := json.Unmarshal(data, &res); err != nil {
			return explorer.Result{}, false
		}
		return res, true
	default:
		return explorer.Result{}, false
	}
}

func runOne(ctx context.Context, r *runner.Runner, q string) (explorer.Result, error) {
	msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{genai.NewPartFromText(q)}}
	sessionID := fmt.Sprintf("s-%d", time.Now().UnixNano())

	var final *explorer.Result
	for ev, err := range r.Run(ctx, "evaluator", sessionID, msg, agent.RunConfig{}) {
		if err != nil {
			return explorer.Result{}, err
		}
		if ev == nil {
			continue
		}
		if os.Getenv("EVALRUN_DEBUG") != "" {
			text, role, longRunning := "", "", len(ev.LongRunningToolIDs)
			if ev.LLMResponse.Content != nil {
				role = ev.LLMResponse.Content.Role
				for _, p := range ev.LLMResponse.Content.Parts {
					if p != nil {
						text += p.Text
					}
				}
			}
			fmt.Printf("      [debug] author=%q final=%v role=%q longrunning=%d output=%#v text=%.200q\n", ev.Author, ev.IsFinalResponse(), role, longRunning, ev.Output, text)
		}
		if ev.Output == nil {
			continue
		}
		if res, ok := coerceResult(ev.Output); ok {
			final = &res
		}
	}
	if final == nil {
		return explorer.Result{}, fmt.Errorf("no finalize output observed")
	}
	return *final, nil
}

func score(predicted string, gt *groundTruth) (hit, acceptable bool) {
	predicted = strings.TrimSpace(predicted)
	if gt.IdealDoc == nil {
		// Ground truth says nothing in the corpus covers this query; a
		// correct agent should come back with no confident candidate.
		return predicted == "", predicted == ""
	}
	if predicted == "" {
		return false, false
	}
	if pathsEqual(predicted, *gt.IdealDoc) {
		return true, true
	}
	for _, alt := range gt.AcceptableDocs {
		if pathsEqual(predicted, alt) {
			return false, true
		}
	}
	return false, false
}

func pathsEqual(a, b string) bool {
	return filepath.Clean(strings.TrimPrefix(a, "./")) == filepath.Clean(strings.TrimPrefix(b, "./"))
}

func pct(n, total int) float64 {
	if total == 0 {
		return 0
	}
	return 100 * float64(n) / float64(total)
}

func loadQueries(path string) ([]query, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var qs []query
	if err := json.Unmarshal(data, &qs); err != nil {
		return nil, err
	}
	return qs, nil
}

func loadGroundTruth(path string) (map[string]*groundTruth, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var gts []groundTruth
	if err := json.Unmarshal(data, &gts); err != nil {
		return nil, err
	}
	out := make(map[string]*groundTruth, len(gts))
	for i := range gts {
		out[gts[i].ID] = &gts[i]
	}
	return out, nil
}

func writeResults(path string, results []evalRecord) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}
