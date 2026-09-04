// Package explorer builds the graph-based document-exploration agent:
// a "search" node that grep/glob/reads the corpus, a "judge" node that
// decides whether the evidence found so far answers the query, and a
// conditional back-edge that loops search->judge up to MaxIterations
// times before a "finalize" node emits the result. This is the explicit,
// inspectable graph version of the same idea as a plain tool-calling
// agent loop (which Claude Code itself uses) — the loop is pulled out
// into graph state so each round can be logged and the stopping
// condition tuned independently of the model's own judgment.
package explorer

import (
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/workflow"
)

// MaxIterations bounds the search<->judge loop regardless of what the
// judge decides, so a model that never signals "done" can't loop forever.
const MaxIterations = 4

// Record threads the running state of one exploration through the graph:
// the original query, how many search rounds have run, what the search
// node should focus on this round, the accumulated findings, and the
// current best guess at the answer document.
type Record struct {
	Query         string   `json:"query"`
	Iteration     int64    `json:"iteration"`
	Instruction   string   `json:"instruction"`
	Findings      []string `json:"findings"`
	CandidatePath string   `json:"candidate_path"`
}

// Judged is a Record plus the judge node's verdict for this round.
type Judged struct {
	Query           string   `json:"query"`
	Iteration       int64    `json:"iteration"`
	Findings        []string `json:"findings"`
	CandidatePath   string   `json:"candidate_path"`
	Done            bool     `json:"done"`
	Confidence      float64  `json:"confidence"`
	Reasoning       string   `json:"reasoning"`
	NextInstruction string   `json:"next_instruction"`
}

// Result is the exploration's final answer, handed back to the caller.
type Result struct {
	DocPath    string  `json:"doc_path"`
	Confidence float64 `json:"confidence"`
	Iterations int64   `json:"iterations"`
	Reasoning  string  `json:"reasoning"`
}

const searchInstruction = `You are searching a corpus of Deno documentation markdown files to find the single best document that answers a user's query.

You are given a JSON object describing the exploration so far: the original query, prior findings from earlier search rounds (if any), and an instruction for what to focus on this round.

Use the glob_docs, grep_docs, and read_doc tools to investigate. Prefer grep_docs to find where a term or error is actually discussed; use glob_docs for broad path/filename pattern discovery (e.g. "**/kv/*.md"); use read_doc to confirm a candidate file really answers the query before naming it.

When you are done searching this round, respond with ONLY a JSON object (no prose, no markdown fences) with this shape:
{
  "query": "<copy the input query unchanged>",
  "iteration": <copy the input iteration unchanged>,
  "instruction": "<copy the input instruction unchanged>",
  "findings": [<the input findings array, PLUS exactly one new string appended describing what you found or ruled out this round, including any candidate file paths and why>],
  "candidate_path": "<your current best-guess file path relative to the corpus root, e.g. runtime/fundamentals/permissions.md, or empty string if you have no candidate yet>"
}`

const judgeInstruction = `You are deciding whether an exploration of the Deno documentation has found the right answer document yet.

You are given a JSON Record: the query, the findings gathered across search rounds so far, and the current candidate document path.

Decide: is the current candidate_path (if any) clearly the right document to answer the query, based on the findings? Or does another search round need to happen, and if so what should it focus on (e.g. a different search term, a different doc section, ruling out a false-positive candidate)?

Respond with ONLY a JSON object (no prose, no markdown fences) with this shape:
{
  "query": "<copy the input query unchanged>",
  "iteration": <copy the input iteration unchanged>,
  "findings": [<copy the input findings array unchanged>],
  "candidate_path": "<the best candidate path — usually the input's candidate_path, but you may clear it to \"\" if the findings show it's wrong>",
  "done": <true if candidate_path is confidently correct, or if you are confident NO document in this corpus answers the query and further search would not help; false otherwise>,
  "confidence": <0.0 to 1.0 confidence in candidate_path being correct, or in "no document covers this" if candidate_path is empty and done is true>,
  "reasoning": "<one or two sentences justifying the decision>",
  "next_instruction": "<if done is false, a specific instruction for the next search round — a new term to grep, a section to check, or why the current candidate needs re-checking; empty string if done>"
}`

// New builds the exploration workflow agent. m is used for both the
// search and judge nodes (a small experiment; using two different models
// per node — e.g. a cheaper one for search, a stronger one for judge —
// is a natural next iteration, not implemented here).
func New(m model.LLM, tools []tool.Tool) (agent.Agent, error) {
	searchAgent, err := llmagent.New(llmagent.Config{
		Name:         "search",
		Description:  "Explores the doc corpus with list/grep/read tools for one round.",
		Model:        m,
		Instruction:  searchInstruction,
		Tools:        tools,
		OutputSchema: recordSchema(),
	})
	if err != nil {
		return nil, fmt.Errorf("explorer: build search agent: %w", err)
	}

	judgeAgent, err := llmagent.New(llmagent.Config{
		Name:         "judge",
		Description:  "Decides whether the exploration has found the answer or needs another round.",
		Model:        m,
		Instruction:  judgeInstruction,
		OutputSchema: judgedSchema(),
	})
	if err != nil {
		return nil, fmt.Errorf("explorer: build judge agent: %w", err)
	}

	initNode := workflow.NewFunctionNode("init", initRecord, workflow.NodeConfig{})

	searchNode, err := workflow.NewAgentNodeTyped[Record, Record](searchAgent, workflow.NodeConfig{
		RetryConfig: workflow.DefaultRetryConfig(),
	})
	if err != nil {
		return nil, fmt.Errorf("explorer: build search node: %w", err)
	}

	judgeNode, err := workflow.NewAgentNodeTyped[Record, Judged](judgeAgent, workflow.NodeConfig{
		RetryConfig: workflow.DefaultRetryConfig(),
	})
	if err != nil {
		return nil, fmt.Errorf("explorer: build judge node: %w", err)
	}

	routeNode := workflow.NewEmittingFunctionNode("route", route, workflow.NodeConfig{})

	finalizeNode := workflow.NewFunctionNode("finalize", finalize, workflow.NodeConfig{})

	eb := workflow.NewEdgeBuilder()
	eb.Add(workflow.Start, initNode)
	eb.Add(initNode, searchNode)
	eb.Add(searchNode, judgeNode)
	eb.Add(judgeNode, routeNode)
	eb.AddRoute(routeNode, searchNode, workflow.StringRoute("continue"))
	eb.AddRoute(routeNode, finalizeNode, workflow.StringRoute("finalize"))

	return workflowagent.New(workflowagent.Config{
		Name:        "deno_doc_explorer",
		Description: "Finds the Deno doc that best answers a query by iteratively searching the corpus.",
		Edges:       eb.Build(),
		SubAgents:   []agent.Agent{searchAgent, judgeAgent},
	})
}

func initRecord(_ agent.Context, query string) (Record, error) {
	return Record{
		Query:       query,
		Iteration:   0,
		Instruction: "Begin your search. Look for the document that most directly answers the query.",
	}, nil
}

func route(ctx agent.Context, j Judged, emit func(*session.Event) error) (any, error) {
	forceStop := j.Iteration+1 >= MaxIterations

	ev := session.NewEvent(ctx, ctx.InvocationID())
	if j.Done || forceStop {
		ev.Routes = []string{"finalize"}
		ev.Output = j
		return nil, emit(ev)
	}

	ev.Routes = []string{"continue"}
	ev.Output = Record{
		Query:         j.Query,
		Iteration:     j.Iteration + 1,
		Instruction:   j.NextInstruction,
		Findings:      j.Findings,
		CandidatePath: j.CandidatePath,
	}
	return nil, emit(ev)
}

func recordSchema() *genai.Schema {
	return &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"query":          {Type: genai.TypeString},
			"iteration":      {Type: genai.TypeInteger},
			"instruction":    {Type: genai.TypeString},
			"findings":       {Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeString}},
			"candidate_path": {Type: genai.TypeString},
		},
		Required: []string{"query", "iteration", "instruction", "findings", "candidate_path"},
	}
}

func judgedSchema() *genai.Schema {
	return &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"query":            {Type: genai.TypeString},
			"iteration":        {Type: genai.TypeInteger},
			"findings":         {Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeString}},
			"candidate_path":   {Type: genai.TypeString},
			"done":             {Type: genai.TypeBoolean},
			"confidence":       {Type: genai.TypeNumber},
			"reasoning":        {Type: genai.TypeString},
			"next_instruction": {Type: genai.TypeString},
		},
		Required: []string{"query", "iteration", "findings", "candidate_path", "done", "confidence", "reasoning", "next_instruction"},
	}
}

func finalize(_ agent.Context, j Judged) (Result, error) {
	return Result{
		DocPath:    j.CandidatePath,
		Confidence: j.Confidence,
		Iterations: j.Iteration + 1,
		Reasoning:  j.Reasoning,
	}, nil
}
