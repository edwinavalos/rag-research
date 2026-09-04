// Package exploreagent is a single-shot document search agent modeled
// directly on Claude Code's real built-in "Explore" subagent
// (src/tools/AgentTool/built-in/exploreAgent.ts in the claude-code repo).
//
// Unlike internal/explorer's graph (search <-> judge, looped up to
// MaxIterations rounds), the real Explore agent has no loop and no judge
// step at all: it is one agent turn that is instructed to search
// thoroughly — favoring parallel tool calls over sequential narrowing —
// and report once. This package reproduces that shape so it can run
// side by side against the graph version on the same eval set.
package exploreagent

import (
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/workflow"

	"raggraph/internal/explorer"
)

// exploreInstruction adapts the real Explore agent's system prompt
// (read-only, favor parallel tool calls, adapt thoroughness, report once)
// to this experiment's doc-search tools. The "spawn multiple parallel
// tool calls" and "don't stop after your first plausible match" lines are
// the two behaviors the graph version's search node was never told to do
// — this is the thing being tested.
const exploreInstruction = `You are a document search specialist. You find the single Deno documentation page that best answers a user's query, by searching a local corpus of markdown files.

This is a READ-ONLY exploration task: you only search and read, you never modify anything.

Guidelines:
- Use grep_docs to search file contents with regex — usually the fastest path to the right doc. Pass context: 3 (or similar) to see surrounding lines per match, so you can judge relevance without a separate read_doc call.
- Use glob_docs for broad filename/path pattern discovery, e.g. "**/*.md", "runtime/**/*.md", "**/kv/*.md".
- Use read_doc when you have a specific candidate file path to confirm.
- Be thorough: this task warrants a VERY THOROUGH search. Wherever possible, issue multiple tool calls in parallel — several grep_docs calls with different terms, or several read_doc calls on candidate files at once — rather than one call at a time, so you cover more ground per round of thinking.
- Do not stop at your first plausible match. Actively look for a more specific or more directly relevant document before committing, and rule out near-miss candidates (e.g. an index/overview page vs. the specific sub-page that actually answers the query) by reading them.
- If, after a thorough search, nothing in the corpus actually answers the query, say so — an empty doc_path with your reasoning is a correct answer, not a failure.

When you are done searching, respond with ONLY a JSON object (no prose, no markdown fences) with this shape:
{
  "doc_path": "<file path relative to the corpus root that best answers the query, e.g. runtime/fundamentals/permissions.md, or empty string if nothing covers it>",
  "confidence": <0.0 to 1.0>,
  "iterations": 1,
  "reasoning": "<one or two sentences justifying the choice>"
}`

// New builds the single-shot exploration agent. Its input/output contract
// (Start receives the raw query string, the terminal node emits
// explorer.Result) matches internal/explorer's graph exactly, so
// cmd/evalrun's scoring and output-capture code works unchanged for both.
func New(m model.LLM, tools []tool.Tool) (agent.Agent, error) {
	searchAgent, err := llmagent.New(llmagent.Config{
		Name:         "explore_singleshot",
		Description:  "Searches the doc corpus thoroughly in one turn (parallel tool calls, no loop) and reports the best-answer document.",
		Model:        m,
		Instruction:  exploreInstruction,
		Tools:        tools,
		OutputSchema: resultSchema(),
	})
	if err != nil {
		return nil, fmt.Errorf("exploreagent: build search agent: %w", err)
	}

	node, err := workflow.NewAgentNodeTyped[string, explorer.Result](searchAgent, workflow.NodeConfig{
		RetryConfig: workflow.DefaultRetryConfig(),
	})
	if err != nil {
		return nil, fmt.Errorf("exploreagent: build search node: %w", err)
	}

	// A trailing identity FunctionNode, exactly mirroring internal/explorer's
	// finalize node. Required, not cosmetic: adk-go schema-validates an
	// AgentNode's raw model text asynchronously in the scheduler's internal
	// bookkeeping, and the *external* event stream (what cmd/evalrun reads)
	// observes the event before that validated value lands on it — so a
	// workflow ending directly on an AgentNode surfaces Output as nil to an
	// outside caller. A FunctionNode sidesteps this because it constructs
	// its own event with Output set directly from its Go return value.
	finalizeNode := workflow.NewFunctionNode("finalize", finalize, workflow.NodeConfig{})

	edges := workflow.Chain(workflow.Start, node, finalizeNode)

	return workflowagent.New(workflowagent.Config{
		Name:        "deno_doc_explore_singleshot",
		Description: "Finds the Deno doc that best answers a query in one thorough, parallel-tool-call search turn (Claude Code Explore-agent style).",
		Edges:       edges,
		SubAgents:   []agent.Agent{searchAgent},
	})
}

func finalize(_ agent.Context, r explorer.Result) (explorer.Result, error) {
	return r, nil
}

func resultSchema() *genai.Schema {
	return &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"doc_path":   {Type: genai.TypeString},
			"confidence": {Type: genai.TypeNumber},
			"iterations": {Type: genai.TypeInteger},
			"reasoning":  {Type: genai.TypeString},
		},
		Required: []string{"doc_path", "confidence", "iterations", "reasoning"},
	}
}
