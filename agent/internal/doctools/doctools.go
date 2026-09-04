// Package doctools implements the three tools the exploration agent uses
// to find documents — glob (file pattern matching), grep (content
// search), and read (fetch one file) — modeled directly on Claude Code's
// real GlobTool/GrepTool/FileReadTool (src/tools/{Glob,Grep,FileRead}Tool
// in the claude-code repo). All three are sandboxed to a single root
// directory so the agent can only ever see the corpus, mirroring how
// Claude Code scopes its own file tools to a project root.
package doctools

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/bmatcuk/doublestar/v4"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

const (
	maxGlobResults = 200
	maxGrepMatches = 60
	maxReadBytes   = 60_000 // ~15k tokens; plenty for one doc page
)

// New builds the three doc-exploration tools rooted at corpusDir (an
// absolute or relative path to a directory of markdown files, e.g.
// corpus/deno-docs).
func New(corpusDir string) ([]tool.Tool, error) {
	root, err := filepath.Abs(corpusDir)
	if err != nil {
		return nil, fmt.Errorf("doctools: resolve corpus dir: %w", err)
	}
	if info, err := os.Stat(root); err != nil || !info.IsDir() {
		return nil, fmt.Errorf("doctools: corpus dir %q not found", root)
	}

	globTool, err := functiontool.New(functiontool.Config{
		Name: "glob_docs",
		Description: "Fast file pattern matching over the doc corpus, using standard glob syntax including \"**\" " +
			"for recursive matching (e.g. \"**/*.md\", \"runtime/**/*.md\", \"**/kv/*.md\", \"deploy/*.md\"). " +
			"Returns matching paths relative to the corpus root, sorted. Use this to find files by name/path " +
			"pattern; use grep_docs instead when you need to search file contents.",
	}, makeGlob(root))
	if err != nil {
		return nil, err
	}

	grepTool, err := functiontool.New(functiontool.Config{
		Name: "grep_docs",
		Description: "Search the text content of every markdown file in the corpus for a regular expression " +
			"(case-insensitive). Returns matching file paths with the matching line. Use this to find which " +
			"doc actually discusses a term, error message, or concept, not just which filename mentions it. " +
			"Pass context (or context_before/context_after) to get N surrounding lines per match — use this " +
			"to judge relevance from the grep call itself instead of a follow-up read_doc round-trip.",
	}, makeGrep(root))
	if err != nil {
		return nil, err
	}

	readTool, err := functiontool.New(functiontool.Config{
		Name:        "read_doc",
		Description: "Read the full text content of one markdown file, given its path relative to the corpus root (as returned by glob_docs, grep_docs, or rank_docs).",
	}, makeRead(root))
	if err != nil {
		return nil, err
	}

	rankTool, err := functiontool.New(functiontool.Config{
		Name: "rank_docs",
		Description: "Rank every doc in the corpus by relevance to a free-text query and return a shortlist, " +
			"each with its score and title — unlike grep_docs (which returns unordered line matches), this " +
			"gives you a single ranked comparison across candidates in one call. Scoring weights path/filename " +
			"term matches highest, then title/heading matches, then body term matches. Good first call for a " +
			"new query, or to directly compare several already-found candidates against each other.",
	}, makeRank(root))
	if err != nil {
		return nil, err
	}

	return []tool.Tool{globTool, grepTool, readTool, rankTool}, nil
}

type globInput struct {
	Pattern string `json:"pattern"`
}

type globOutput struct {
	Paths     []string `json:"paths"`
	Total     int      `json:"total_matching"`
	Truncated bool     `json:"truncated"`
}

func makeGlob(root string) func(agent.Context, globInput) (globOutput, error) {
	return func(_ agent.Context, in globInput) (globOutput, error) {
		pattern := in.Pattern
		if pattern == "" {
			pattern = "**/*.md"
		}
		if !doublestar.ValidatePattern(pattern) {
			return globOutput{}, fmt.Errorf("doctools: invalid glob pattern %q", pattern)
		}
		var matches []string
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
			rel, _ := filepath.Rel(root, path)
			rel = filepath.ToSlash(rel)
			if ok, _ := doublestar.Match(pattern, rel); ok {
				matches = append(matches, rel)
			}
			return nil
		})
		if err != nil {
			return globOutput{}, fmt.Errorf("doctools: walk: %w", err)
		}
		sort.Strings(matches)
		total := len(matches)
		truncated := total > maxGlobResults
		if truncated {
			matches = matches[:maxGlobResults]
		}
		return globOutput{Paths: matches, Total: total, Truncated: truncated}, nil
	}
}

type grepInput struct {
	Pattern string `json:"pattern"`
	// ContextBefore/ContextAfter mirror ripgrep's -B/-A: lines of
	// surrounding context to include around each match, so the caller
	// can judge relevance from the grep call itself rather than needing
	// a follow-up read_doc round-trip. Context is symmetric: set only
	// Context (mirrors -C) to apply the same value to both sides.
	ContextBefore int `json:"context_before"`
	ContextAfter  int `json:"context_after"`
	Context       int `json:"context"`
}

type grepMatch struct {
	File string `json:"file"`
	Line int    `json:"line"`
	// Snippet is the matching line plus any requested context, each
	// line prefixed with its 1-based line number; the actual matching
	// line is marked with ">" (mirrors ripgrep -n output with a match
	// marker, since context lines alone don't say which one matched).
	Snippet string `json:"snippet"`
}

type grepOutput struct {
	Matches   []grepMatch `json:"matches"`
	Truncated bool        `json:"truncated"`
}

func makeGrep(root string) func(agent.Context, grepInput) (grepOutput, error) {
	return func(_ agent.Context, in grepInput) (grepOutput, error) {
		re, err := regexp.Compile("(?i)" + in.Pattern)
		if err != nil {
			return grepOutput{}, fmt.Errorf("doctools: invalid regex %q: %w", in.Pattern, err)
		}
		before, after := in.ContextBefore, in.ContextAfter
		if in.Context > 0 {
			before, after = in.Context, in.Context
		}
		var out grepOutput
		err = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if out.Truncated {
				return nil
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
			data, err := os.ReadFile(path)
			if err != nil {
				return nil // skip unreadable files rather than failing the whole search
			}
			rel, _ := filepath.Rel(root, path)
			lines := strings.Split(string(data), "\n")
			for i, line := range lines {
				if re.MatchString(line) {
					out.Matches = append(out.Matches, grepMatch{
						File:    rel,
						Line:    i + 1,
						Snippet: buildSnippet(lines, i, before, after),
					})
					if len(out.Matches) >= maxGrepMatches {
						out.Truncated = true
						return nil
					}
				}
			}
			return nil
		})
		if err != nil {
			return grepOutput{}, fmt.Errorf("doctools: walk: %w", err)
		}
		return out, nil
	}
}

// buildSnippet renders the matched line (index i in lines) plus up to
// `before`/`after` lines of surrounding context, each line numbered
// 1-based, with the matched line marked.
func buildSnippet(lines []string, i, before, after int) string {
	start := i - before
	if start < 0 {
		start = 0
	}
	end := i + after
	if end > len(lines)-1 {
		end = len(lines) - 1
	}
	var sb strings.Builder
	for n := start; n <= end; n++ {
		marker := "  "
		if n == i {
			marker = "> "
		}
		fmt.Fprintf(&sb, "%s%d: %s\n", marker, n+1, strings.TrimRight(lines[n], "\r"))
	}
	return strings.TrimSuffix(sb.String(), "\n")
}

// rank_docs — a weighted keyword scorer modeled on Claude Code's
// ToolSearchTool (src/tools/ToolSearchTool/ToolSearchTool.ts
// searchToolsWithKeywords), which solves the same shape of problem for
// finding the right tool among many: score every candidate by term
// overlap, weighted by where the term matched, and return the ranked
// shortlist — rather than the model having to eyeball an unordered dump
// of grep hits itself.
const (
	defaultRankLimit = 10
	maxRankLimit     = 50

	scorePathExact  = 10
	scorePathPrefix = 5
	scorePathSubstr = 3
	scoreTitle      = 6
	scoreBody       = 2
)

type rankInput struct {
	Query string `json:"query"`
	Limit int    `json:"limit"` // default 10, max 50; 0 means default
}

type rankedDoc struct {
	File  string `json:"file"`
	Score int    `json:"score"`
	Title string `json:"title"`
}

type rankOutput struct {
	Results         []rankedDoc `json:"results"`
	TotalCandidates int         `json:"total_candidates"`
}

var termSplitRe = regexp.MustCompile(`[^a-zA-Z0-9]+`)

func splitTerms(s string) []string {
	var terms []string
	for _, t := range termSplitRe.Split(strings.ToLower(s), -1) {
		if len(t) >= 2 {
			terms = append(terms, t)
		}
	}
	return terms
}

// docTitle returns the first Markdown H1/H2 heading, or the first
// non-empty line if the doc has none — a cheap proxy for "what is this
// page about" without a full frontmatter/AST parser.
func docTitle(content string) string {
	for _, line := range strings.Split(content, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		return strings.TrimLeft(strings.TrimSpace(strings.TrimLeft(line, "#")), " ")
	}
	return ""
}

func makeRank(root string) func(agent.Context, rankInput) (rankOutput, error) {
	return func(_ agent.Context, in rankInput) (rankOutput, error) {
		terms := splitTerms(in.Query)
		if len(terms) == 0 {
			return rankOutput{}, fmt.Errorf("doctools: query has no usable terms")
		}
		limit := in.Limit
		if limit <= 0 {
			limit = defaultRankLimit
		}
		if limit > maxRankLimit {
			limit = maxRankLimit
		}

		termRes := make([]*regexp.Regexp, len(terms))
		for i, term := range terms {
			termRes[i] = regexp.MustCompile(`\b` + regexp.QuoteMeta(term) + `\b`)
		}

		var candidates int
		var scored []rankedDoc
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
			data, err := os.ReadFile(path)
			if err != nil {
				return nil
			}
			candidates++
			rel, _ := filepath.Rel(root, path)
			rel = filepath.ToSlash(rel)
			content := string(data)
			title := docTitle(content)
			score := scoreDoc(terms, termRes, rel, title, content)
			if score > 0 {
				scored = append(scored, rankedDoc{File: rel, Score: score, Title: title})
			}
			return nil
		})
		if err != nil {
			return rankOutput{}, fmt.Errorf("doctools: walk: %w", err)
		}

		sort.Slice(scored, func(i, j int) bool {
			if scored[i].Score != scored[j].Score {
				return scored[i].Score > scored[j].Score
			}
			return scored[i].File < scored[j].File // stable tie-break
		})
		if len(scored) > limit {
			scored = scored[:limit]
		}
		return rankOutput{Results: scored, TotalCandidates: candidates}, nil
	}
}

func scoreDoc(terms []string, termRes []*regexp.Regexp, path, title, content string) int {
	pathParts := termSplitRe.Split(strings.ToLower(path), -1)
	pathLower := strings.ToLower(path)
	titleLower := strings.ToLower(title)
	contentLower := strings.ToLower(content)

	score := 0
	for i, term := range terms {
		switch {
		case containsExact(pathParts, term):
			score += scorePathExact
		case containsPrefix(pathParts, term):
			score += scorePathPrefix
		case strings.Contains(pathLower, term):
			score += scorePathSubstr
		}
		if termRes[i].MatchString(titleLower) {
			score += scoreTitle
		}
		if termRes[i].MatchString(contentLower) {
			score += scoreBody
		}
	}
	return score
}

func containsExact(parts []string, term string) bool {
	for _, p := range parts {
		if p == term {
			return true
		}
	}
	return false
}

func containsPrefix(parts []string, term string) bool {
	for _, p := range parts {
		if strings.HasPrefix(p, term) {
			return true
		}
	}
	return false
}

type readInput struct {
	Path string `json:"path"`
}

type readOutput struct {
	Content   string `json:"content"`
	Truncated bool   `json:"truncated"`
}

func makeRead(root string) func(agent.Context, readInput) (readOutput, error) {
	return func(_ agent.Context, in readInput) (readOutput, error) {
		clean := filepath.Clean("/" + in.Path) // neutralize ../ escapes
		full := filepath.Join(root, clean)
		if !strings.HasPrefix(full, root) {
			return readOutput{}, fmt.Errorf("doctools: path %q escapes corpus root", in.Path)
		}
		data, err := os.ReadFile(full)
		if err != nil {
			return readOutput{}, fmt.Errorf("doctools: read %q: %w", in.Path, err)
		}
		truncated := len(data) > maxReadBytes
		if truncated {
			data = data[:maxReadBytes]
		}
		return readOutput{Content: string(data), Truncated: truncated}, nil
	}
}
