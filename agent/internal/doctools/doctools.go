// Package doctools implements the three tools the exploration agent uses
// to find documents: list (glob-ish filename search), grep (content
// search), and read (fetch one file). All three are sandboxed to a single
// root directory so the agent can only ever see the corpus, mirroring how
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

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

const (
	maxListResults = 200
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

	listTool, err := functiontool.New(functiontool.Config{
		Name: "list_docs",
		Description: "List markdown file paths in the corpus whose path contains the given substring " +
			"(case-insensitive). Use an empty substring to list a sample of the whole corpus. " +
			"Returns paths relative to the corpus root, e.g. \"runtime/fundamentals/permissions.md\".",
	}, makeList(root))
	if err != nil {
		return nil, err
	}

	grepTool, err := functiontool.New(functiontool.Config{
		Name: "grep_docs",
		Description: "Search the text content of every markdown file in the corpus for a regular expression " +
			"(case-insensitive). Returns matching file paths with the matching line and a snippet of " +
			"surrounding context. Use this to find which doc actually discusses a term, error message, " +
			"or concept, not just which filename mentions it.",
	}, makeGrep(root))
	if err != nil {
		return nil, err
	}

	readTool, err := functiontool.New(functiontool.Config{
		Name:        "read_doc",
		Description: "Read the full text content of one markdown file, given its path relative to the corpus root (as returned by list_docs or grep_docs).",
	}, makeRead(root))
	if err != nil {
		return nil, err
	}

	return []tool.Tool{listTool, grepTool, readTool}, nil
}

type listInput struct {
	Substring string `json:"substring"`
}

type listOutput struct {
	Paths     []string `json:"paths"`
	Total     int      `json:"total_matching"`
	Truncated bool     `json:"truncated"`
}

func makeList(root string) func(agent.Context, listInput) (listOutput, error) {
	return func(_ agent.Context, in listInput) (listOutput, error) {
		needle := strings.ToLower(in.Substring)
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
			if needle == "" || strings.Contains(strings.ToLower(rel), needle) {
				matches = append(matches, rel)
			}
			return nil
		})
		if err != nil {
			return listOutput{}, fmt.Errorf("doctools: walk: %w", err)
		}
		sort.Strings(matches)
		total := len(matches)
		truncated := total > maxListResults
		if truncated {
			matches = matches[:maxListResults]
		}
		return listOutput{Paths: matches, Total: total, Truncated: truncated}, nil
	}
}

type grepInput struct {
	Pattern string `json:"pattern"`
}

type grepMatch struct {
	File    string `json:"file"`
	Line    int    `json:"line"`
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
			for i, line := range strings.Split(string(data), "\n") {
				if re.MatchString(line) {
					out.Matches = append(out.Matches, grepMatch{
						File:    rel,
						Line:    i + 1,
						Snippet: strings.TrimSpace(line),
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
