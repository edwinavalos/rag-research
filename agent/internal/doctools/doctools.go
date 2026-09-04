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
			"(case-insensitive). Returns matching file paths with the matching line and a snippet of " +
			"surrounding context. Use this to find which doc actually discusses a term, error message, " +
			"or concept, not just which filename mentions it.",
	}, makeGrep(root))
	if err != nil {
		return nil, err
	}

	readTool, err := functiontool.New(functiontool.Config{
		Name:        "read_doc",
		Description: "Read the full text content of one markdown file, given its path relative to the corpus root (as returned by glob_docs or grep_docs).",
	}, makeRead(root))
	if err != nil {
		return nil, err
	}

	return []tool.Tool{globTool, grepTool, readTool}, nil
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
