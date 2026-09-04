// Package indexer defines the corpus catalog format shared between
// cmd/buildindex (which generates it, out-of-band from any eval run) and
// internal/exploreagent (which loads it and inlines it into the agent's
// system prompt). The catalog is a one-line-per-doc summary of the whole
// corpus, built once so a query-time agent can see what exists across the
// entire corpus in one shot, instead of inferring corpus coverage from
// whatever a partial grep/glob search happened to surface.
package indexer

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

// Entry describes one corpus document for the catalog. Title and Summary
// are LLM-generated (see cmd/buildindex); Path is always the ground truth
// relative path, never generated.
type Entry struct {
	Path    string `json:"path"`
	Title   string `json:"title"`
	Summary string `json:"summary"`
}

// Load reads a catalog JSON file (an array of Entry). A missing file is
// not an error — callers should treat it as "no catalog available" (the
// agent falls back to pure search) rather than fail the whole run over a
// catalog that hasn't been built yet.
func Load(path string) ([]Entry, error) {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("indexer: read %q: %w", path, err)
	}
	var entries []Entry
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil, fmt.Errorf("indexer: parse %q: %w", path, err)
	}
	return entries, nil
}

// Save writes entries as a pretty-printed JSON array, sorted by path for
// a stable diff across regenerations.
func Save(path string, entries []Entry) error {
	sorted := make([]Entry, len(entries))
	copy(sorted, entries)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Path < sorted[j].Path })
	data, err := json.MarshalIndent(sorted, "", "  ")
	if err != nil {
		return fmt.Errorf("indexer: marshal: %w", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("indexer: write %q: %w", path, err)
	}
	return nil
}

// FormatCatalog renders entries as a compact "path: summary" block
// suitable for inlining directly into an agent's system prompt. Title is
// omitted from the rendered text (kept in the JSON for reference/
// debugging) since the summary is written to already convey what the
// title would — including both would mostly duplicate tokens for no
// extra signal.
func FormatCatalog(entries []Entry) string {
	sorted := make([]Entry, len(entries))
	copy(sorted, entries)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Path < sorted[j].Path })
	var sb strings.Builder
	for _, e := range sorted {
		summary := e.Summary
		if summary == "" {
			summary = e.Title
		}
		fmt.Fprintf(&sb, "%s: %s\n", e.Path, summary)
	}
	return strings.TrimSuffix(sb.String(), "\n")
}
