// Package llmprovider centralizes model.LLM construction so every command
// in this experiment (cmd/evalrun, cmd/buildindex) selects providers and
// default models the same way.
package llmprovider

import (
	"context"
	"fmt"
	"os"
	"time"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/openaimodel"

	"raggraph/internal/anthropicmodel"
)

// Build constructs the model.LLM for the chosen provider. "openai" uses
// adk-go's native openaimodel adapter directly; "anthropic" uses the
// hand-rolled adapter in internal/anthropicmodel (adk-go ships no
// Anthropic adapter as of v2.3.0). requestDelay is anthropic-only (see
// anthropicmodel.ClientConfig.RequestDelay); ignored for openai.
func Build(ctx context.Context, provider, modelName string, requestDelay time.Duration) (model.LLM, error) {
	switch provider {
	case "openai":
		apiKey := os.Getenv("OPENAI_API_KEY")
		if apiKey == "" {
			return nil, fmt.Errorf("set OPENAI_API_KEY")
		}
		if modelName == "" {
			// gpt-5.4-nano is the smallest/cheapest model in the current
			// gpt-5.4 line as of this experiment (see `curl .../v1/models`).
			modelName = "gpt-5.4-nano"
		}
		return openaimodel.NewModel(ctx, modelName, &openaimodel.ClientConfig{APIKey: apiKey})
	case "anthropic":
		apiKey := os.Getenv("ANTHROPIC_API_KEY")
		authToken := os.Getenv("ANTHROPIC_AUTH_TOKEN")
		if apiKey == "" && authToken == "" {
			return nil, fmt.Errorf("set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN")
		}
		if modelName == "" {
			modelName = "claude-sonnet-5"
		}
		return anthropicmodel.NewModel(ctx, modelName, &anthropicmodel.ClientConfig{
			APIKey:       apiKey,
			AuthToken:    authToken,
			RequestDelay: requestDelay,
		})
	default:
		return nil, fmt.Errorf("unknown provider %q (want \"openai\" or \"anthropic\")", provider)
	}
}
