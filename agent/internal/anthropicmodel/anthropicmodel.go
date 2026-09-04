// Package anthropicmodel is a minimal adk-go model.LLM adapter for the
// Anthropic Messages API. adk-go (as of v2.3.0) ships native adapters for
// Gemini and OpenAI only; this fills the gap for Claude with just enough
// coverage for a text + function-calling agent (no images, no streaming
// deltas, no citations). Good enough for a small experiment; not a
// general-purpose adapter.
package anthropicmodel

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// ClientConfig configures the Anthropic client. Exactly one of APIKey or
// AuthToken should be set: APIKey for a standard sk-ant-api03-... key
// (sent as X-Api-Key), AuthToken for a Claude subscription OAuth access
// token (sent as "Authorization: Bearer ..." plus the oauth beta header —
// see NewModel for why that combination is used).
type ClientConfig struct {
	APIKey    string
	AuthToken string
	MaxTokens int64 // defaults to 4096 if unset
}

type anthropicModel struct {
	client    anthropic.Client
	name      string
	maxTokens int64
}

// NewModel constructs an adk-go model.LLM backed by Claude.
func NewModel(_ context.Context, modelName string, cfg *ClientConfig) (model.LLM, error) {
	if modelName == "" {
		return nil, fmt.Errorf("anthropicmodel: model name is required")
	}
	if cfg == nil {
		cfg = &ClientConfig{}
	}
	var opts []option.RequestOption
	switch {
	case cfg.AuthToken != "":
		// OAuth access tokens (sk-ant-oat01-...) authenticate via Bearer,
		// and require this beta header or the API rejects the token.
		opts = append(opts,
			option.WithAuthToken(cfg.AuthToken),
			option.WithHeader("anthropic-beta", "oauth-2025-04-20"),
		)
	case cfg.APIKey != "":
		opts = append(opts, option.WithAPIKey(cfg.APIKey))
	default:
		return nil, fmt.Errorf("anthropicmodel: one of APIKey or AuthToken is required")
	}
	maxTokens := cfg.MaxTokens
	if maxTokens == 0 {
		maxTokens = 4096
	}
	return &anthropicModel{
		client:    anthropic.NewClient(opts...),
		name:      modelName,
		maxTokens: maxTokens,
	}, nil
}

func (m *anthropicModel) Name() string { return m.name }

// GenerateContent always performs one blocking call, even when stream is
// requested: this experiment has no interactive UI consuming partial
// deltas, so a single non-partial LLMResponse is sufficient and much
// simpler than implementing SSE aggregation against genai's shape.
func (m *anthropicModel) GenerateContent(ctx context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if req == nil {
			yield(nil, fmt.Errorf("anthropicmodel: nil request"))
			return
		}
		params, err := buildParams(m.name, m.maxTokens, req)
		if err != nil {
			yield(nil, err)
			return
		}
		resp, err := m.client.Messages.New(ctx, params)
		if err != nil {
			yield(nil, fmt.Errorf("anthropicmodel: call failed: %w", err))
			return
		}
		llmResp, err := convertResponse(resp)
		if err != nil {
			yield(nil, err)
			return
		}
		yield(llmResp, nil)
	}
}

func buildParams(modelName string, maxTokens int64, req *model.LLMRequest) (anthropic.MessageNewParams, error) {
	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(modelName),
		MaxTokens: maxTokens,
	}

	if req.Config != nil && req.Config.SystemInstruction != nil {
		var sysText string
		for _, p := range req.Config.SystemInstruction.Parts {
			sysText += p.Text
		}
		if sysText != "" {
			params.System = []anthropic.TextBlockParam{{Text: sysText}}
		}
	}

	msgs, err := convertContents(req.Contents)
	if err != nil {
		return params, err
	}
	params.Messages = msgs

	if req.Config != nil {
		for _, t := range req.Config.Tools {
			for _, fd := range t.FunctionDeclarations {
				toolParam, err := convertToolDeclaration(fd)
				if err != nil {
					return params, err
				}
				params.Tools = append(params.Tools, anthropic.ToolUnionParam{OfTool: &toolParam})
			}
		}
	}

	return params, nil
}

func convertContents(contents []*genai.Content) ([]anthropic.MessageParam, error) {
	var out []anthropic.MessageParam
	for _, c := range contents {
		role := anthropic.MessageParamRoleUser
		if c.Role == genai.RoleModel {
			role = anthropic.MessageParamRoleAssistant
		}
		var blocks []anthropic.ContentBlockParamUnion
		for _, p := range c.Parts {
			switch {
			case p.Text != "":
				blocks = append(blocks, anthropic.NewTextBlock(p.Text))
			case p.FunctionCall != nil:
				id := p.FunctionCall.ID
				if id == "" {
					id = p.FunctionCall.Name
				}
				blocks = append(blocks, anthropic.NewToolUseBlock(id, p.FunctionCall.Args, p.FunctionCall.Name))
			case p.FunctionResponse != nil:
				id := p.FunctionResponse.ID
				if id == "" {
					id = p.FunctionResponse.Name
				}
				text, err := json.Marshal(p.FunctionResponse.Response)
				if err != nil {
					return nil, fmt.Errorf("anthropicmodel: marshal function response: %w", err)
				}
				blocks = append(blocks, anthropic.NewToolResultBlock(id, string(text), false))
			}
		}
		if len(blocks) == 0 {
			continue
		}
		out = append(out, anthropic.MessageParam{Role: role, Content: blocks})
	}
	return out, nil
}

func convertToolDeclaration(fd *genai.FunctionDeclaration) (anthropic.ToolParam, error) {
	schema := anthropic.ToolInputSchemaParam{}
	// functiontool populates ParametersJsonSchema (a *jsonschema.Schema from
	// google/jsonschema-go, reflected off the Go input struct), not the
	// genai-native Parameters field — prefer it, falling back to Parameters
	// for tools built directly against the genai types.
	switch {
	case fd.ParametersJsonSchema != nil:
		props, required, err := convertSchemaAny(fd.ParametersJsonSchema)
		if err != nil {
			return anthropic.ToolParam{}, err
		}
		schema.Properties = props
		schema.Required = required
	case fd.Parameters != nil:
		props, required, err := convertSchemaAny(fd.Parameters)
		if err != nil {
			return anthropic.ToolParam{}, err
		}
		schema.Properties = props
		schema.Required = required
	}
	return anthropic.ToolParam{
		Name:        fd.Name,
		Description: anthropic.String(fd.Description),
		InputSchema: schema,
	}, nil
}

// convertSchemaAny converts a JSON-schema-shaped value's top-level object
// properties into the map shape Anthropic's tool input_schema expects, via
// a generic JSON round-trip. Only object/string/integer/number/boolean/
// array-of-string params are needed for this experiment's tools, so this
// skips a full recursive converter.
func convertSchemaAny(s any) (map[string]any, []string, error) {
	raw, err := json.Marshal(s)
	if err != nil {
		return nil, nil, fmt.Errorf("anthropicmodel: marshal schema: %w", err)
	}
	var full map[string]any
	if err := json.Unmarshal(raw, &full); err != nil {
		return nil, nil, fmt.Errorf("anthropicmodel: unmarshal schema: %w", err)
	}
	props, _ := full["properties"].(map[string]any)
	var required []string
	if reqAny, ok := full["required"].([]any); ok {
		for _, r := range reqAny {
			if rs, ok := r.(string); ok {
				required = append(required, rs)
			}
		}
	}
	return props, required, nil
}

func convertResponse(resp *anthropic.Message) (*model.LLMResponse, error) {
	content := &genai.Content{Role: genai.RoleModel}
	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			content.Parts = append(content.Parts, genai.NewPartFromText(block.Text))
		case "tool_use":
			var args map[string]any
			if len(block.Input) > 0 {
				if err := json.Unmarshal(block.Input, &args); err != nil {
					return nil, fmt.Errorf("anthropicmodel: unmarshal tool_use input: %w", err)
				}
			}
			content.Parts = append(content.Parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{ID: block.ID, Name: block.Name, Args: args},
			})
		}
	}

	llmResp := &model.LLMResponse{
		Content:      content,
		TurnComplete: true,
		FinishReason: convertFinishReason(resp.StopReason),
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     int32(resp.Usage.InputTokens),
			CandidatesTokenCount: int32(resp.Usage.OutputTokens),
		},
		CustomMetadata: map[string]any{
			"anthropic_message_id": resp.ID,
			"anthropic_model":      string(resp.Model),
		},
	}
	return llmResp, nil
}

func convertFinishReason(r anthropic.StopReason) genai.FinishReason {
	switch r {
	case anthropic.StopReasonEndTurn, anthropic.StopReasonToolUse, anthropic.StopReasonStopSequence:
		return genai.FinishReasonStop
	case anthropic.StopReasonMaxTokens:
		return genai.FinishReasonMaxTokens
	default:
		return genai.FinishReasonUnspecified
	}
}
