//go:build poc

// Copyright (c) 2023-present Mattermost, Inc. All Rights Reserved.
// See LICENSE.txt for license information.

// Safe, local reproduction of a same-request MCP ABAC gap.
//
// Reproduce:
//
//	go test -tags poc -race -count=2 -v ./api/ -run TestPoC_MCPABACPreDispatchRevocation
//
// This test expects the current behavior: a tool resolver built while the user
// is allowed still calls the MCP server after the simulated PDP has switched
// to DENY, as long as that switch happens before the model returns the tool
// call. It does not change production code. The protected record is synthetic
// and exists only inside the in-process recording MCP server.
//
// A passing run confirms the gap. After a fix, pre_dispatch_deny_still_executes
// should fail because the MCP server must not be called.

package api

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/mattermost/mattermost-plugin-agents/v2/llm"
	llmcontext "github.com/mattermost/mattermost-plugin-agents/v2/llmcontext"
	"github.com/mattermost/mattermost-plugin-agents/v2/mcp"
	"github.com/mattermost/mattermost-plugin-agents/v2/public/bridgeclient"
	gomcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

const (
	pocUserA     = "pocusera1234567890123456ab" // 26-char Mattermost user id
	pocMCPID     = "pocmcpid1234567890123456ab"
	pocToolName  = "read_protected_record"
	pocCanary    = "MM-ABAC-TOCTOU-CANARY-9B41"
	pocProtected = "protected-record:" + pocCanary
)

type pocEvent struct {
	Name string
	At   time.Duration
	PDP  string
	Note string
}

type pocTimeline struct {
	mu     sync.Mutex
	start  time.Time
	events []pocEvent
}

func newPoCTimeline() *pocTimeline {
	return &pocTimeline{start: time.Now()}
}

func (l *pocTimeline) add(name, pdp, note string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.events = append(l.events, pocEvent{Name: name, At: time.Since(l.start), PDP: pdp, Note: note})
}

func (l *pocTimeline) find(t *testing.T, name string) pocEvent {
	t.Helper()
	l.mu.Lock()
	defer l.mu.Unlock()
	for _, event := range l.events {
		if event.Name == name {
			return event
		}
	}
	t.Fatalf("missing timeline event %s; events=%+v", name, l.events)
	return pocEvent{}
}

func (l *pocTimeline) log(t *testing.T) {
	t.Helper()
	l.mu.Lock()
	defer l.mu.Unlock()
	for _, event := range l.events {
		t.Logf("%-28s %10s  pdp=%-5s  %s", event.Name, event.At, event.PDP, event.Note)
	}
}

type pocPDP struct {
	mu     sync.Mutex
	denied map[string]bool
	tl     *pocTimeline
}

func (p *pocPDP) CanUseMCPServer(_ context.Context, userID, serverID string) error {
	p.mu.Lock()
	denied := p.denied[userID]
	p.mu.Unlock()
	decision := "ALLOW"
	if denied {
		decision = "DENY"
	}
	if p.tl != nil {
		p.tl.add("CanUseMCPServer", decision, "user="+userID+" server="+serverID)
	}
	if denied {
		return fmt.Errorf("mcp denied for %s", userID)
	}
	return nil
}

func (p *pocPDP) deny(userID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.denied == nil {
		p.denied = map[string]bool{}
	}
	p.denied[userID] = true
	if p.tl != nil {
		p.tl.add("PDP_DENY", "DENY", "user="+userID+" server="+pocMCPID)
	}
}

func (p *pocPDP) allowed(userID string) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return !p.denied[userID]
}

type pocMCP struct {
	server *httptest.Server
	execs  atomic.Int32
	bodies []string
	mu     sync.Mutex
	tl     *pocTimeline
}

func startPoCMCP(t *testing.T, tl *pocTimeline) *pocMCP {
	t.Helper()
	rec := &pocMCP{tl: tl}
	server := gomcp.NewServer(&gomcp.Implementation{Name: "poc-mcp", Version: "1.0"}, nil)
	gomcp.AddTool(server, &gomcp.Tool{
		Name:        pocToolName,
		Description: "read a synthetic protected record",
	}, func(_ context.Context, _ *gomcp.CallToolRequest, _ struct{}) (*gomcp.CallToolResult, struct{}, error) {
		if rec.tl != nil {
			rec.tl.add("MCP_RECEIVED", "", "tools/call "+pocToolName)
			rec.tl.add("MCP_EXECUTED", "", pocProtected)
		}
		rec.execs.Add(1)
		rec.mu.Lock()
		rec.bodies = append(rec.bodies, pocProtected)
		rec.mu.Unlock()
		return &gomcp.CallToolResult{
			Content: []gomcp.Content{&gomcp.TextContent{Text: pocProtected}},
		}, struct{}{}, nil
	})
	rec.server = httptest.NewServer(gomcp.NewStreamableHTTPHandler(func(*http.Request) *gomcp.Server {
		return server
	}, nil))
	t.Cleanup(rec.server.Close)
	return rec
}

func (r *pocMCP) results() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]string, len(r.bodies))
	copy(out, r.bodies)
	return out
}

func wirePoCMCP(t *testing.T, e *TestEnvironment, origin string, checker mcp.ServerAccessChecker, httpClient *http.Client) {
	t.Helper()
	manager := mcp.NewClientManager(
		mcp.Config{Servers: []mcp.ServerConfig{{
			ID: pocMCPID, Name: "PoC MCP", Enabled: true, BaseURL: origin,
		}}},
		e.client.Log,
		e.client,
		nil,
		nil,
		httpClient,
		nil,
		mcp.RemoteMCPAlwaysAllowed,
		checker,
	)
	t.Cleanup(manager.Close)
	e.api.contextBuilder = llmcontext.NewLLMContextBuilder(
		e.client,
		&testLLMContextToolProvider{},
		manager,
		&testLLMContextConfigProvider{},
	)
}

func findPoCTool(tools []bridgeclient.BridgeToolInfo) (bridgeclient.BridgeToolInfo, bool) {
	for _, tool := range tools {
		if tool.BareName == pocToolName || strings.HasSuffix(tool.Name, "__"+pocToolName) {
			return tool, true
		}
	}
	return bridgeclient.BridgeToolInfo{}, false
}

// pocBarrierLLM emits one tool call on the first completion, but only after
// beforeReturn runs. That callback is the synchronization point for the PDP flip.
type pocBarrierLLM struct {
	mu           sync.Mutex
	requests     []llm.CompletionRequest
	calls        int
	toolName     string
	finalText    string
	beforeReturn func()
	tl           *pocTimeline
}

func (f *pocBarrierLLM) ChatCompletion(_ context.Context, conversation llm.CompletionRequest, _ ...llm.LanguageModelOption) (*llm.TextStreamResult, error) {
	f.mu.Lock()
	f.requests = append(f.requests, conversation)
	idx := f.calls
	f.calls++
	f.mu.Unlock()

	if idx == 0 && f.tl != nil {
		n := 0
		if conversation.Context != nil && conversation.Context.Tools != nil {
			n = len(conversation.Context.Tools.GetTools())
		}
		f.tl.add("MODEL_HAS_SCHEMA", "", fmt.Sprintf("tools=%d", n))
	}
	if idx == 0 && f.beforeReturn != nil {
		f.beforeReturn()
	}

	stream := make(chan llm.TextStreamEvent)
	go func() {
		defer close(stream)
		if idx == 0 {
			if f.tl != nil {
				f.tl.add("MODEL_RETURNS_TOOL_CALL", "", f.toolName)
			}
			stream <- llm.TextStreamEvent{
				Type: llm.EventTypeToolCalls,
				Value: []llm.ToolCall{{
					ID:        "tc-poc",
					Name:      f.toolName,
					Arguments: json.RawMessage(`{}`),
				}},
			}
			stream <- llm.TextStreamEvent{Type: llm.EventTypeEnd}
			return
		}
		if f.tl != nil {
			f.tl.add("TOOL_RESULT_IN_NEXT_TURN", "", "second model round")
		}
		stream <- llm.TextStreamEvent{Type: llm.EventTypeText, Value: f.finalText}
		stream <- llm.TextStreamEvent{Type: llm.EventTypeEnd}
	}()
	return &llm.TextStreamResult{Stream: stream}, nil
}

func (f *pocBarrierLLM) ChatCompletionNoStream(context.Context, llm.CompletionRequest, ...llm.LanguageModelOption) (string, error) {
	return f.finalText, nil
}

func (f *pocBarrierLLM) CountTokens(context.Context, llm.CompletionRequest, ...llm.LanguageModelOption) (int, error) {
	return 0, llm.ErrUnsupportedTokenCount
}

func (f *pocBarrierLLM) InputTokenLimit() int  { return 100000 }
func (f *pocBarrierLLM) OutputTokenLimit() int { return 8192 }

func (f *pocBarrierLLM) toolResults() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []string
	for _, req := range f.requests {
		for _, post := range req.Posts {
			for _, call := range post.ToolUse {
				if call.Result != "" {
					out = append(out, call.Result)
				}
			}
		}
	}
	return out
}

func setupPoC(t *testing.T, tl *pocTimeline) (*TestEnvironment, *pocPDP, *pocMCP, *bridgeclient.Client) {
	t.Helper()
	gin.SetMode(gin.ReleaseMode)
	gin.DefaultWriter = io.Discard

	pdp := &pocPDP{denied: map[string]bool{}, tl: tl}
	rec := startPoCMCP(t, tl)

	e := SetupTestEnvironment(t)
	t.Cleanup(func() { e.Cleanup(t) })
	e.mockAPI.On("KVGet", mock.AnythingOfType("string")).Return([]byte(nil), nil).Maybe()
	e.mockAPI.On("KVSet", mock.AnythingOfType("string"), mock.Anything).Return(nil).Maybe()
	e.mockAPI.On("KVDelete", mock.AnythingOfType("string")).Return(nil).Maybe()
	e.mockAPI.On("KVSetWithOptions", mock.AnythingOfType("string"), mock.Anything, mock.Anything).Return(true, nil).Maybe()

	wirePoCMCP(t, e, rec.server.URL, pdp, rec.server.Client())
	e.setupTestBot(llm.BotConfig{
		Name:                  "testbot",
		DisplayName:           "Test Bot",
		UserAccessLevel:       llm.UserAccessLevelAll,
		AutoEnableNewMCPTools: true,
	})
	return e, pdp, rec, e.CreateBridgeClient()
}

func completeAs(client *bridgeclient.Client, userID, toolName, message string) (string, error) {
	return client.AgentCompletion(testBotUserID, bridgeclient.CompletionRequest{
		Posts:        []bridgeclient.Post{{Role: "user", Message: message}},
		AllowedTools: []string{toolName},
		UserID:       userID,
	})
}

// TestPoC_MCPABACPreDispatchRevocation is the reproducible confirmation.
//
// pre_dispatch_deny_still_executes is the vulnerability: DENY is applied after
// the catalog and resolver exist, and before the model returns the tool call.
// The production tool runner then invokes that resolver. The recording MCP
// server receives tools/call and returns the synthetic protected record.
func TestPoC_MCPABACPreDispatchRevocation(t *testing.T) {
	t.Run("pre_dispatch_deny_still_executes", func(t *testing.T) {
		tl := newPoCTimeline()
		env, pdp, rec, client := setupPoC(t, tl)

		tl.add("INITIAL_ALLOW", "ALLOW", "user="+pocUserA)
		require.True(t, pdp.allowed(pocUserA))

		tools, err := client.GetAgentTools(testBotUserID, pocUserA)
		require.NoError(t, err)
		info, ok := findPoCTool(tools)
		require.True(t, ok, "tool must be listed while the PDP allows the user")
		tl.add("CATALOG_LOADED", "ALLOW", info.Name)

		fake := &pocBarrierLLM{
			toolName:  info.Name,
			finalText: "turn-finished",
			tl:        tl,
			beforeReturn: func() {
				pdp.deny(pocUserA)
				require.False(t, pdp.allowed(pocUserA))
				fresh, freshErr := client.GetAgentTools(testBotUserID, pocUserA)
				require.NoError(t, freshErr)
				_, stillThere := findPoCTool(fresh)
				require.False(t, stillThere, "a fresh catalog must omit the tool after DENY")
				tl.add("FRESH_CATALOG_EMPTY", "DENY", "GetAgentTools omitted "+info.Name)
			},
		}
		for _, bot := range env.bots.GetAllBots() {
			bot.SetLLMForTest(fake)
		}

		before := rec.execs.Load()
		result, err := completeAs(client, pocUserA, info.Name, "read the protected record")
		require.NoError(t, err)
		tl.add("COMPLETION_RETURNED", "DENY", result)
		tl.log(t)

		deny := tl.find(t, "PDP_DENY")
		returned := tl.find(t, "MODEL_RETURNS_TOOL_CALL")
		received := tl.find(t, "MCP_RECEIVED")
		require.Less(t, deny.At, returned.At, "PDP DENY must happen before the model returns the tool call")
		require.Less(t, returned.At, received.At, "the tool call must be returned before the MCP server is contacted")

		require.Equal(t, before+1, rec.execs.Load(), "recording MCP server executed the tool after DENY")
		require.Equal(t, []string{pocProtected}, rec.results())
		require.Contains(t, strings.Join(fake.toolResults(), "\n"), pocProtected,
			"the production tool runner delivered the protected record into the next model round")
		require.Equal(t, "turn-finished", result)
	})

	t.Run("deny_before_catalog_does_not_execute", func(t *testing.T) {
		_, pdp, rec, client := setupPoC(t, nil)
		pdp.deny(pocUserA)
		tools, err := client.GetAgentTools(testBotUserID, pocUserA)
		require.NoError(t, err)
		_, ok := findPoCTool(tools)
		require.False(t, ok)

		before := rec.execs.Load()
		_, err = completeAs(client, pocUserA, pocToolName, "read the protected record")
		require.Error(t, err)
		require.Equal(t, before, rec.execs.Load())
	})

	t.Run("new_turn_after_revoke_does_not_execute", func(t *testing.T) {
		env, pdp, rec, client := setupPoC(t, nil)
		tools, err := client.GetAgentTools(testBotUserID, pocUserA)
		require.NoError(t, err)
		info, ok := findPoCTool(tools)
		require.True(t, ok)

		first := NewFakeLLM("first-turn")
		first.StreamEventSequence = fakeLLMAutoRunSequence("tc-first", info.Name, "first-turn")
		for _, bot := range env.bots.GetAllBots() {
			bot.SetLLMForTest(first)
		}
		_, err = completeAs(client, pocUserA, info.Name, "first turn")
		require.NoError(t, err)
		require.Equal(t, int32(1), rec.execs.Load())

		pdp.deny(pocUserA)
		second := NewFakeLLM("should-not-run")
		second.StreamEventSequence = fakeLLMAutoRunSequence("tc-second", info.Name, "should-not-run")
		for _, bot := range env.bots.GetAllBots() {
			bot.SetLLMForTest(second)
		}
		before := rec.execs.Load()
		_, err = completeAs(client, pocUserA, info.Name, "second turn after revoke")
		require.Error(t, err)
		require.Equal(t, before, rec.execs.Load())
		require.Empty(t, second.AllRequests)
	})
}
