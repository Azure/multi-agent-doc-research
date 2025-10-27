# MS Agent Framework (AFW) Services

This directory contains the Microsoft Agent Framework implementation of the document inquiry chatbot services, migrated from Semantic Kernel (`services_sk`).

## Architecture Overview

The AFW implementation uses Microsoft Agent Framework's workflow patterns with executors and agents:

```
┌─────────────────────────────────────────────────────────────┐
│                  PlanSearchOrchestrator                       │
│                  (Main Orchestrator)                        │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ├─► IntentAnalyzerExecutor
                  │   └─► Analyzes user intent & enriches query
                  │
                  ├─► SearchPlannerExecutor
                  │   └─► Creates search plan with sub-topics
                  │
                  ├─► WebSearchExecutor (Phase 1)
                  │   └─► Performs Bing search (optional)
                  │
                  ├─► AISearchExecutor (Phase 2) ✅
                  │   └─► Searches uploaded documents in Azure AI Search
                  │
                  ├─► GroundingExecutor (Phase 2) ✅
                  │   └─► Uses Bing Grounding API with Azure AI Agents
                  │
                  ├─► YouTubeMCPExecutor (Phase 2) ✅
                  │   └─► Searches YouTube videos via MCP
                  │
                  ├─► MultiAgentResearchExecutor (Phase 2) ✅
                  │   └─► Parallel writer/reviewer research workflows
                  │
                  └─► ResponseGeneratorExecutor
                      └─► Synthesizes final response from all contexts
```

## Key Components

### Core Orchestrator

- **`plan_search_orchestrator.py`**: Main orchestrator using WorkflowBuilder pattern
  - `IntentAnalyzerExecutor`: Analyzes user intent using `intent_analyze_prompt.yaml`
  - `SearchPlannerExecutor`: Generates structured search plans
  - `ResponseGeneratorExecutor`: Synthesizes final responses

### Specialized Executors

- **`search_executor_afw.py`**: Web search functionality
  - `WebSearchHelper`: Core Bing API search implementation
  - `WebSearchExecutor`: Executor for workflow integration
  - `search_web()`: AI function tool for agents

- **`group_chatting_executor.py`**: Multi-agent writer/reviewer collaboration (Phase 1)
  - `GroupChattingExecutor`: Main executor for group chat orchestration
  - `ApprovalTerminationStrategy`: Custom termination logic for approval-based completion
  - Implements turn-based writer/reviewer refinement pattern
  - Compatible with SK plugin interface

### Phase 2 Executors

- **`ai_search_executor.py`**: AI Search for uploaded documents ✅
  - `AISearchExecutor`: Azure AI Search integration with hybrid/semantic/vector search
  - Supports document filtering by type, industry, company, year
  - Embedding generation for vector search

- **`grounding_executor.py`**: Bing Grounding API integration ✅
  - `GroundingExecutor`: Uses Azure AI Agents with Bing Grounding Tool
  - Real-time web search with citations
  - Supports multiple queries in one request

- **`youtube_executor.py`**: YouTube search via MCP ✅
  - `YouTubeMCPExecutor`: YouTube Data API via Model Context Protocol
  - Video search with metadata extraction
  - Context creation for video information

- **`multi_agent_executor.py`**: Multi-agent research workflows ✅
  - `MultiAgentResearchExecutor`: Parallel writer/sequential reviewer pattern
  - Supports concurrent writer tasks with semaphore-based limiting
  - Writer agents draft answers, reviewer agents validate and refine
  - Compatible with SK plugin interface

### Reused from SK

- **`unified_file_upload_plugin.py`**: File upload handling (no migration needed)

## Usage

### In main.py

```python
from services_afw.plan_search_orchestrator import PlanSearchOrchestrator

# Select framework based on request
if request.framework == "afw":
    executor = PlanSearchOrchestrator(settings)
else:
    executor = PlanSearchExecutorSK(settings)

# Generate response
async for chunk in executor.generate_response(
    messages=messages,
    planning=True,
    stream=True,
    ...
):
    yield chunk
```

### Standalone Usage

```python
from services_afw.plan_search_orchestrator import PlanSearchOrchestrator
from config.config import Settings
from model.models import ChatMessage

# Initialize
settings = Settings()
executor = PlanSearchOrchestrator(settings)

# Prepare messages
messages = [
    ChatMessage(role="user", content="What is machine learning?")
]

# Generate response
async for response in executor.generate_response(
    messages=messages,
    planning=True,
    include_web_search=True,
    stream=False
):
    print(response)
```

### Using Group Chat for Research

```python
from services_afw.group_chatting_executor import GroupChattingExecutor
from config.config import Settings

# Initialize
settings = Settings()
group_executor = GroupChattingExecutor(settings)

# Execute writer/reviewer collaboration
result = await group_executor.group_chat(
    sub_topic="Azure OpenAI pricing models",
    question="What are the pricing options for Azure OpenAI?",
    sub_topic_contexts="Azure OpenAI offers pay-as-you-go and commitment-based pricing...",
    locale="ko-KR",
    max_rounds=5,
    max_tokens=8000
)

# Parse JSON result
import json
parsed = json.loads(result)
print(f"Status: {parsed['status']}")
print(f"Rounds: {parsed['rounds_used']}")
print(f"Answer: {parsed['final_answer']}")
```

## Key Differences from Semantic Kernel

| Aspect | Semantic Kernel (SK) | Agent Framework (AFW) |
|--------|---------------------|----------------------|
| **Core Pattern** | Kernel + Plugins | Executors + Workflows |
| **Functions** | `@kernel_function` | `@ai_function` + `@handler` |
| **Orchestration** | Kernel.invoke() | WorkflowBuilder + Executor graph |
| **Chat Client** | AzureChatCompletion | AzureOpenAIChatClient |
| **Message Flow** | Function invocation | WorkflowContext.send_message() |
| **Streaming** | Manual chunk handling | WorkflowOutputEvent streaming |

## Migration Status

✅ **Completed**:
- Base orchestrator with workflow pattern
- Intent analysis executor
- Search planning executor
- Response generation executor
- Web search executor and helper
- Framework selection in main.py
- Basic test suite

🚧 **In Progress**:
- Additional executor implementations
- Integration with existing plugins

📋 **TODO**:
- AI Search executor (for uploaded documents)
- Grounding executor
- YouTube search integration
- Multi-agent collaboration patterns
- File upload handling
- Complete test coverage

## Testing

### Run All Tests

```bash
# Using pytest
pytest app/backend/tests/test_orchestrator_afw.py -v
pytest app/backend/tests/test_search_afw.py -v

# Or run manually
python app/backend/tests/test_orchestrator_afw.py
python app/backend/tests/test_search_afw.py
```

### Test Framework Selection

```bash
# Test with AFW
curl -X POST http://localhost:8000/plan_search \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is AI?"}],
    "framework": "afw",
    "planning": true,
    "stream": false
  }'

# Test with SK (default)
curl -X POST http://localhost:8000/plan_search \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is AI?"}],
    "framework": "sk",
    "planning": true,
    "stream": false
  }'
```

## Design Principles

1. **Reuse Prompts**: All YAML prompts from `prompts/` are reused as-is
   - `IntentAnalyzerExecutor` uses `intent_analyze_prompt.yaml`
   - `SearchPlannerExecutor` uses `research_planner_prompt.yaml` or `general_planner_prompt.yaml`
   - `ResponseGeneratorExecutor` uses `general_answer_prompt.yaml`
   - `GroupChattingExecutor` uses `research_writer_prompt.yaml` and `research_reviewer_prompt.yaml`
2. **Executor Pattern**: Each major function is an Executor with `@handler` methods
3. **Workflow Composition**: Build complex flows using WorkflowBuilder
4. **Type Safety**: Use WorkflowContext type parameters for message flow
5. **Streaming Support**: Use `WorkflowOutputEvent` for streaming responses
6. **Error Handling**: Graceful fallbacks at each executor stage
7. **Metadata Passing**: Pass locale and settings via workflow metadata

## Best Practices

### Creating Executors

```python
from agent_framework import Executor, WorkflowContext, handler

class MyExecutor(Executor):
    def __init__(self, id: str, chat_client, settings):
        super().__init__(id=id)
        self._chat_client = chat_client
        self._settings = settings
    
    @handler
    async def process(
        self, 
        input_data: Dict[str, Any], 
        ctx: WorkflowContext[Dict[str, Any]]
    ) -> None:
        # Process data
        result = await self._do_work(input_data)
        
        # Send to next executor
        await ctx.send_message(result)
```

### Building Workflows

```python
from agent_framework import WorkflowBuilder

# Create executors
executor1 = Executor1(id="step1", ...)
executor2 = Executor2(id="step2", ...)
executor3 = Executor3(id="step3", ...)

# Build sequential workflow
workflow = (
    WorkflowBuilder()
    .set_start_executor(executor1)
    .add_edge(executor1, executor2)
    .add_edge(executor2, executor3)
    .build()
)

# Build parallel workflow (fan-out/fan-in)
workflow = (
    WorkflowBuilder()
    .set_start_executor(dispatcher)
    .add_fan_out_edges(dispatcher, [worker1, worker2, worker3])
    .add_fan_in_edges([worker1, worker2, worker3], aggregator)
    .build()
)
```

## References

- [Microsoft Agent Framework Documentation](https://learn.microsoft.com/en-us/agent-framework/)
- [Agent Framework GitHub](https://github.com/microsoft/agent-framework)
- [Workflow Patterns Guide](https://microsoft.github.io/agent-framework/workflows/)
- Reference notebooks in `app/backend/reference/`

## Support

For issues or questions about the AFW migration, please refer to:
1. This README
2. Reference notebooks in `../reference/`
3. Test files for usage examples
4. Microsoft Agent Framework documentation
