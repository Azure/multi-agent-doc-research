# MagenticExecutor Documentation

## Overview

`MagenticExecutor` implements the **Magentic orchestration pattern** from Microsoft Agent Framework for intelligent multi-agent research collaboration. Unlike the sequential turn-based approach in `GroupChattingExecutor`, Magentic uses an intelligent orchestrator that dynamically plans, coordinates, and manages specialized agents to accomplish complex research tasks.

## Architecture (3-Agent Pattern)

```
┌──────────────────────────────────────────────────────────┐
│            Magentic Orchestrator                         │
│     (Intelligent Planning, Coordination & QA)            │
└─────────┬──────────────┬──────────────┬─────────────────┘
          │              │              │
  ┌───────▼────────┐ ┌──▼─────────┐ ┌─▼──────────────┐
  │ResearchAnalyst │ │ResearchWr..│ │ResearchReviewer│
  │(Synthesize)    │ │(Generate)  │ │(Validate)      │
  │• Extract info  │ │• Structure │ │• Check quality │
  │• Find patterns │ │• Format    │ │• Score (1-5)   │
  │• Cite sources  │ │• Cite URLs │ │• Approve/Revise│
  └────────────────┘ └────────────┘ └────────────────┘
```

### Key Components (3-Agent Pattern)

1. **Orchestrator**: Intelligent manager that plans task execution and coordinates agents
2. **ResearchAnalyst**: Specializes in information synthesis, pattern recognition, and source extraction
3. **ResearchWriter**: Focuses on creating well-structured, comprehensive research content with proper citations
4. **ResearchReviewer**: Validates quality, accuracy, completeness, and citation integrity before final output

## Comparison with GroupChattingExecutor

| Aspect | GroupChattingExecutor | MagenticExecutor |
|--------|----------------------|------------------|
| **Pattern** | Sequential turn-based dialogue | Intelligent orchestration |
| **Coordination** | Fixed Writer → Reviewer loop | Dynamic planning by orchestrator |
| **Agent Roles** | Writer + Reviewer (2 agents) | Researcher + Writer + Reviewer (3 agents) |
| **Planning** | None (predetermined flow) | Built-in intelligent planning |
| **Adaptability** | Low (fixed rounds) | High (adaptive based on task) |
| **Execution** | Turn-taking with approval | Orchestrated collaboration |
| **Quality Assurance** | Approval-based | Score-based (1-5 scale) |
| **Best For** | Iterative refinement | Complex research with QA |
| **Token Usage** | Medium | High (due to orchestration + 3 agents) |
| **Speed** | Fast (3-5 rounds) | Medium (up to 7 adaptive rounds) |

## Features

### ✅ Intelligent Orchestration
- Orchestrator dynamically plans agent collaboration
- Adaptive execution based on task complexity
- Automatic task decomposition and delegation

### ✅ Specialized Agents (3-Agent Pattern)
- **ResearchAnalyst**: Information synthesis, pattern identification, insight extraction, source tracking
- **ResearchWriter**: Content generation, structuring, comprehensive reporting with citations
- **ResearchReviewer**: Quality validation, accuracy checking, citation verification, scoring (1-5)

### ✅ Streaming Support
- Real-time progress updates
- Token-by-token streaming for agent responses
- Orchestrator planning visibility
- TTFT (Time To First Token) tracking

### ✅ Robust Error Handling
- Graceful degradation on failures
- Detailed error reporting
- Fallback to raw output if JSON parsing fails

### ✅ Context Management
- Automatic context truncation (15,000 chars per sub-topic)
- Efficient memory usage
- Multi-source context integration (Web, AI Search, YouTube)

## Usage

### Basic Integration

```python
from services_afw.magentic_executor import MagenticExecutor
from agent_framework.azure import AzureOpenAIChatClient
from config.config import Settings

# Initialize chat client
chat_client = AzureOpenAIChatClient(
    deployment_name="gpt-4o",
    endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-10-21"
)

# Create MagenticExecutor
magentic_executor = MagenticExecutor(
    id="magentic_research",
    chat_client=chat_client,
    settings=Settings(),
    context_max_chars=400000,
    max_document_length=10000,
    writer_parallel_limit=4
)
```

### In PlanSearchOrchestratorAFW

```python
# Set multi_agent_type to "afw_magentic"
await orchestrator.generate_response(
    messages=messages,
    research=True,
    multi_agent_type="afw_magentic",  # ✅ Use Magentic pattern
    locale="en-US",
    stream=True
)
```

## Input Format

The executor expects research data in the following format:

```python
research_data = {
    "question": "Main research question",
    "enriched_query": "Enhanced query with context",
    "original_query": "User's original question",
    "sub_topics": [
        {
            "sub_topic": "Topic name",
            "queries": ["query1", "query2"]
        }
    ],
    "sub_topic_web_contexts": {
        "Topic name": [
            {
                "results": [
                    {
                        "title": "...",
                        "snippet": "...",
                        "url": "..."
                    }
                ]
            }
        ]
    },
    "sub_topic_youtube_contexts": {
        "Topic name": {
            "videos": [
                {
                    "title": "...",
                    "description": "...",
                    "url": "..."
                }
            ]
        }
    },
    "sub_topic_ai_search_contexts": {
        "Topic name": {
            "documents": [
                {
                    "title": "...",
                    "summary": "...",
                    "content": "...",
                    "file_name": "...",
                    "page_number": "..."
                }
            ]
        }
    },
    "metadata": {
        "locale": "en-US",
        "verbose": False,
        "max_tokens": 8000
    }
}
```

## Output Format

The executor yields streaming output and returns results per sub-topic:

```python
{
    "status": "success" | "error",
    "sub_topic": "Topic name",
    "question": "Research question",
    "final_answer": "Markdown formatted answer",
    "citations": [
        {
            "source": "Source name/URL",
            "relevance": "Why this source is relevant"
        }
    ],
    "orchestration_rounds": 5,  # Number of orchestration rounds
    "orchestrator_messages": 3,  # Number of orchestrator messages
    "agent_responses": 8,        # Number of agent responses
    "error": None | "Error message"
}
```

## Configuration Parameters

### Constructor Parameters

- **id** (str): Unique identifier for the executor
- **chat_client** (AzureOpenAIChatClient): Azure OpenAI chat client for agents
- **settings** (Settings): Application settings
- **context_max_chars** (int): Maximum context size (default: 400,000)
- **max_document_length** (int): Maximum length per document (default: 10,000)
- **writer_parallel_limit** (int): Parallel processing limit (default: 4)

### Orchestrator Parameters

The Magentic workflow is configured with:
- **max_round_count**: 5 (maximum orchestration rounds)
- **max_stall_count**: 2 (stop if no progress)
- **max_reset_count**: 1 (allow one reset if stuck)

## Streaming Events

The executor emits various streaming events:

### Progress Events
```
data: ### 🎯 Magentic orchestration for 'Topic' [1/3]
data: 🎯 [Orchestrator] planning
data: ✅ [ResearchAnalyst] completed response
data: ✅ [ResearchWriter] completed response
data: 🎉 Magentic orchestration complete for 'Topic'
data: ### ✅ Completed 'Topic' [1/3]
```

### TTFT Marker
```
data: __TTFT_MARKER__
```
Emitted when the first token is generated for performance tracking.

### Final Output
```markdown
## Topic Name

[Comprehensive research content in markdown format]

- Key finding 1
- Key finding 2
...
```

## Best Practices

### 1. When to Use Magentic

✅ **Use MagenticExecutor when:**
- Research tasks require multi-step analysis
- Complex task decomposition is needed
- Dynamic agent coordination is beneficial
- You want adaptive, intelligent planning
- Quality is more important than speed

❌ **Use GroupChattingExecutor when:**
- Simple iterative refinement is sufficient
- Speed is critical
- Fixed writer-reviewer flow works well
- You want lower token consumption

### 2. Context Management

```python
# Provide rich, structured context
research_data = {
    "sub_topic_web_contexts": {...},      # Web search results
    "sub_topic_ai_search_contexts": {...}, # Knowledge base
    "sub_topic_youtube_contexts": {...}    # Video content
}
```

The executor automatically:
- Combines contexts from all sources
- Truncates to fit within limits
- Formats for optimal agent understanding

### 3. Error Handling

The executor handles errors gracefully:
- JSON parsing failures → fallback to raw text
- Agent errors → logged with detailed trace
- Orchestration failures → status marked as "error"

### 4. Performance Optimization

```python
# Adjust parameters for your use case
magentic_executor = MagenticExecutor(
    id="magentic_research",
    chat_client=chat_client,
    settings=settings,
    context_max_chars=400000,  # Increase for larger contexts
    max_document_length=10000,  # Adjust based on document size
    writer_parallel_limit=4     # Parallel processing capacity
)
```

## Implementation Details

### Magentic Workflow Construction

```python
workflow = (
    MagenticBuilder()
    .participants(
        researcher=researcher_agent,  # ResearchAnalyst
        writer=writer_agent            # ResearchWriter
    )
    .on_event(on_magentic_event, mode=MagenticCallbackMode.STREAMING)
    .with_standard_manager(
        chat_client=self.chat_client,
        max_round_count=5,
        max_stall_count=2,
        max_reset_count=1
    )
    .build()
)
```

### Agent Instructions

Both agents receive:
- Current date and locale
- Research topic and question
- Available context from all sources
- Clear role definitions and guidelines

### Callback System

```python
async def on_magentic_event(event: MagenticCallbackEvent):
    # Handle different event types:
    # - MagenticOrchestratorMessageEvent (planning)
    # - MagenticAgentDeltaEvent (streaming tokens)
    # - MagenticAgentMessageEvent (complete responses)
    # - MagenticFinalResultEvent (final output)
```

## Troubleshooting

### Common Issues

**1. "No output received from Magentic workflow"**
- Check agent instructions are clear
- Verify context is not empty
- Increase max_round_count if task is complex

**2. JSON parsing failures**
- Executor automatically falls back to raw text
- Check WriterAgent instructions for JSON format
- Review agent responses in logs

**3. Slow execution**
- Magentic pattern involves orchestration overhead
- Consider using GroupChattingExecutor for simpler tasks
- Monitor orchestration_rounds in output

**4. High token usage**
- Orchestrator adds planning overhead
- Reduce context_max_chars if needed
- Consider fewer sub-topics per request

## Logging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("services_afw.magentic_executor")
```

Log output includes:
- Orchestration progress
- Agent responses
- Context statistics
- Error traces

## Future Enhancements

Potential improvements:
- [ ] Support for more specialized agents (fact-checker, summarizer)
- [ ] Configurable orchestration strategies
- [ ] Parallel sub-topic processing
- [ ] Advanced caching for repeated queries
- [ ] Custom termination conditions
- [ ] Agent performance metrics

## References

- [Microsoft Agent Framework Documentation](https://learn.microsoft.com/en-us/agent-framework/)
- [Magentic Orchestration Pattern](https://github.com/microsoft/agent-framework/tree/main/python/samples)
- [AutoGen Multi-Agent Research](https://arxiv.org/abs/2308.08155)

## License

Part of doc-inquiry-chatbot project.
