# Python Coding Conventions

## Python Instructions

- Write clear and concise comments for each function.
- Ensure functions have descriptive names and include type hints.
- Provide docstrings following PEP 257 conventions.
- Use the `typing` module for type annotations (e.g., `List[str]`, `Dict[str, int]`).
- Break down complex functions into smaller, more manageable functions.

## General Instructions

- Always prioritize readability and clarity.
- For algorithm-related code, include explanations of the approach used.
- Write code with good maintainability practices, including comments on why certain design decisions were made.
- Handle edge cases and write clear exception handling.
- For libraries or external dependencies, mention their usage and purpose in comments.
- Use consistent naming conventions and follow language-specific best practices.
- Write concise, efficient, and idiomatic code that is also easily understandable.

## Code Style and Formatting

- Follow the **PEP 8** style guide for Python.
- Maintain proper indentation (use 4 spaces for each level of indentation).
- Ensure lines do not exceed 79 characters.
- Place function and class docstrings immediately after the `def` or `class` keyword.
- Use blank lines to separate functions, classes, and code blocks where appropriate.

## Edge Cases and Testing

- Always include test cases for critical paths of the application.
- Account for common edge cases like empty inputs, invalid data types, and large datasets.
- Include comments for edge cases and the expected behavior in those cases.
- Write unit tests for functions and document them with docstrings explaining the test cases.

## Example of Proper Documentation

```python
def calculate_area(radius: float) -> float:
    """
    Calculate the area of a circle given the radius.
    
    Parameters:
    radius (float): The radius of the circle.
    
    Returns:
    float: The area of the circle, calculated as π * radius^2.
    """
    import math
    return math.pi * radius ** 2
```

# Multi-Agent Doc Research - AI Agent Instructions

## Architecture Overview

This is a **dual-orchestration multi-agent research system** that uses both **Microsoft Agent Framework (AFW)** and **Semantic Kernel (SK)** to perform deep document research with specialized AI agents.

### Core Workflow Pattern
```
User Query → Intent Analysis → Task Planning → Multi-Source Search → Multi-Agent Collaboration → Response
                                                 ├─ Web Search (Bing)
                                                 ├─ AI Search (Azure AI Search)
                                                 └─ YouTube Search (MCP Server)
```

## Dual Orchestration Architectures

### 1. Agent Framework (`services_afw/`) - Production Pattern
- **Pattern**: Executor-based workflow with `@handler` methods
- **Key Classes**: `IntentAnalyzerExecutor`, `TaskPlannerExecutor`, `ResponseGeneratorExecutor`
- **Agent Patterns**:
  - `GroupChattingExecutor`: Writer ↔ Reviewer approval loop (3-5 rounds)
  - `MagenticExecutor`: Intelligent orchestration with dynamic agent coordination (Orchestrator → ResearchAnalyst/Writer/Reviewer)
- **Workflow**: Uses `WorkflowBuilder` with `ctx.send_message()` for inter-executor communication
- **Entry**: `PlanSearchOrchestratorAFW.generate_response()` in `doc_research_orchestrator_afw.py`

### 2. Semantic Kernel (`services_sk/`) - Alternative Implementation
- **Pattern**: Plugin-based with `@kernel_function` decorators
- **Key Classes**: `IntentPlanPlugin`, `GroupChattingPlugin`, `MagenticPlugin`
- **Integration**: Uses `Kernel` object with `AzureChatCompletion` service
- **Entry**: `PlanSearchOrchestratorSK.generate_response()` in `doc_research_orchestrator_sk.py`

**Critical**: Both orchestrators provide identical interfaces but different implementation strategies. AFW is event-driven; SK is function-calling based.

## Project Structure Conventions

### Backend Organization (`app/backend/`)
```
services_afw/          # Agent Framework executors (preferred)
services_sk/           # Semantic Kernel plugins (alternative)
prompts/              # YAML prompt templates (shared by both)
config/config.py      # Pydantic Settings with .env loading
model/models.py       # Pydantic models for API contracts
utils/                # Shared utilities (enum.py, json_control.py, yield_message.py)
tests/                # pytest-based tests
evals/                # Azure AI Evaluation with batch_eval.py
```

### Prompt Management
- **All prompts are YAML files** in `prompts/` directory
- Load using: `load_prompt(path, encoding="utf-8")`
- Naming: `{purpose}_{role}_prompt.yaml` (e.g., `research_writer_prompt.yaml`)
- Bilingual support: `_kr.yaml` suffix for Korean variants

### Frontend (`app/frontend/`)
- **Framework**: Chainlit 2.8.0 (not Gradio/Streamlit)
- **Entry**: `src/app.py` 
- **Pattern**: Async event handlers with `@cl.on_message`, `@cl.on_chat_start`
- **API Communication**: Streaming via `aiohttp` with SSE to backend FastAPI

## Critical Development Patterns

### Adding New Executors/Plugins
1. **AFW Executor**: Inherit from `Executor`, add `@handler` methods, use `ctx.send_message()` for workflow
2. **SK Plugin**: Add `@kernel_function` methods, use `KernelArguments` for parameters
3. **Both must**: Accept `settings: Settings`, implement async streaming, handle errors gracefully

### Multi-Agent Pattern Selection
- **Use GroupChat** when: Fixed writer-reviewer workflow, speed prioritized, 💰 medium tokens
- **Use Magentic** when: Complex multi-step reasoning needed, quality over speed, 💰💰 high tokens
- **Configuration**: Via `multi_agent_type` parameter in `PlanSearchRequest`
  - `"afw_group_chat"` → `GroupChattingExecutor`
  - `"afw_magentic"` → `MagenticExecutor`
  - `"sk_group_chat"` → SK equivalent
  - `"vanilla"` → Direct AOAI SDK without agents

### Streaming Response Protocol
All orchestrators yield base64-encoded messages:
```python
yield f"### {step_name}#input#{description}"              # Progress update
yield f"### {step_name}#code#{base64_encoded_content}"    # Code/markdown content
yield "actual_token_content"                               # Stream tokens
```
Frontend decodes with `decode_step_content()` in `app_utils.py`.

### Configuration Management
- **Never hardcode**: Load from `Settings` (backed by `.env`)
- **Azure OpenAI**: Multiple deployments for different tasks
  - `AZURE_OPENAI_DEPLOYMENT_NAME` → Main chat
  - `AZURE_OPENAI_QUERY_DEPLOYMENT_NAME` → Query rewriting
  - `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` → Document vectorization
- **Search Engines**: Use `SearchEngine` enum from `utils/enum.py`
- **Localization**: Check `i18n/locale_msg.py` for UI text

## Testing & Evaluation

### Running Tests
```bash
cd app/backend
pytest tests/ -v                           # All tests
pytest tests/test_file_upload_system.py    # File upload only
```

### Batch Evaluation
```bash
cd app/backend/evals
python batch_eval.py --input data/RTF_queries.csv --max_concurrent 3 --temperature 0.5
```
- Uses **Azure AI Evaluation SDK**: `RelevanceEvaluator`, `SimilarityEvaluator`, `RetrievalEvaluator`
- Results saved to `results/` with timestamped JSON/matplotlib plots

## Local Development Workflow

### Backend Setup
```bash
cd app/backend
uv venv .venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env              # Configure Azure credentials
uv run run.py                     # Starts on http://localhost:8000
```

### Frontend Setup
```bash
cd app/frontend
./run_app.sh                      # Starts Chainlit on http://localhost:7860
```

### Environment Variables Required
See `.env.example` for full list. Critical ones:
- `AZURE_OPENAI_*` (API key, endpoint, deployments)
- `AZURE_AI_SEARCH_*` (for document search)
- `AZURE_DOCUMENT_INTELLIGENCE_*` (for PDF processing)
- `BING_API_KEY` (optional, for web search)

## Deployment

### Azure Container Apps (Bicep)
```bash
cd infra
./deploy.sh                       # Deploys to Azure Container Apps
```
- **Identity**: Uses `main_aca_identity_exist.bicep` if managed identity exists
- **Resources**: Log Analytics → Container App Environment → Backend + Frontend apps
- **Configuration**: Edit `main.parameters.json` before deploying

## Common Pitfalls

1. **Mixing orchestrators**: Don't call AFW executors from SK plugins or vice versa
2. **Prompt loading**: Always use `load_prompt()` from langchain, not raw file reads
3. **Streaming**: Must yield string chunks; never return complete responses in streaming mode
4. **Timezone**: Use `settings.TIME_ZONE` (default: "Asia/Seoul") for timestamps
5. **Logging**: SK agents are noisy; see `main.py` for logger silencing pattern
6. **File uploads**: Use `UnifiedFileUploadPlugin` which handles semantic/page chunking via `PROCESSING_METHOD` env var

## Key Dependencies

- **agent-framework**: 1.0.0b251112 (MS Agent Framework)
- **semantic-kernel**: 1.37.0 (MS Semantic Kernel)
- **chainlit**: 2.8.0 (Frontend)
- **fastapi**: 0.103.0+ (Backend API)
- **azure-ai-evaluation**: 1.11.0 (Evaluation)
- **azure-search-documents**: 11.5.3 (Document search)
- **langchain**: 0.3.25 (Prompt management)

## When to Use What

| Task | Tool/Pattern |
|------|-------------|
| Add new search source | Create Executor in `services_afw/` with `@handler` |
| Modify agent behavior | Edit YAML prompts in `prompts/` |
| Change orchestration flow | Modify `PlanSearchOrchestratorAFW.generate_response()` |
| Add evaluation metric | Extend `batch_eval.py` with new Azure evaluator |
| Debug streaming | Check `yield_message.py` utilities and base64 encoding |
| Add UI feature | Edit Chainlit handlers in `app/frontend/src/app.py` |


