# Multi-Agent Doc Research
This AI-powered chatbot performs custom deep research on uploaded documents using a semantic chunking strategy for precise and meaningful vectorization. Through multi-agent collaboration, it delivers accurate, context-aware answers to user queries.

Built with FastAPI, Azure OpenAI, and Chainlit, the system showcases advanced techniques for enhancing LLM-based applications—such as agentic patterns, modular architecture, multi-agent orchestration, and evaluation support.

At its core, the multi-agent deep research engine combines Microsoft Agent Framework and Semantic Kernel to generate high-quality analytical reports. By employing group chat coordination and a magnetic multi-agent pattern, it achieves deeper reasoning and consistent, well-structured outputs.

![multi-agent-doc-research-architecture-Page-2.jpg](/images/multi-agent-doc-research-architecture-Page-2.jpg)


### 🧠 MS Agent Framework Integration
- The chatbot now incorporates [MS Agent Framework](https://github.com/microsoft/agent-framework), Microsoft's an open-source SDK and runtime designed to let developers build, deploy, and manage sophisticated multi-agent systems with ease. It unifies the enterprise-ready foundations of Semantic Kernel with the innovative orchestration of AutoGen, so teams no longer have to choose between experimentation and production.

### 🧠 Semantic Kernel Integration
- The chatbot now incorporates [Semantic Kernel](https://github.com/microsoft/semantic-kernel), Microsoft's open-source orchestration SDK for LLM apps.
- Enables more intelligent planning and contextual understanding, resulting in richer, more accurate responses.
- Supports planner-based execution and native function calling for complex multi-step tasks.


### 🔍 Verbose Mode
- Introduced **verbose mode** for improved debugging and traceability.
- Logs include:
  - Raw input/output data
  - API call history
  - Function invocation details
- Helps track down issues and optimize prompt behavior.

### 🎨 UI Framework
- Now supports the following UI framework:
  - [Chainlit](https://github.com/Chainlit/chainlit) – great for interactive prototyping
  
### 🔁 Query Rewrite
- A module that reformulates user queries to improve response quality and informativeness.  
- Helps the LLM better understand the user's intent and generate more accurate, context-aware answers.

### 🧭 Plan & Execute
- Implements planning techniques to **enrich search keywords** based on the original query context.  
- Automatically decomposes **complex questions into sub-queries**, searches them, and returns synthesized context to the chatbot.  
- Boosts performance in multi-intent or multi-hop question scenarios.

## 🤖 Multi-Agent Collaboration Patterns

This project implements sophisticated multi-agent collaboration patterns such as Group Chat, Magentic patterns using Microsoft Agent Framework, enabling intelligent coordination between specialized AI agents for complex research tasks.

### Available Patterns

#### 1. **Group Chat Pattern** 
Sequential turn-based collaboration where agents refine outputs through iterative dialogue.

- **Architecture**: Writer ↔ Reviewer loop with approval-based termination
- **Agents**: 
  - `ResearchWriter`: Generates comprehensive research content
  - `ResearchReviewer`: Validates quality, accuracy, and citation integrity
- **Best For**: 
  - Iterative content refinement
  - Quality assurance workflows
  - Approval-based processes
- **Performance**: ⚡ Fast | 💰 Medium tokens | ⭐⭐⭐⭐ Quality

**Usage:**
```python
orchestrator = PlanSearchOrchestratorAFW(settings)
async for chunk in orchestrator.generate_response(
    messages=messages,
    research=True,
    multi_agent_type="MS Agent Framework GroupChat",
    stream=True
):
    print(chunk, end="")
```

#### 2. **Magentic Orchestration Pattern** ⭐
Intelligent orchestration with a manager agent coordinating specialized agents adaptively.

- **Architecture**: Orchestrator → Dynamic agent coordination → Adaptive execution
- **Agents**:
  - `Orchestrator`: Intelligent planning and task decomposition
  - `ResearchAnalyst`: Information synthesis and pattern identification
  - `ResearchWriter`: Comprehensive content generation with citations
  - `ResearchReviewer`: Quality validation and scoring
- **Best For**:
  - Complex multi-step research tasks
  - Dynamic task decomposition
  - Adaptive problem-solving requiring different expertise
- **Performance**: 🐢 Medium speed | 💰💰 Higher tokens | ⭐⭐⭐⭐⭐ Excellent quality

**Usage:**
```python
orchestrator = PlanSearchOrchestratorAFW(settings)
async for chunk in orchestrator.generate_response(
    messages=messages,
    research=True,
    multi_agent_type="MS Agent Framework Magentic",
    stream=True
):
    print(chunk, end="")
```

### Pattern Comparison

| Aspect | Group Chat | Magentic Orchestration |
|--------|-----------|------------------------|
| **Execution** | Sequential dialogue | Intelligent orchestration |
| **Planning** | None (fixed workflow) | Built-in adaptive planning |
| **Agent Coordination** | Turn-based | Dynamic by orchestrator |
| **Rounds** | 3-5 fixed iterations | 1-5+ adaptive rounds |
| **Speed** | ⚡ Fast | 🐢 Medium |
| **Token Usage** | 💰 Medium | 💰💰 High |
| **Quality** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Best For** | Refinement workflows | Complex multi-step tasks |

### When to Use Each Pattern

**Use Group Chat when:**
- ✅ You need iterative refinement with clear review cycles
- ✅ Speed is important
- ✅ Fixed writer-reviewer workflow is sufficient
- ✅ Lower token consumption is preferred

**Use Magentic Orchestration when:**
- ✅ Research requires multi-step analysis and synthesis
- ✅ Complex task decomposition is needed
- ✅ Adaptive coordination provides value
- ✅ Quality is prioritized over speed
- ✅ Tasks require different types of expertise

### Implementation Details

Both patterns are fully integrated into the orchestration workflow:

```
User Query → Intent Analysis → Search Planning → Multi-Source Search
                                                    ↓
                                    (Web + AI Search + YouTube)
                                                    ↓
                                        ┌───────────────────────┐
                                        │  Multi-Agent Pattern  │
                                        │                       │
                                        │  • Group Chat         │
                                        │  • Magentic          │
                                        └───────────┬───────────┘
                                                    ↓
                                          Streaming Markdown Output
```

**Key Features:**
- 🔄 **Streaming Support**: Real-time progress updates and token-by-token streaming
- 📊 **Context Integration**: Seamless integration with Web Search, AI Search, and YouTube contexts
- 🎯 **Sub-topic Processing**: Parallel processing of multiple research sub-topics
- ⚡ **TTFT Tracking**: Time-to-first-token monitoring for performance optimization
- 🛡️ **Error Handling**: Robust error handling with graceful degradation
- 📝 **Citation Management**: Automatic source attribution and reference tracking


# Project Structure

The project is organized into two main parts:

- `backend`: Contains the FastAPI server and all backend functionality
- `frontend`: Contains the frontend UI

### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Azure subscription with OpenAI service enabled
- uv
```bash
uv venv .venv --python 3.12 --seed
source .venv/bin/activate
```
### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multi-agent-doc-research.git
   cd multi-agent-doc-research/app/backend
   ```

2. Install backend dependencies using uv:
    ```bash
    uv pip install -e .
    ```

    For development dependencies:
    ```bash
    uv pip install -e ".[dev]"
    ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Then edit the `.env` file and add your Azure OpenAI credentials:
   ```
    # Azure OpenAI Configuration
    AZURE_OPENAI_API_KEY=your-api-key-here
    AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
    AZURE_OPENAI_API_VERSION=2023-05-15
    AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
    AZURE_OPENAI_QUERY_DEPLOYMENT_NAME=your-query-deployment-name
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=your-embedding-deployment-name

    # Redis Configuration
    REDIS_USE=False
    REDIS_HOST=localhost
    REDIS_PORT=6379
    REDIS_PASSWORD=redis_secure_password
    REDIS_DB=0
    REDIS_CACHE_EXPIRED_SECOND=604800

    # Application Settings
    LOG_LEVEL=INFO
    MAX_TOKENS=2000
    DEFAULT_TEMPERATURE=0.7

    # When you use the Bing Custom Search API, you need to set the custom configuration ID.

    # Planner Settings
    PLANNER_MAX_PLANS=3

    # AI Search
    AZURE_AI_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
    AZURE_AI_SEARCH_API_KEY=your-search-service-api-key
    AZURE_AI_SEARCH_INDEX_NAME=doc_inquiry_index
    AZURE_AI_SEARCH_SEARCH_TYPE=semantic  # Options: "semantic", "simple", "hybrid"

    # Document Intelligence
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-cognitive-services-account.cognitiveservices.azure.com/
    AZURE_DOCUMENT_INTELLIGENCE_API_KEY=your-document-intelligence-api-key

    # Chunking Method
    # Use "semantic" for semantic chunking, "page" for page-based chunking
    PROCESSING_METHOD=semantic  


   ```

   

### Running the Backend

Start the FastAPI server:
```bash
uv run run.py
```

The API will be available at:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

### Running the Frontend

Run the application:
   ```bash
   ./run_app.sh
   ```

## Usage

- Open your web browser and navigate to public URL `http://localhost:7860/` to access the Chainlit interface.
- Upload documents using the "Upload" button.
- Enter your message in the input box and click "Submit" to interact with the chatbot.


## Contributing

Feel free to submit issues or pull requests if you have suggestions or improvements for the project.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
````