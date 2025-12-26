# GraphRAG MCP Server

독립적인 MCP (Model Context Protocol) 서버로 GraphRAG 기능을 제공합니다.

## 🎯 Purpose

GraphRAG는 `numpy<2.0`을 요구하지만, main backend는 `agent-framework-redis`를 위해 `numpy>=2.2.6`이 필요합니다. 이 충돌을 해결하기 위해 GraphRAG를 **독립 프로세스**로 실행하여 numpy 버전을 격리합니다.

## 📦 Architecture

```
Main Backend (numpy>=2.2.6)
    ↓ stdio MCP protocol
GraphRAG MCP Server (numpy<2.0) ← Isolated venv
    ↓ subprocess calls
GraphRAG CLI (python -m graphrag.index/query)
```

## 🚀 Installation

### 자동 설치 (권장)

```bash
cd /afh/code/multi-agent-doc-research/graphrag_mcp_server
./run_server.sh  # 자동으로 venv 생성 및 패키지 설치
```

### 수동 설치

```bash
cd /afh/code/multi-agent-doc-research/graphrag_mcp_server

# 1. 가상환경 생성
python3 -m venv .venv
source .venv/bin/activate

# 2. 패키지 설치
pip install --upgrade pip
pip install -e .

# 3. 서버 실행
python -m graphrag_mcp_server.server
```

## ⚙️ Environment Variables

다음 환경 변수가 필요합니다 (`.env` 파일 또는 시스템 환경):

```bash
# Azure OpenAI (GraphRAG 인덱싱/검색에 사용)
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-large
AZURE_OPENAI_API_VERSION=2024-08-01-preview

# GraphRAG 데이터 경로
GRAPHRAG_ROOT=./graphrag
```

## 🔧 MCP Tools

GraphRAG MCP 서버는 3개의 MCP tool을 제공합니다:

### 1. index_documents

Markdown 파일들을 GraphRAG 형식으로 인덱싱합니다.

**Parameters:**
- `markdown_files` (array of strings): 인덱싱할 markdown 파일 경로들
- `force_reindex` (boolean, optional): 강제 재인덱싱 여부 (기본값: false)

**Returns:**
```json
{
  "status": "success",
  "message": "Indexed 5 files",
  "files_indexed": 5,
  "output_dir": "/path/to/graphrag/output"
}
```

### 2. local_search

엔티티 중심의 상세한 검색 (특정 개체/인물/조직에 대한 질문에 적합)

**Parameters:**
- `query` (string): 검색 쿼리
- `top_k` (integer, optional): 반환할 결과 수 (기본값: 10)

**Returns:**
```json
{
  "status": "success",
  "query": "주요 사업 전략은?",
  "response": "상세한 검색 결과...",
  "search_type": "local"
}
```

### 3. global_search

커뮤니티 중심의 주제별 검색 (전체 문서의 주제/트렌드 질문에 적합)

**Parameters:**
- `query` (string): 검색 쿼리

**Returns:**
```json
{
  "status": "success",
  "query": "전체 문서의 주요 주제는?",
  "response": "종합적인 검색 결과...",
  "search_type": "global"
}
```

## 📁 Project Structure

```
graphrag_mcp_server/
├── graphrag_mcp_server/          # Package directory
│   ├── __init__.py               # Package initialization
│   └── server.py                 # MCP server implementation (450+ lines)
├── pyproject.toml                # Package configuration
├── run_server.sh                 # Server startup script
├── .venv/                        # Isolated Python environment (numpy<2.0)
└── README.md                     # This file
```

## 🔌 Integration with Backend

### Semantic Kernel Plugin

```python
from services_sk.graphrag_mcp_plugin import GraphRAGMCPPlugin

plugin = GraphRAGMCPPlugin()

# Index documents
result = await plugin.index_documents(
    markdown_files=json.dumps(["/path/to/doc1.md", "/path/to/doc2.md"])
)

# Local search
result = await plugin.local_search(
    query="주요 사업은?",
    top_k=10
)

# Global search
result = await plugin.global_search(query="전체 주제는?")

# Cleanup
await plugin.cleanup()
```

### Agent Framework Executor

```python
from services_afw.graphrag_executor import GraphRAGExecutor

executor = GraphRAGExecutor()

# Via workflow context
await executor.index_documents(
    ctx=workflow_context,
    workflow_input={"markdown_files": [...]}
)

await executor.local_search(
    ctx=workflow_context,
    workflow_input={"query": "...", "top_k": 10}
)
```

## 🧪 Testing

```bash
# Backend 테스트 실행
cd /afh/code/multi-agent-doc-research/app/backend
pytest tests/test_graphrag_mcp.py -v -s
```

테스트는 다음 시나리오를 검증합니다:
1. PDF 업로드 및 Markdown 생성
2. GraphRAG 인덱싱 (MCP Plugin)
3. GraphRAG 인덱싱 (AFW Executor)
4. Local Search (MCP Plugin)
5. Local Search (AFW Executor)
6. Global Search (MCP Plugin)
7. Global Search (AFW Executor)
8. 연결 정리

## 📊 Data Flow

```
1. Upload PDF
   ↓
2. Document Intelligence → Markdown
   ↓
3. Save to graphrag/input/*.md
   ↓
4. MCP Client calls index_documents
   ↓
5. MCP Server runs: python -m graphrag.index --root ./graphrag
   ↓
6. GraphRAG creates parquet files in output/
   ↓
7. Search via local_search or global_search
   ↓
8. Results returned to backend
```

## 🛠️ Troubleshooting

### "Unable to determine which files to ship"

**원인**: hatchling이 패키지 디렉토리를 찾지 못함

**해결**:
```bash
# 구조가 올바른지 확인:
graphrag_mcp_server/
  graphrag_mcp_server/  # 패키지 디렉토리 (NOT root)
    __init__.py
    server.py
  pyproject.toml
```

### "ModuleNotFoundError: No module named 'graphrag_mcp_server'"

**원인**: 패키지가 설치되지 않음

**해결**:
```bash
cd graphrag_mcp_server
source .venv/bin/activate
pip install -e .
```

### "numpy version conflict"

**원인**: main backend의 numpy>=2.2.6와 충돌

**해결**: MCP 서버는 **독립 venv**에서 실행되므로 충돌하지 않습니다. `run_server.sh`를 사용하면 자동으로 격리됩니다.

## 📖 GraphRAG Settings

인덱싱 시 자동으로 생성되는 `settings.yaml`:

```yaml
llm:
  api_key: ${AZURE_OPENAI_API_KEY}
  type: azure_openai_chat
  model: gpt-4o
  api_base: ${AZURE_OPENAI_ENDPOINT}
  api_version: ${AZURE_OPENAI_API_VERSION}
  deployment_name: ${AZURE_OPENAI_DEPLOYMENT_NAME}

embeddings:
  async_mode: threaded
  llm:
    api_key: ${AZURE_OPENAI_API_KEY}
    type: azure_openai_embedding
    model: text-embedding-3-large
    api_base: ${AZURE_OPENAI_ENDPOINT}
    api_version: ${AZURE_OPENAI_API_VERSION}
    deployment_name: ${AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME}

input:
  type: file
  file_type: text
  base_dir: "input"
  
# ... more GraphRAG settings
```

## 🔗 References

- [GraphRAG Documentation](https://microsoft.github.io/graphrag/)
- [MCP Protocol Specification](https://github.com/modelcontextprotocol/specification)
- [Agent Framework](https://github.com/microsoft/agent-framework)
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel)

## 📝 License

Same as parent project (multi-agent-doc-research)
