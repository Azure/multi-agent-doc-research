from typing import Optional
from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Project root directory (multi-agent-doc-research/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()


def resolve_project_path(path_str: str) -> Path:
    """
    Resolve relative paths to absolute paths based on project root.
    
    Args:
        path_str: Path string (relative or absolute)
        
    Returns:
        Absolute Path object
        
    Examples:
        ./graphrag/input -> /afh/code/multi-agent-doc-research/graphrag/input
        /absolute/path -> /absolute/path (unchanged)
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file"""

    # Azure OpenAI Settings
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_VERSION: str = "2023-05-15"
    AZURE_OPENAI_DEPLOYMENT_NAME: str
    AZURE_OPENAI_QUERY_DEPLOYMENT_NAME: Optional[str] = None
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME: str
    AZURE_OPENAI_REASONING_DEPLOYMENT_NAME: Optional[str] = None  
    PLANNER_MAX_PLANS: int = 3  # Maximum number of plans to generate

    # Redis Settings
    REDIS_USE: bool = False
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_DB: int = 0
    REDIS_CACHE_EXPIRED_SECOND: int = 604800  # 7 days

    # Google Search API Settings
    GOOGLE_API_KEY: Optional[str] = None
    GOOGLE_CSE_ID: Optional[str] = None
    GOOGLE_MAX_RESULTS: int = 10

    # Optional SERP API Key (if needed)
    SERP_API_KEY: Optional[str] = None

    # Application Settings
    LOG_LEVEL: str = "INFO"
    MAX_TOKENS: int = 10000
    DEFAULT_TEMPERATURE: float = 0.7
    TIME_ZONE: str = "Asia/Seoul"
    
    # AI Search Settings
    AZURE_AI_SEARCH_ENDPOINT: str = None
    AZURE_AI_SEARCH_API_KEY: str = None
    AZURE_AI_SEARCH_INDEX_NAME: str = None
    AZURE_AI_SEARCH_SEARCH_TYPE: str = None

    # Document Intelligence Settings
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT: str = None
    AZURE_DOCUMENT_INTELLIGENCE_API_KEY: str = None
    
    # GraphRAG Settings
    GRAPHRAG_ENABLED: bool = False
    GRAPHRAG_ROOT: str = "./graphrag"
    GRAPHRAG_INPUT_DIR: str = "./graphrag/input"  # ✅ Match .env variable name
    GRAPHRAG_OUTPUT_DIR: str = "./graphrag/output"  # ✅ Add output dir
    GRAPHRAG_MCP_SERVER_SCRIPT: Optional[str] = None
    
    def get_graphrag_input_dir(self) -> Path:
        """Get absolute path to GraphRAG input directory"""
        return resolve_project_path(self.GRAPHRAG_INPUT_DIR)
    
    def get_graphrag_output_dir(self) -> Path:
        """Get absolute path to GraphRAG output directory"""
        return resolve_project_path(self.GRAPHRAG_OUTPUT_DIR)
    
    def get_graphrag_root(self) -> Path:
        """Get absolute path to GraphRAG root directory"""
        return resolve_project_path(self.GRAPHRAG_ROOT)
    
    # Bing Search Settings
    BING_API_KEY: Optional[str] = None
    BING_ENDPOINT: str = "https://api.bing.microsoft.com/v7.0/search"
    
    # Locale
    LOCALE: str = "ko"

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",  # Ignore extra fields to prevent validation errors
    )
