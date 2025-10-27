from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file"""

    # Azure OpenAI Settings
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_VERSION: str = "2023-05-15"
    AZURE_OPENAI_DEPLOYMENT_NAME: str
    AZURE_OPENAI_QUERY_DEPLOYMENT_NAME: Optional[str] = None
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME: str
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

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",  # Ignore extra fields to prevent validation errors
    )
