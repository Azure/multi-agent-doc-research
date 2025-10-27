from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from utils.enum import SearchEngine


class UploadedFile(BaseModel):
    """Uploaded file metadata"""
    filename: str = Field(..., description="Original filename")
    file_path: str = Field(..., description="Temporary file path on server")
    file_size: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="MIME type of the file")


class FileUploadRequest(BaseModel):
    """File upload request model"""
    files: List[UploadedFile] = Field(..., description="List of uploaded files (max 10)")
    document_type: str = Field("IR_REPORT", description="Type of document")
    company: Optional[str] = Field(None, description="Company name")
    industry: Optional[str] = Field(None, description="Industry category")
    report_year: Optional[str] = Field(None, description="Report year")
    force_upload: bool = Field(False, description="Force upload even if document exists")


class FileUploadResponse(BaseModel):
    """File upload response model"""
    status: str = Field(..., description="Overall upload status")
    total_files: int = Field(..., description="Total number of files processed")
    successful_uploads: int = Field(..., description="Number of successful uploads")
    failed_uploads: int = Field(..., description="Number of failed uploads")
    results: List[Dict] = Field(..., description="Detailed results for each file")
    message: str = Field(..., description="Summary message")
    upload_id: Optional[str] = Field(None, description="Unique upload identifier for status tracking")


class ChatMessage(BaseModel):
    """Chat message model with role and content"""
    role: str = Field(..., description="Role of the message sender (system, user, assistant)")
    content: str = Field(..., description="Content of the message")
    files: Optional[List[dict]] = Field(None, description="Optional list of uploaded files")



class PlanSearchRequest(BaseModel):
    """Risk search request model with additional DART-specific parameters"""
    messages: List[ChatMessage] = Field(
        ..., 
        description="List of chat messages in the conversation history"
    )
    max_tokens: Optional[int] = Field(
        None, 
        description="Maximum number of tokens to generate"
    )
    temperature: Optional[float] = Field(
        None, 
        description="Temperature for response generation (0.0 to 1.0)"
    )
    stream: bool = Field(
        False,
        description="Enable streaming response"
    )
    research: bool = Field(
        True,
        description="Enable research mode using multi-agent group chat"
    )
    planning: bool = Field(
        True,
        description="Enable search planning for better search results"
    )
    search_engine: SearchEngine = Field(
        SearchEngine.BING_SEARCH_CRAWLING,
        description="Search engine to use for retrieving information"
    )
    locale: str = Field(
        "ko-KR",
        description="Locale for the response"
    )
    include_web_search: bool = Field(
        True,
        description="Include web search results"
    )
    include_ytb_search: bool = Field(
        True,
        description="Include YouTube search results"
    )
    include_mcp_server: bool = Field(
        True,
        description="Include MCP server integration"
    )
    include_ai_search: bool = Field(
        True,
        description="Include AI search results from uploaded documents"
    )
    multi_agent_type: str = Field(
        "vanilla",
        description="The type of multi-agent system to use (e.g., 'sk', 'vanilla')"
    )
    verbose: bool = Field(
        True,
        description="Verbose mode for detailed output"
    )
    

class ChatResponse(BaseModel):
    """Chat response model with message and success status"""
    message: str = Field(..., description="Response message from the chatbot")
    success: bool = Field(..., description="Indicates if the request was successful")
