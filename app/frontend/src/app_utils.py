import threading
import time
import base64
import logging
from datetime import datetime
from collections import defaultdict
from typing import Dict, Tuple, Any, List
import chainlit as cl

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# Starter Configuration - 중앙 관리
# ============================================================================

class StarterConfig:
    """Centralized starter configuration"""
    
    # Starter categories in order
    CATEGORIES = ["upload", "report", "ask_questions"]
    
    # Category configuration with emoji and images
    CATEGORY_CONFIG = {
        "upload": {
            "emoji": "⤴️",
            "image": "/public/images/2934_color.png",
            "send_to_backend": False  # Upload starter doesn't call backend
        },
        "report": {
            "emoji": "💡",
            "image": "/public/images/1f4a1_color.png",
            "send_to_backend": True
        },
        "ask_questions": {
            "emoji": "❓",
            "image": "/public/images/2753_color.png",
            "send_to_backend": True
        }
    }
    
    @classmethod
    def get_category_config(cls, category: str) -> Dict[str, Any]:
        """Get configuration for a category"""
        return cls.CATEGORY_CONFIG.get(category, {
            "emoji": "🤖",
            "image": None,
            "send_to_backend": True
        })
    
    @classmethod
    def should_send_to_backend(cls, category_index: int) -> bool:
        """Check if starter should trigger backend call"""
        if 0 <= category_index < len(cls.CATEGORIES):
            category = cls.CATEGORIES[category_index]
            return cls.CATEGORY_CONFIG.get(category, {}).get("send_to_backend", True)
        return True

class ChatSettings:
    """Chat settings for managing user preferences"""
    def __init__(self):
        self.research = True
        self.web_search = False
        self.planning = True
        self.ytb_search = False
        self.mcp_server = False
        self.ai_search = True
        self.multi_agent_type = "sk" # sk, vanilla
        self.verbose = True
        self.search_engine = "grounding_bing"  # Default value
        self.language = "en-US"
        self.max_tokens = 4000
        self.temperature = 0.7

class StepNameManager:
    """Thread-safe step name counter with cleanup"""
    def __init__(self, cleanup_interval: int = 3600):  # 1 hour cleanup interval
        self._counter = defaultdict(int)
        self._timestamps = {}
        self._lock = threading.Lock()
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()
    
    def get_unique_name(self, base_name: str) -> str:
        """Generate a unique step name with intelligent deduplication."""
        with self._lock:
            # Cleanup old entries periodically
            current_time = time.time()
            if current_time - self._last_cleanup > self._cleanup_interval:
                self._cleanup_old_entries(current_time)
                self._last_cleanup = current_time
            
            # Clean the base name for better readability
            clean_name = self._clean_step_name(base_name)
            
            # Check if name already exists
            if clean_name not in self._counter:
                self._counter[clean_name] = 1
                self._timestamps[clean_name] = current_time
                return clean_name
            else:
                # Generate numbered variant
                self._counter[clean_name] += 1
                count = self._counter[clean_name]
                unique_name = f"{clean_name} ({count})"
                self._timestamps[unique_name] = current_time
                return unique_name
    
    def _clean_step_name(self, name: str) -> str:
        """Clean and normalize step name for better display."""
        return name.strip()
    
    def _cleanup_old_entries(self, current_time: float):
        """Remove old entries to prevent memory leaks."""
        cutoff_time = current_time - (self._cleanup_interval * 2)
        
        # Find entries to remove
        to_remove = []
        for name, timestamp in self._timestamps.items():
            if timestamp < cutoff_time:
                to_remove.append(name)
        
        # Remove old entries
        for name in to_remove:
            if name in self._counter:
                del self._counter[name]
            if name in self._timestamps:
                del self._timestamps[name]

    def reset_counter(self, name_pattern: str = None):
        """Reset counters for specific patterns or all if no pattern given."""
        with self._lock:
            if name_pattern:
                to_remove = [name for name in self._counter if name_pattern in name]
                for name in to_remove:
                    del self._counter[name]
                    if name in self._timestamps:
                        del self._timestamps[name]
            else:
                self._counter.clear()
                self._timestamps.clear()
    
    def get_stats(self) -> Dict:
        """Get statistics about current step names."""
        with self._lock:
            return {
                'total_names': len(self._counter),
                'most_common': max(self._counter.items(), key=lambda x: x[1]) if self._counter else None,
                'memory_usage_bytes': sum(len(k.encode()) + len(str(v).encode()) for k, v in self._counter.items())
            }

class UploadManager:
    """Manage active uploads with thread-safe operations"""
    def __init__(self):
        self._uploads = {}
        self._lock = threading.Lock()
    
    def add_upload(self, upload_id: str, files: List[str], message: cl.Message, task):
        """Add a new upload to tracking"""
        with self._lock:
            self._uploads[upload_id] = {
                "files": files,
                "message": message,
                "task": task,
                "examples_sent": False,
                "created_at": datetime.now()
            }
    
    def get_upload(self, upload_id: str) -> Dict:
        """Get upload information"""
        with self._lock:
            return self._uploads.get(upload_id)
    
    def has_active_task(self, upload_id: str) -> bool:
        """Check if upload has an active task"""
        with self._lock:
            entry = self._uploads.get(upload_id)
            return entry is not None and entry.get("task") is not None
    
    def clear_task(self, upload_id: str):
        """Clear task for an upload"""
        with self._lock:
            if upload_id in self._uploads:
                self._uploads[upload_id]["task"] = None
    
    def set_examples_sent(self, upload_id: str):
        """Mark that example questions have been sent"""
        with self._lock:
            if upload_id in self._uploads:
                self._uploads[upload_id]["examples_sent"] = True
    
    def remove_upload(self, upload_id: str):
        """Remove upload from tracking"""
        with self._lock:
            if upload_id in self._uploads:
                del self._uploads[upload_id]
    
    def get_all_uploads(self) -> Dict:
        """Get all active uploads"""
        with self._lock:
            return dict(self._uploads)

# ============================================================================
# Utility Functions
# ============================================================================

def decode_step_content(content: str, step_name_manager) -> Tuple[str, str, str]:
    """
    Decode step content and generate unique step name.
    
    Args:
        content: The step content to decode
        step_name_manager: StepNameManager instance for generating unique names
    
    Returns:
        tuple: (step_name, code_content, description)
    """
    try:
        # Parse the content for different components
        step_name = content.strip()
        code_content = ""
        description = ""
        
        # Check for #input# tags for descriptions
        if '#input#' in content:
            parts = content.split('#input#')
            if len(parts) >= 2:
                step_name = parts[0].strip()
                description = parts[1].strip()
        
        # Check for #code# tags for code content
        if '#code#' in content:
            parts = content.split('#code#')
            if len(parts) >= 2:
                if not step_name or step_name == content.strip():
                    step_name = parts[0].strip()
                
                try:
                    # Decode base64 encoded code
                    encoded_code = parts[1].strip()
                    decoded_bytes = base64.b64decode(encoded_code)
                    code_content = decoded_bytes.decode('utf-8')
                except Exception as e:
                    logger.warning(f"Failed to decode code content: {e}")
                    code_content = parts[1].strip()  # Use raw content as fallback
        
        # # Generate unique step name using the manager not use anymore
        #unique_step_name = step_name_manager.get_unique_name(step_name) 
        
        return step_name, code_content, description
        
    except Exception as e:
        logger.error(f"Error decoding step content: {e}")
        # Fallback to basic parsing
        fallback_name = step_name_manager.get_unique_name(content[:50] + "..." if len(content) > 50 else content)
        return fallback_name, "", ""

def create_api_payload(settings: ChatSettings) -> dict:
    """Create API payload from settings"""
    message_history = cl.chat_context.to_openai()
    return {
        "messages": message_history[-10:],
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "research": settings.research,
        "planning": settings.planning,
        "include_web_search": settings.web_search,
        "include_ytb_search": settings.ytb_search,
        "include_mcp_server": settings.mcp_server,
        "include_ai_search": settings.ai_search,
        "multi_agent_type": settings.multi_agent_type,
        "search_engine": settings.search_engine,
        "stream": True,
        "locale": settings.language,
        "verbose": settings.verbose,
    }

async def safe_stream_token(msg: cl.Message, content: str) -> bool:
    """Safely stream token with connection check"""
    try:
        await msg.stream_token(content)
        return True
    except Exception as e:
        logger.warning(f"Failed to stream token: {str(e)}")
        return False

async def safe_send_step(step: cl.Step) -> bool:
    """Safely send step with connection check"""
    try:
        await step.send()
        return True
    except Exception as e:
        logger.warning(f"Failed to send step: {str(e)}")
        return False

async def safe_update_message(msg: cl.Message) -> bool:
    """Safely update message with connection check"""
    try:
        await msg.update()
        return True
    except Exception as e:
        logger.warning(f"Failed to update message: {str(e)}")
        return False

async def handle_error_response(msg: cl.Message, error_type: str, error_msg: str):
    """Handle different types of errors uniformly"""
    full_msg = f"❌ **{error_type}**: {error_msg}"
    await safe_stream_token(msg, full_msg)
    logger.error(f"{error_type}: {error_msg}")

async def retry_async_operation(
    operation,
    *args,
    max_retries: int = 3,
    initial_delay: float = 0.2,
    backoff_factor: float = 2.0,
    **kwargs
) -> Tuple[bool, Any]:
    """
    Retry an async operation with exponential backoff
    
    Args:
        operation: Async function to retry
        *args: Positional arguments for the operation
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        **kwargs: Keyword arguments for the operation
    
    Returns:
        Tuple of (success: bool, result: Any)
    """
    import asyncio
    
    delay = initial_delay
    last_error = None
    
    for attempt in range(max_retries):
        try:
            result = await operation(*args, **kwargs)
            return True, result
        except Exception as e:
            last_error = e
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {str(e)}"
            )
            
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
                delay *= backoff_factor
    
    logger.error(f"Operation failed after {max_retries} attempts: {last_error}")
    return False, last_error