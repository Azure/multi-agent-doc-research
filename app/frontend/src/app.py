import chainlit as cl
import os
import sys
import json
import logging
import base64
import asyncio
import threading
import time
from datetime import datetime
from collections import defaultdict
from typing import Dict, Tuple, Any, List, Optional
from i18n.locale_msg_front import UI_TEXT, EXAMPLE_PROMPTS
from pathlib import Path
from io import BytesIO
from enum import Enum
import os
import aiohttp

os.environ["CHAINLIT_CORS_ALLOW_ORIGIN"] = (
    "*"  # 개발 환경용, 프로덕션에서는 특정 도메인 지정
)

# Import classes and utilities from app_utils
from app_utils import (
    ChatSettings,
    StepNameManager,
    UploadManager,
    StarterConfig,
    decode_step_content,
    create_api_payload,
    safe_stream_token,
    safe_send_step,
    retry_async_operation,
    safe_update_message,
    handle_error_response,
)

# Configuration from environment variables
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "ms_user")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "msuser123")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
SK_API_URL = os.getenv("SK_API_URL", "http://localhost:8000/plan_search")
# Derive upload endpoint from SK_API_URL
UPLOAD_API_URL = os.getenv(
    "UPLOAD_API_URL", SK_API_URL.rsplit("/", 1)[0] + "/upload_documents"
)
# Status check endpoint
UPLOAD_STATUS_URL = os.getenv(
    "UPLOAD_STATUS_URL", SK_API_URL.rsplit("/", 1)[0] + "/upload_status"
)

active_uploads = {}  # { upload_id: { files: [...], message: cl.Message, task: asyncio.Task } }

# Define the search engines
SEARCH_ENGINES = {
    "Bing Search": "bing_search_crawling",
    "Grounding Gen": "grounding_bing",
}

# Define the multi_agent_type
MULTI_AGENT_TYPES = {
    "MS Agent Framework GroupChat": "afw_group_chat",
    "MS Agent Framework Magentic": "afw_magentic",
    "Semantic Kernel GroupChat": "sk_group_chat",
    "Semantic Kernel Magentic(Deep-Research-Agents)": "sk_magentic",
    "Vanilla AOAI SDK": "vanilla",
}

# Internationalization constants
SUPPORTED_LANGUAGES = {"en-US": "English", "ko-KR": "한국어"}

# Global instances
step_name_manager = StepNameManager()
upload_manager = UploadManager()


# @cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Simple password authentication - fixed version"""
    try:
        logger.info(f"🔐 Authentication attempt - username: {username}")

        # MS 사용자 인증
        if username == AUTH_USERNAME and password == AUTH_PASSWORD:
            logger.info("✅ MS user authentication successful")
            return cl.User(
                identifier="ms_user",
                metadata={
                    "role": "user",
                    "name": "Microsoft User",
                    "login_time": datetime.now().isoformat(),
                },
            )

        # 관리자 인증
        elif username == "admin" and password == ADMIN_PASSWORD:
            logger.info("✅ Admin authentication successful")
            return cl.User(
                identifier="admin",
                metadata={
                    "role": "admin",
                    "name": "Administrator",
                    "login_time": datetime.now().isoformat(),
                },
            )

        logger.warning(f"❌ Authentication failed for user: {username}")
        return None

    except Exception as e:
        logger.error(f"❌ Authentication error: {e}")
        return None


# ============================================================================
# Helper Functions - 통합 및 간소화
# ============================================================================


def get_user_language(settings=None) -> str:
    """Get current user language from settings with fallback"""
    if settings is None:
        settings = cl.user_session.get("settings")
    return getattr(settings, "language", "en-US")


def get_chat_profile_language() -> str:
    """Get language from current chat profile"""
    current_profile = cl.user_session.get("chat_profile", "English")
    return "en-US" if current_profile == "English" else "ko-KR"


def get_starter_prompt(language: str, category: str) -> str:
    """Get prompt text for a category in the specified language"""
    return EXAMPLE_PROMPTS.get(language, {}).get(category, {}).get("prompt", "")


def get_starter_label(language: str, category: str) -> str:
    """Get starter label for a category in the specified language"""
    return EXAMPLE_PROMPTS.get(language, {}).get(category, {}).get("title", "")


def find_starter_category_for_prompt(language: str, prompt: str) -> Optional[str]:
    """Identify guide-only starter category when its prompt is sent verbatim."""
    if not prompt:
        return None

    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        return None

    for category in StarterConfig.CATEGORIES:
        config = StarterConfig.get_category_config(category)
        # guide-only 스타터만 검사 (send_to_backend=False)
        if config.get("send_to_backend", True):
            continue

        starter_prompt = get_starter_prompt(language, category)
        if starter_prompt and starter_prompt.strip() == normalized_prompt:
            return category

    return None


# ============================================================================
# Starter Functions - 통합 및 중앙화
# ============================================================================


def get_starters_for_language(language: str) -> List[cl.Starter]:
    """Get starters for a specific language - 통합 버전

    Args:
        language: Language code (e.g., 'ko-KR', 'en-US')

    Returns:
        List of Starter objects configured for the language
    """
    starters = []

    logger.info(f"Getting starters for language: {language}")
    logger.info(f"Available categories: {StarterConfig.CATEGORIES}")

    for i, category in enumerate(StarterConfig.CATEGORIES):
        if category not in EXAMPLE_PROMPTS.get(language, {}):
            logger.warning(
                f"Category '{category}' not found in EXAMPLE_PROMPTS for language '{language}'"
            )
            continue

        config = StarterConfig.get_category_config(category)

        starter = cl.Starter(
            label=get_starter_label(language, category),
            message=get_starter_prompt(language, category),
            icon=config.get("image"),
        )
        starters.append(starter)

        logger.info(
            f"Added starter: {category} - {starter.label} (send_to_backend: {config.get('send_to_backend', True)})"
        )

    return starters


async def create_starter_actions(
    starters: List[cl.Starter], language: str
) -> List[cl.Action]:
    """Create action buttons from starters - 통합 버전

    Args:
        starters: List of Starter objects
        language: Current language code (e.g., 'ko-KR', 'en-US')

    Returns:
        List of Action objects with proper configuration
    """
    actions = []

    for i, starter in enumerate(starters):
        if i >= len(StarterConfig.CATEGORIES):
            logger.warning(f"Starter index {i} exceeds available categories")
            continue

        category = StarterConfig.CATEGORIES[i]
        config = StarterConfig.get_category_config(category)
        emoji = config.get("emoji", "🤖")

        actual_message = get_starter_prompt(language, category)

        actions.append(
            cl.Action(
                name=f"starter_{i}",
                payload={
                    "message": actual_message,  # 실제 프롬프트 내용
                    "label": starter.label,
                    "index": i,
                    "category": category,
                    "send_to_backend": config.get("send_to_backend", True),
                },
                label=f"{emoji} {starter.label}",
                description=f"Use starter: {starter.label}",
            )
        )

    logger.info(f"Created {len(actions)} action buttons for language: {language}")
    return actions


async def send_starters_as_actions(language: str):
    """Send starters as action buttons in a message - 통합 버전

    Args:
        language: Current language code (e.g., 'ko-KR', 'en-US')
    """
    starters = get_starters_for_language(language)
    actions = await create_starter_actions(starters, language)

    # Create message with starter list
    starters_message = "📋 **Quick Start Options:**\n\n"

    for i, starter in enumerate(starters):
        if i >= len(StarterConfig.CATEGORIES):
            continue
        config = StarterConfig.get_category_config(StarterConfig.CATEGORIES[i])
        emoji = config.get("emoji", "🤖")
        starters_message += f"{emoji} **{starter.label}**\n"

    await cl.Message(content=starters_message, actions=actions).send()


# ============================================================================
# Upload-related functions
# ============================================================================


async def check_upload_status_once(upload_id: str) -> dict | None:
    """단발성 업로드 상태 조회 (aiohttp 사용)"""
    aiohttp_session = None
    try:
        # ✅ aiohttp ClientSession (짧은 타임아웃, 간단한 요청)
        timeout = aiohttp.ClientTimeout(total=None, connect=10.0, sock_read=30.0)
        aiohttp_session = aiohttp.ClientSession(timeout=timeout)

        async with aiohttp_session.get(f"{UPLOAD_STATUS_URL}/{upload_id}") as resp:
            if resp.status != 200:
                return None

            return await resp.json()

    except aiohttp.ClientError as e:
        logger.warning(f"[upload:{upload_id}] HTTP error during status check: {e}")
        return None
    except asyncio.TimeoutError as e:
        logger.warning(f"[upload:{upload_id}] Timeout during status check: {e}")
        return None
    except Exception as e:
        logger.warning(f"[upload:{upload_id}] 상태 조회 실패: {e}")
        return None
    finally:
        # ✅ aiohttp client 정리
        if aiohttp_session and not aiohttp_session.closed:
            try:
                await aiohttp_session.close()
                await asyncio.sleep(0.1)  # Graceful cleanup
            except Exception as e:
                logger.error(f"❌ Error closing status check client: {e}")


async def poll_upload_status_loop(
    upload_id: str, msg: cl.Message, interval: float = 3.0
):
    """주기적으로 상태를 폴링해서 동일 메시지를 갱신"""
    try:
        while True:
            status_data = await check_upload_status_once(upload_id)
            if not status_data:
                msg.content = (
                    f"⚠️ 업로드 ID {upload_id[:8]} 상태 조회 실패. 재시도 중..."
                )
                await msg.update()
                await asyncio.sleep(interval)
                continue

            status = status_data.get("status", "unknown")
            message = status_data.get("message", "")
            progress = int(status_data.get("progress", 0))
            file_results = status_data.get("file_results", [])

            if status == "processing":
                green_blocks = progress // 10
                progress_bar = "🟩" * green_blocks + "⬜" * (10 - green_blocks)
                msg.content = (
                    f"📤 **Uploading files...** (ID: {upload_id[:8]})\n"
                    f"{message}\n\n"
                    f"Progress: {progress}%\n{progress_bar}"
                )
                await msg.update()
            elif status == "completed":
                success_cnt = len(
                    [r for r in file_results if r.get("status") == "success"]
                )
                fail_cnt = len([r for r in file_results if r.get("status") == "error"])
                skip_cnt = len(
                    [r for r in file_results if r.get("status") == "skipped"]
                )
                msg.content = (
                    f"✅ **Upload complete** (ID: {upload_id[:8]})\n"
                    f"{message}\n\n"
                    f"📄 Success: {success_cnt} / Failure: {fail_cnt} / Skipped: {skip_cnt}\n"
                    f"💡 You can now ask questions about the document!"
                )
                await msg.update()
                # 예시 질문 자동 전송 (1회)
                entry = upload_manager.get_upload(upload_id)
                if entry and not entry.get("examples_sent"):
                    await send_example_questions(upload_id)
                    upload_manager.set_examples_sent(upload_id)
                break
            elif status == "error":
                msg.content = f"❌ **Upload failed** (ID: {upload_id[:8]})\n{message}"
                await msg.update()
                break
            else:
                msg.content = f"ℹ️ 알 수 없는 상태 ({status}) - 재시도 중..."
                await msg.update()

            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info(f"[upload:{upload_id}] 폴링 태스크 취소됨")
    except Exception as e:
        logger.error(f"[upload:{upload_id}] 폴링 중 예외: {e}")
        msg.content += f"\n\n⚠️ 상태 업데이트 중 오류 발생: {e}"
        await msg.update()
    finally:
        # 완료/오류/취소 시 registry 정리
        upload_manager.clear_task(upload_id)


async def send_example_questions(upload_id: str):
    """업로드 완료 후 문서 기반 예시 질문 1회 자동 전송"""
    entry = upload_manager.get_upload(upload_id)
    if not entry:
        return

    files = entry.get("files", [])
    # 세션에서 언어 가져오기
    settings = cl.user_session.get("settings")
    language = getattr(settings, "language", "ko-KR") if settings else "ko-KR"


def start_progress_tracker(upload_id: str, files: List[str], base_message: cl.Message):
    """비동기 폴링 태스크 시작 및 registry 저장"""
    if upload_manager.has_active_task(upload_id):
        logger.info(f"[upload:{upload_id}] 기존 폴링 태스크 재사용")
        return

    task = asyncio.create_task(poll_upload_status_loop(upload_id, base_message))
    upload_manager.add_upload(upload_id, files, base_message, task)


async def handle_file_upload(
    files,
    settings=None,
    document_type: str = "IR_REPORT",
    company: str = None,
    industry: str = None,
    report_year: str = None,
    force_upload: bool = False,
):
    """Unified file upload handler with aiohttp"""
    aiohttp_session = None

    try:
        # Initial upload message
        status_message = cl.Message(
            content="📤 **Uploading documents...**\n\nUploading your files to AI Search..."
        )
        await status_message.send()

        # Process and validate files
        files_payload = []
        valid_files = []
        MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
        allowed_extensions = {".pdf", ".docx", ".txt"}

        # ✅ aiohttp ClientSession 생성 (파일 다운로드 + 업로드용)
        timeout = aiohttp.ClientTimeout(total=None, connect=30.0, sock_read=60.0)
        aiohttp_session = aiohttp.ClientSession(
            timeout=timeout, connector=aiohttp.TCPConnector(limit=10)
        )

        for att in files:
            # Get filename
            filename = None
            if hasattr(att, "name"):
                filename = att.name
            elif hasattr(att, "filename"):
                filename = att.filename
            elif isinstance(att, dict) and "name" in att:
                filename = att["name"]
            else:
                filename = "unknown_file"

            logger.info(f"Processing file: {filename}")

            # Check file extension
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in allowed_extensions:
                await cl.Message(
                    content=f"❌ **Unsupported file format**: {filename}\n\nSupported formats: PDF, DOCX, TXT"
                ).send()
                continue

            file_bytes = None
            content_type = "application/octet-stream"

            # Get file content
            if hasattr(att, "content") and att.content:
                file_bytes = att.content
                content_type = getattr(
                    att, "mime", getattr(att, "content_type", content_type)
                )
            elif hasattr(att, "path") and att.path:
                # Read from file path
                try:
                    with open(att.path, "rb") as f:
                        file_bytes = f.read()
                except Exception as e:
                    await cl.Message(
                        content=f"❌ **File read failed**: {filename} - {e}"
                    ).send()
                    continue
            elif isinstance(att, dict) and ("content" in att or "data" in att):
                b64 = att.get("content") or att.get("data")
                try:
                    file_bytes = base64.b64decode(b64)
                except Exception:
                    file_bytes = b""
                content_type = att.get("content_type", content_type)
            elif hasattr(att, "url"):
                # ✅ aiohttp로 URL 다운로드
                url = getattr(att, "url")
                try:
                    async with aiohttp_session.get(url) as r:
                        r.raise_for_status()
                        file_bytes = await r.read()
                        content_type = r.headers.get("Content-Type", content_type)
                except aiohttp.ClientError as e:
                    await cl.Message(
                        content=f"❌ **File download failed**: {filename} - {e}"
                    ).send()
                    continue
            else:
                logger.warning(f"Cannot process file: {filename} - unsupported format")
                await cl.Message(
                    content=f"❌ **Unsupported file format**: {filename}"
                ).send()
                continue

            # Check if we got file content
            if not file_bytes:
                await cl.Message(content=f"❌ **Empty file**: {filename}").send()
                continue

            # Check file size
            if len(file_bytes) > MAX_FILE_SIZE:
                await cl.Message(
                    content=f"❌ **File size exceeded**: {filename}\n\nMaximum size: 50MB"
                ).send()
                continue

            # Add to upload payload
            files_payload.append((filename, BytesIO(file_bytes), content_type))
            valid_files.append(filename)
            logger.info(f"Added file to upload: {filename} ({len(file_bytes)} bytes)")

        if not files_payload:
            status_message.content = "❌ **No valid files to upload.**"
            await status_message.update()
            return False

        # Check file count limit
        if len(files_payload) > 10:
            status_message.content = (
                "❌ **File count exceeded**: You can only upload up to 10 files."
            )
            await status_message.update()
            return False

        # Update message with file list
        status_message.content = (
            f"📤 **Uploading files...**\n\n"
            f"Files to upload ({len(valid_files)}):\n"
            + "\n".join([f"• {f}" for f in valid_files])
        )
        await status_message.update()

        # Prepare form data with aiohttp
        data = aiohttp.FormData()
        data.add_field("document_type", document_type)
        data.add_field("company", company or "")
        data.add_field("industry", industry or "")
        data.add_field("report_year", report_year or "")
        data.add_field("force_upload", str(force_upload).lower())

        # Add files to form data
        for filename, file_data, content_type in files_payload:
            data.add_field(
                "files", file_data, filename=filename, content_type=content_type
            )

        # ✅ aiohttp로 multipart 업로드 (긴 타임아웃)
        try:
            upload_timeout = aiohttp.ClientTimeout(
                total=None,
                connect=30.0,
                sock_read=180.0,  # 업로드는 더 긴 타임아웃
            )

            async with aiohttp_session.post(
                UPLOAD_API_URL, data=data, timeout=upload_timeout
            ) as resp:
                if resp.status == 200:
                    try:
                        resp_json = await resp.json()
                        upload_id = resp_json.get("upload_id")

                        if upload_id:
                            # Start tracking upload status
                            start_progress_tracker(
                                upload_id, valid_files, status_message
                            )
                            return True
                        else:
                            message = resp_json.get("message", "upload complete")
                            status_message.content = (
                                f"✅ **Upload response**: {message}"
                            )
                            await status_message.update()
                            return True

                    except Exception as e:
                        resp_text = await resp.text()
                        status_message.content = f"✅ **Upload complete**: {resp_text}"
                        await status_message.update()
                        return True
                else:
                    resp_text = await resp.text()
                    status_message.content = (
                        f"❌ **Upload failed**: {resp.status} - {resp_text}"
                    )
                    await status_message.update()
                    return False

        except asyncio.TimeoutError as e:
            status_message.content = f"❌ **Upload timeout**: {e}"
            await status_message.update()
            logger.error(f"Upload timeout: {e}")
            return False
        except aiohttp.ClientError as e:
            status_message.content = f"❌ **Upload HTTP error**: {e}"
            await status_message.update()
            logger.error(f"Upload HTTP error: {e}")
            return False

    except Exception as e:
        await cl.Message(content=f"❌ **Upload error**: {str(e)}").send()
        logger.error(f"Upload error: {e}")
        return False

    finally:
        # ✅ aiohttp client 정리
        if aiohttp_session and not aiohttp_session.closed:
            try:
                await aiohttp_session.close()
                await asyncio.sleep(0.1)  # Graceful cleanup
                logger.info("🔌 Upload aiohttp client closed successfully")
            except Exception as e:
                logger.error(f"❌ Error closing upload client: {e}")


@cl.set_chat_profiles
async def chat_profile():
    """Set up chat profiles for different languages"""
    return [
        cl.ChatProfile(
            name="English",
            markdown_description="## Multi-Agent Doc Research",
            icon="/public/images/azure-ai-search.png",
            starters=get_starters_for_language("en-US"),
        ),
        cl.ChatProfile(
            name="Korean",
            markdown_description="## Multi-Agent Doc Research",
            icon="/public/images/azure-ai-search.png",
            starters=get_starters_for_language("ko-KR"),
        ),
    ]


@cl.on_chat_start
async def start():
    """Initialize chat session with user welcome"""
    logger.info("🚀 Chat session starting...")
    settings = ChatSettings()

    # Ensure the settings panel is re-sent for every new session
    cl.user_session.set("chat_settings_sent", False)

    # Enable file upload UI - this is CRITICAL for settings icon to appear
    cl.user_session.set(
        "files",
        {
            "accept": {
                "application/pdf": [".pdf"],
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [
                    ".docx"
                ],
                "text/plain": [".txt"],
            },
            "max_size_mb": 50,
            "max_files": 10,
        },
    )

    # Get current chat profile with robust fallback
    profile = cl.user_session.get("chat_profile")
    logger.info(f"📋 Chat profile detected: {profile}")

    # Determine language based on profile (English is first/default)
    if profile == "Korean":
        language = "ko-KR"
    else:
        language = "en-US"  # Default to English

    logger.info(f"🌍 Language set to: {language}")

    # Initialize chat settings
    settings.language = language
    cl.user_session.set("settings", settings)

    # Allow the WebSocket to settle before sending UI components
    await asyncio.sleep(0.35)

    # ✨ 설정 UI 먼저 전송 (가장 우선순위)
    await ensure_chat_settings_ui(language, force=True)

    # Schedule a background retry if the panel is still missing
    if not cl.user_session.get("chat_settings_sent"):
        asyncio.create_task(_delayed_settings_retry(language))

    # 사용자 정보 가져오기
    user = cl.user_session.get("user")

    # 사용자 환영 메시지
    if user:
        user_role = user.metadata.get("role", "user")

        # 관리자 권한이 있는 경우 추가 메시지
        if user_role == "admin":
            await cl.Message(
                content="🔧 **Admin Access Granted**\nYou have administrator privileges."
            ).send()

    # ✨ 그 다음 starter actions 표시
    await send_starters_as_actions(language)


@cl.on_chat_resume
async def on_resume():
    """Resend settings when the client reconnects after a dropped socket."""
    logger.info("🔄 Chat session resuming...")

    settings = cl.user_session.get("settings")
    if not settings:
        settings = ChatSettings()
        cl.user_session.set("settings", settings)

    language = get_user_language(settings)
    logger.info(f"🌍 Resuming with language: {language}")

    # Reset and re-send the settings panel on every resume
    cl.user_session.set("chat_settings_sent", False)

    await asyncio.sleep(0.5)

    # 재연결 시 설정 UI 다시 전송
    await ensure_chat_settings_ui(language, force=True)

    if not cl.user_session.get("chat_settings_sent"):
        asyncio.create_task(_delayed_settings_retry(language))


@cl.on_settings_update
async def setup_agent(settings_dict: Dict[str, Any]):
    """Simplified settings update"""
    settings = cl.user_session.get("settings")

    # Update settings with simple mapping
    for key, value in settings_dict.items():
        if hasattr(settings, key):
            setattr(settings, key, value)

    # Handle special cases
    if "search_engine" in settings_dict:
        search_engine_name = settings_dict["search_engine"]
        settings.search_engine = SEARCH_ENGINES.get(
            search_engine_name, list(SEARCH_ENGINES.values())[0]
        )

    # Check if user wants to show starters
    show_starters = settings_dict.get("show_starters", False)
    if show_starters:
        language = get_chat_profile_language()
        await send_starters_as_actions(language)

    cl.user_session.set("settings", settings)
    await cl.Message(content="⚙️ Settings updated successfully!").send()


async def stream_chat_with_api(message: str, settings: ChatSettings) -> None:
    """Stream-enabled chat function with aiohttp for production-grade long-running connections"""
    if not message or message.strip() == "":
        return

    # Prepare the API payload
    payload = create_api_payload(settings)

    # Create message for streaming response
    ui_text = UI_TEXT[settings.language]
    msg = cl.Message(content="")
    await msg.send()

    # ✅ aiohttp 설정: 장시간 연결 유지에 최적화 (10분으로 증가)
    timeout = aiohttp.ClientTimeout(
        total=None,  # 무제한 (multi-agent는 예측 불가)
        connect=10,  # 연결 시작 10초
        sock_connect=10,  # 소켓 연결 10초
        sock_read=600,  # ✅ 소켓 읽기 600초 (10분) - Magentic 처리 시간 고려
    )

    # ✅ 백오프 재시도 설정
    max_retries = 3
    retry_delays = [1, 2, 4]

    # Helper function to clean text content
    def clean_response_text(text: str) -> str:
        cleaned_text = text.replace("~~", "==")
        return cleaned_text

    retry_count = 0
    last_error = None

    while retry_count <= max_retries:
        # ✅ 재시도마다 새로운 stop_event 생성 (핵심 수정!)
        aiohttp_session = None
        keepalive_task = None
        stop_event = asyncio.Event()  # 매 재시도마다 새로 생성
        last_activity = None  # 초기화

        # ✅ 개선된 Keepalive sender
        async def keepalive_sender():
            """Monitor SSE activity instead of probing WebSocket"""
            nonlocal last_activity
            try:
                while not stop_event.is_set():
                    await asyncio.sleep(30)
                    if not stop_event.is_set() and last_activity:
                        current_time = asyncio.get_event_loop().time()
                        idle_time = current_time - last_activity

                        # ✅ SSE 데이터가 120초(2분) 이상 안 오면 warning (10분 대기)
                        if idle_time > 120:
                            logger.warning(
                                f"⚠️ No SSE data for {idle_time:.0f} seconds (Magentic may be processing)"
                            )
                        else:
                            logger.debug(
                                f"✅ SSE active (last data {idle_time:.0f}s ago)"
                            )
            except asyncio.CancelledError:
                logger.debug("Keepalive sender cancelled")

        try:
            # ✅ aiohttp ClientSession 생성
            aiohttp_session = aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(
                    limit=100,
                    limit_per_host=30,
                    ttl_dns_cache=300,
                    keepalive_timeout=600,  # ✅ 10분으로 증가
                ),
            )

            # ✅ Start keepalive task
            keepalive_task = asyncio.create_task(keepalive_sender())

            logger.info(
                f"🌐 Connecting to API (attempt {retry_count + 1}/{max_retries + 1})"
            )

            # ✅ POST request with streaming
            async with aiohttp_session.post(
                SK_API_URL,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "Connection": "keep-alive",
                    "Cache-Control": "no-cache",
                },
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"API returned status {response.status}: {error_text}"
                    )

                logger.info(f"✅ Connected to API - Status: {response.status}")

                # Check content type
                content_type = response.headers.get("content-type", "")

                if "text/event-stream" in content_type:
                    # ✅ Process SSE with tool calling steps
                    accumulated_content = ""
                    current_tool_step = None
                    tool_steps = {}
                    last_activity = asyncio.get_event_loop().time()
                    last_keepalive_log = asyncio.get_event_loop().time()
                    chunk_count = 0

                    # ✅ Buffer for incomplete lines
                    buffer = ""

                    logger.info("Starting SSE processing loop with aiohttp...")

                    # ✅ Process SSE stream chunk by chunk and parse lines
                    async for chunk in response.content.iter_any():
                        # ✅ 연결 끊김 체크
                        if stop_event.is_set():
                            logger.warning(
                                "⚠️ Stop event triggered - stopping stream processing"
                            )
                            break

                        # 주기적 로깅
                        current_time = asyncio.get_event_loop().time()
                        if current_time - last_keepalive_log > 30:  # 30초마다 로깅
                            logger.info(
                                f"💓 SSE connection alive (stream active, last data {current_time - last_activity:.0f}s ago)"
                            )
                            last_keepalive_log = current_time

                        last_activity = current_time
                        chunk_count += 1

                        try:
                            chunk_text = chunk.decode("utf-8")
                        except UnicodeDecodeError:
                            logger.warning("⚠️ Failed to decode chunk - skipping")
                            continue

                        # Add to buffer
                        buffer += chunk_text

                        # Process complete lines
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()

                            if not line:
                                continue

                            logger.debug(f"SSE line received: {line[:100]}...")

                            # ✅ Handle SSE comments (keepalive from backend)
                            if line.startswith(":"):
                                logger.debug(f"SSE comment (backend keepalive): {line}")
                                continue

                            # Handle SSE format (data: prefix)
                            if line.startswith("data: "):
                                line = line[6:].strip()

                                # Check for stream end marker
                                if line == "[DONE]":
                                    logger.info("✅ Stream completed successfully")
                                    break

                                # ✅ Status message handling - create tool steps
                                if line.startswith("### "):
                                    step_content = line[4:]

                                    # Complete previous step
                                    if current_tool_step:
                                        current_tool_step.output = "✅ Completed"
                                        await safe_send_step(current_tool_step)

                                    # Decode step content
                                    step_name, code_content, description = (
                                        decode_step_content(
                                            step_content, step_name_manager
                                        )
                                    )

                                    # ✅ Determine step type and icon
                                    step_type = "tool"
                                    step_icon = "🔧"

                                    original_name_lower = step_content.lower()
                                    logger.info(f"Creating tool step: {step_name}")

                                    try:
                                        if original_name_lower.startswith(
                                            ui_text.get("analyzing", "").lower()
                                        ):
                                            step_type = "intent"
                                            step_icon = "🧠"
                                        elif original_name_lower.startswith(
                                            ui_text.get("analyze_complete", "").lower()
                                        ):
                                            step_type = "intent"
                                            step_icon = "🧠"
                                        elif original_name_lower.startswith(
                                            ui_text.get("task_planning", "").lower()
                                        ):
                                            step_type = "planning"
                                            step_icon = "📋"
                                        elif original_name_lower.startswith(
                                            ui_text.get("plan_done", "").lower()
                                        ):
                                            step_type = "planning"
                                            step_icon = "📋"
                                        elif original_name_lower.startswith(
                                            ui_text.get("searching", "").lower()
                                        ):
                                            step_type = "retrieval"
                                            step_icon = "🌐"
                                        elif original_name_lower.startswith(
                                            ui_text.get("search_done", "").lower()
                                        ):
                                            step_type = "retrieval"
                                            step_icon = "🌐"
                                        elif original_name_lower.startswith(
                                            ui_text.get("searching_YouTube", "").lower()
                                        ):
                                            step_type = "retrieval"
                                            step_icon = "🎬"
                                        elif original_name_lower.startswith(
                                            ui_text.get("YouTube_done", "").lower()
                                        ):
                                            step_type = "retrieval"
                                            step_icon = "🎬"
                                        elif original_name_lower.startswith(
                                            ui_text.get(
                                                "searching_ai_search", ""
                                            ).lower()
                                        ):
                                            step_type = "retrieval"
                                            step_icon = "🏁"
                                        elif original_name_lower.startswith(
                                            ui_text.get(
                                                "ai_search_context_done", ""
                                            ).lower()
                                        ):
                                            step_type = "retrieval"
                                            step_icon = "🏁"
                                        elif original_name_lower.startswith(
                                            ui_text.get("answering", "").lower()
                                        ):
                                            step_type = "llm"
                                            step_icon = "👨‍💻"
                                        elif original_name_lower.startswith(
                                            ui_text.get("start_research", "").lower()
                                        ):
                                            step_type = "research"
                                            step_icon = "✏️"
                                        elif original_name_lower.startswith(
                                            ui_text.get("organize_research", "").lower()
                                        ):
                                            step_type = "research"
                                            step_icon = "✏️"
                                        elif original_name_lower.startswith(
                                            ui_text.get("write_research", "").lower()
                                        ):
                                            step_type = "research"
                                            step_icon = "✏️"
                                        elif original_name_lower.startswith(
                                            ui_text.get("review_research", "").lower()
                                        ):
                                            step_type = "research"
                                            step_icon = "✏️"
                                        elif (
                                            "context information" in original_name_lower
                                        ):
                                            step_type = "tool"
                                            step_icon = "📃"
                                    except KeyError as e:
                                        logger.warning(f"Missing UI text key: {e}")

                                    # ✅ Create new step with icon
                                    current_tool_step = cl.Step(
                                        name=f"{step_icon} {step_name}", type=step_type
                                    )

                                    # Set input
                                    if code_content:
                                        current_tool_step.input = (
                                            f"```python\n{code_content}\n```"
                                        )
                                    elif description:
                                        current_tool_step.input = description
                                    else:
                                        current_tool_step.input = (
                                            f"Executing: {step_name}"
                                        )

                                    if not await safe_send_step(current_tool_step):
                                        logger.warning(
                                            f"Failed to send tool step: {step_name}"
                                        )
                                        break

                                    tool_steps[step_name] = current_tool_step

                                else:
                                    # ✅ Regular content
                                    cleaned_line = clean_response_text(line)

                                    if accumulated_content:
                                        if cleaned_line.startswith(
                                            ("•", "-", "#", "1.", "2.", "3.")
                                        ) or accumulated_content.endswith(
                                            (".", "!", "?", ":")
                                        ):
                                            accumulated_content += "\n\n" + cleaned_line
                                        else:
                                            accumulated_content += "\n" + cleaned_line
                                    else:
                                        accumulated_content = cleaned_line

                                    # Stream to UI
                                    if not await safe_stream_token(
                                        msg, cleaned_line + "\n"
                                    ):
                                        logger.warning("Stream connection lost")
                                        break

                            else:
                                # ✅ Regular content without 'data:' prefix
                                cleaned_line = clean_response_text(line)

                                if accumulated_content:
                                    if cleaned_line.startswith(
                                        ("•", "-", "#", "1.", "2.", "3.")
                                    ) or accumulated_content.endswith(
                                        (".", "!", "?", ":")
                                    ):
                                        accumulated_content += "\n\n" + cleaned_line
                                    else:
                                        accumulated_content += "\n" + cleaned_line
                                else:
                                    accumulated_content = cleaned_line

                                if not await safe_stream_token(
                                    msg, cleaned_line + "\n"
                                ):
                                    logger.warning("Stream connection lost")
                                    break

                    # ✅ Close remaining step
                    if current_tool_step:
                        current_tool_step.output = "✅ Completed"
                        await safe_send_step(current_tool_step)

                    logger.info(
                        f"✅ Stream processed successfully - {chunk_count} chunks, {len(tool_steps)} tool steps"
                    )

                else:
                    # ✅ Handle non-streaming response
                    logger.info("Not a streaming response, reading full content")
                    try:
                        response_text = await response.text()
                        cleaned_response = clean_response_text(response_text)

                        # Try JSON parse
                        try:
                            response_data = json.loads(response_text)
                            if (
                                isinstance(response_data, dict)
                                and "content" in response_data
                            ):
                                cleaned_content = clean_response_text(
                                    response_data["content"]
                                )
                                await safe_stream_token(msg, cleaned_content)
                            else:
                                await safe_stream_token(msg, cleaned_response)
                        except json.JSONDecodeError:
                            await safe_stream_token(msg, cleaned_response)
                    except Exception as e:
                        error_msg = f"Error processing response: {str(e)}"
                        await safe_stream_token(msg, error_msg)
                        logger.error(error_msg)

                # ✅ 성공 - 루프 탈출
                break  # 성공 시 재시도 루프 종료

        except aiohttp.ClientError as e:
            last_error = e
            logger.error(f"❌ aiohttp client error (attempt {retry_count + 1}): {e}")

            if retry_count < max_retries:
                delay = retry_delays[retry_count]
                logger.info(f"🔄 Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                retry_count += 1
            else:
                await handle_error_response(
                    msg,
                    "Connection Error",
                    f"Failed to connect to API after {max_retries + 1} attempts: {str(e)}",
                )
                break

        except asyncio.TimeoutError as e:
            last_error = e
            logger.error(f"❌ Timeout error (attempt {retry_count + 1}): {e}")

            if retry_count < max_retries:
                delay = retry_delays[retry_count]
                logger.info(f"🔄 Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                retry_count += 1
            else:
                await handle_error_response(
                    msg,
                    "Timeout Error",
                    f"Request timed out after {max_retries + 1} attempts",
                )
                break

        except Exception as e:
            last_error = e
            logger.error(f"❌ Unexpected error (attempt {retry_count + 1}): {e}")

            if retry_count < max_retries:
                delay = retry_delays[retry_count]
                logger.info(f"🔄 Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                retry_count += 1
            else:
                await handle_error_response(msg, "Processing Error", str(e))
                break

        finally:
            # ✅ Cleanup for this attempt
            if keepalive_task and not keepalive_task.done():
                stop_event.set()
                try:
                    keepalive_task.cancel()
                    await asyncio.wait_for(keepalive_task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

            if aiohttp_session and not aiohttp_session.closed:
                await aiohttp_session.close()
                # ✅ Give time for proper cleanup
                await asyncio.sleep(0.25)

    # ✅ Final message update
    await safe_update_message(msg)
    logger.info("Streaming completed")


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    logger.info(
        f"📨 Message received - Content: '{message.content[:50] if message.content else 'EMPTY'}...'"
    )

    language = get_chat_profile_language()

    await ensure_chat_settings_ui(language, force=True)

    settings = cl.user_session.get("settings")
    if not settings:
        settings = ChatSettings()
        cl.user_session.set("settings", settings)

    # 🔧 FIX: Update settings.language with current chat profile language
    settings.language = language
    cl.user_session.set("settings", settings)

    # ✨ 최우선: 빈 메시지 체크 (스타터 클릭으로 인한 빈 메시지 필터링)
    if not message.content or message.content.strip() == "":
        logger.info("📭 Empty message received - checking for file attachments only")

        # 파일이 첨부되었는지 확인
        attachments = (
            getattr(message, "elements", None)
            or getattr(message, "files", None)
            or getattr(message, "attachments", None)
        )

        if attachments:
            logger.info(f"📎 File attachments found: {len(attachments)} files")
            # 파일만 있는 경우 업로드 처리
            await handle_file_upload(attachments, settings)
        else:
            logger.info(
                "🚫 No content and no attachments - ignoring message completely"
            )

        return  # ✨ 여기서 완전히 종료 - 백엔드 호출 없음

    logger.info(f"✅ Valid message with content - processing...")

    # Check for file attachments with text content
    attachments = (
        getattr(message, "elements", None)
        or getattr(message, "files", None)
        or getattr(message, "attachments", None)
    )

    # ✨ 파일 첨부가 없는 경우에만 guide-only 스타터 체크
    if not attachments:
        guide_only_category = find_starter_category_for_prompt(
            language, message.content
        )
        if guide_only_category:
            logger.info(
                f"🛑 Guide-only starter prompt detected for category '{guide_only_category}'. "
                "Skipping backend call - message already displayed."
            )
            return

    uploaded = None

    if attachments:
        logger.info(
            f"📎 Processing {len(attachments)} file attachments with text content"
        )
        uploaded = await handle_file_upload(attachments, settings)
        # If only files (no textual content), stop here
        # This should never happen due to the check above, but keep for safety
        if (not message.content) or (message.content.strip() == ""):
            logger.warning("⚠️ Content became empty after file upload - stopping")
            return

    message_content = message.content

    # Provide light feedback if user sent text along with freshly uploaded files
    if uploaded and message_content:
        logger.info("📎 Files uploaded with text content - processing both")
        await cl.Message(
            content="📎 첨부한 파일을 처리한 후 답변을 생성합니다..."
        ).send()
        await stream_chat_with_api(message.content, settings)
        return  # ✨ 중복 호출 방지

    # Process the message with streaming (only if we reach here)
    logger.info("💬 Processing text message without files")
    await stream_chat_with_api(message.content, settings)


# ============================================================================
# Action Callbacks - 통합 및 간소화
# ============================================================================


@cl.action_callback("clear_chat")
async def on_action(action: cl.Action):
    """Handle clear chat action"""
    # Clear the chat context
    cl.chat_context.clear()

    # Send confirmation
    await cl.Message(content="Chat history cleared!").send()

    # Return success
    return "Chat cleared successfully"


@cl.action_callback("help_action")
async def on_help_action(action: cl.Action):
    """Handle help action"""
    settings = cl.user_session.get("settings")
    language = getattr(settings, "language", "ko-KR") if settings else "ko-KR"
    ui_text = UI_TEXT[language]
    help_message = ui_text.get("starter_message", "Help information not available")

    await cl.Message(content=help_message).send()
    return "Help displayed"


@cl.action_callback("show_starters_action")
async def on_show_starters_action(action: cl.Action):
    """Handle show starters action"""
    language = get_chat_profile_language()
    await send_starters_as_actions(language)
    return "Starters displayed"


@cl.action_callback("starter_0")
@cl.action_callback("starter_1")
@cl.action_callback("starter_2")
async def on_starter_action(action: cl.Action):
    """Unified Starter Action Handler - Handles All Starters"""
    # Payload에서 정보 추출
    language = get_chat_profile_language()
    ensure_chat_settings_ui(language=language, force=True)

    message_content = action.payload.get("message", "")
    starter_label = action.payload.get("label", "Unknown")
    starter_index = action.payload.get("index", 0)
    category = action.payload.get("category", "")
    send_to_backend = action.payload.get("send_to_backend", True)

    logger.info("=" * 60)
    logger.info(f"🎬 Starter action triggered: {starter_label}")
    logger.info(f"   - Index: {starter_index}, Category: {category}")
    logger.info(f"   - Send to backend: {send_to_backend}")
    logger.info(
        f"   - Message length: {len(message_content) if message_content else 0}"
    )
    logger.info(
        f"   - Message preview: '{message_content[:50] if message_content else 'EMPTY'}...'"
    )
    logger.info("=" * 60)

    # ✨ send_to_backend에 따라 분기 처리
    if not send_to_backend:
        # 가이드 메시지만 표시 (백엔드 호출 없음)
        logger.info(f"📋 Showing guide message for '{category}' (no backend call)")
        language = get_user_language()
        guide_message = get_starter_prompt(language, category)

        if guide_message:
            await cl.Message(content=guide_message).send()
            logger.info(f"✅ Displayed guide message for '{category}'")
        else:
            # 폴백 메시지 (EXAMPLE_PROMPTS에서 찾지 못한 경우)
            logger.warning(
                f"⚠️ No guide message found for '{category}' in language '{language}'"
            )
            if language == "ko-KR":
                fallback = f"ℹ️ **{starter_label}** 가이드를 찾을 수 없습니다."
            else:
                fallback = f"ℹ️ Cannot find guide for **{starter_label}**."
            await cl.Message(content=fallback).send()

        return f"Guide displayed for {starter_label}"

    # 백엔드로 전송해야 하는 경우
    logger.info(f"🚀 Processing starter with backend call: {starter_label}")

    # 빈 메시지 검증
    if not message_content or message_content.strip() == "":
        logger.warning(
            f"⚠️ Empty message for backend starter '{starter_label}' - aborting"
        )
        language = get_user_language()

        if language == "ko-KR":
            await cl.Message(
                content=f"⚠️ **{starter_label}**: 메시지가 비어있습니다."
            ).send()
        else:
            await cl.Message(content=f"⚠️ **{starter_label}**: Message is empty.").send()
        return f"Empty message for {starter_label}"

    # 사용자 메시지를 채팅 히스토리에 추가
    user_message = cl.Message(
        author="User", content=message_content, type="user_message"
    )
    await user_message.send()

    # 설정 가져오기 또는 생성
    settings = cl.user_session.get("settings")
    if not settings:
        settings = ChatSettings()
        cl.user_session.set("settings", settings)

    # 백엔드 API 호출
    logger.info(f"📤 Sending to backend: {len(message_content)} characters")
    await stream_chat_with_api(message_content, settings)

    logger.info(f"✅ Processed starter: {starter_label}")
    return f"Processed starter: {starter_label}"


@cl.action_callback("check_upload_status")
async def on_check_upload_status(action: cl.Action):
    """모든 활성 업로드의 최신 상태 요약 출력"""
    all_uploads = upload_manager.get_all_uploads()

    if not all_uploads:
        await cl.Message(content="📋 **현재 진행 중인 업로드가 없습니다.**").send()
        return "No active uploads"

    lines = ["📊 **현재 진행 중인 업로드 목록**\n"]
    for upload_id, info in all_uploads.items():
        # 메시지 객체의 최신 content 일부 활용
        msg_obj = info.get("message")
        preview = ""
        if msg_obj and getattr(msg_obj, "content", None):
            preview = msg_obj.content.splitlines()[0][:60]
        lines.append(f"• {upload_id[:8]} ({', '.join(info['files'])})")
        if preview:
            lines.append(f"  ↳ {preview}")

    await cl.Message(content="\n".join(lines)).send()
    return "Listed active uploads"


def _create_settings_components(language: str) -> list:
    """Create settings UI components for the given language"""
    ui_text = UI_TEXT.get(language, UI_TEXT.get("en-US", {}))

    return [
        cl.input_widget.Switch(
            id="research",
            label=ui_text.get("research_title", "Research"),
            initial=True,
            tooltip=ui_text.get("research_desc", ""),
        ),
        cl.input_widget.Switch(
            id="planning",
            label=ui_text.get("planning_title", "Planning"),
            initial=False,
            tooltip=ui_text.get("planning_desc", ""),
        ),
        cl.input_widget.Switch(
            id="ai_search",
            label=ui_text.get("ai_search_title", "AI Search"),
            initial=True,
            tooltip=ui_text.get("ai_search_desc", ""),
        ),
        cl.input_widget.Select(
            id="multi_agent_type",
            label=ui_text.get("multi_agent_type_title", "Multi-Agent Type"),
            initial_index=0,
            values=list(MULTI_AGENT_TYPES.keys()),
            tooltip=ui_text.get("multi_agent_type_desc", ""),
        ),
        cl.input_widget.Switch(
            id="verbose",
            label=ui_text.get("verbose_title", "Verbose"),
            initial=True,
            tooltip=ui_text.get("verbose_desc", ""),
        ),
        cl.input_widget.Switch(
            id="show_starters",
            label="📋 Show Quick Start Options",
            initial=False,
            tooltip="Toggle to show/hide quick start prompts",
        ),
        cl.input_widget.Slider(
            id="max_tokens",
            label="Max Tokens",
            initial=4000,
            min=1000,
            max=8000,
            step=500,
            tooltip="Maximum number of tokens in response",
        ),
        cl.input_widget.Slider(
            id="temperature",
            label="Temperature",
            initial=0.7,
            min=0.0,
            max=1.0,
            step=0.1,
            tooltip="Controls randomness in response generation",
        ),
    ]


async def _send_settings_once(language: str):
    """Send settings UI once (used by retry logic)"""
    settings_components = _create_settings_components(language)
    await cl.ChatSettings(settings_components).send()
    cl.user_session.set("chat_settings_sent", True)
    await asyncio.sleep(0.15)  # Stabilization delay


async def ensure_chat_settings_ui(language: str, force: bool = False):
    """Guarantee the settings panel appears even after reconnects."""
    already_sent = cl.user_session.get("chat_settings_sent", False)
    if already_sent and not force:
        logger.info("⚙️ Chat settings already sent, skipping...")
        return

    logger.info(f"⚙️ Sending chat settings UI for language: {language}")

    # Use retry utility
    success, result = await retry_async_operation(
        _send_settings_once,
        language,
        max_retries=3,
        initial_delay=0.2,
        backoff_factor=2.0,
    )

    if success:
        logger.info("✅ Chat settings UI sent successfully")
    else:
        logger.error(f"❌ Failed to send chat settings: {result}")
        cl.user_session.set("chat_settings_sent", False)


async def _delayed_settings_retry(
    language: str, delay: float = 1.0, max_attempts: int = 2
):
    """Background retry to ensure the settings panel eventually appears."""
    for attempt in range(max_attempts):
        await asyncio.sleep(delay * (attempt + 1))
        logger.info(f"🔁 Delayed settings retry attempt {attempt + 1}/{max_attempts}")
        await ensure_chat_settings_ui(language, force=True)
        if cl.user_session.get("chat_settings_sent"):
            logger.info("✅ Delayed retry succeeded – settings panel is visible.")
            return
    logger.error("❌ Delayed retries exhausted – settings panel still missing.")
