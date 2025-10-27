import os
import logging
import json
from typing import Annotated, Optional, Dict, List, Any
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from dotenv import load_dotenv
from datetime import datetime
from jinja2 import Template

# MCP 클라이언트 라이브러리 import
try:
    import mcp
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    import mcp.types as types

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logging.warning(
        "MCP 라이브러리가 설치되지 않았습니다. pip install mcp를 실행하세요."
    )

load_dotenv(override=True)

# YouTube MCP 서버 설정
YOUTUBE_MCP_SERVER_COMMAND = "youtube-data-mcp-server"
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
TIME_ZONE = os.getenv("TIME_ZONE", "Asia/Seoul")

logger = logging.getLogger(__name__)


# YouTube MCP 플러그인: 순서별로 정리
class YouTubeMCPPlugin:
    """YouTube 검색을 위한 MCP 플러그인 (YouTube MCP 서버 연결)"""

    def __init__(self):
        # 연결 객체
        self.mcp_client = None
        self.mcp_session = None
        self.max_results = 10

        # MCP 서버 연결 확인
        if not MCP_AVAILABLE:
            logger.warning(
                "MCP 라이브러리가 설치되지 않았습니다. YouTube MCP 기능이 비활성화됩니다."
            )
            return

        if not YOUTUBE_API_KEY:
            logger.warning(
                "YOUTUBE_API_KEY 환경 변수가 설정되지 않았습니다. YouTube 기능이 비활성화됩니다."
            )
            return

        logger.info("YouTube MCP Plugin 초기화 완료.")

    # =====================================
    # 연결 관리 메서드
    # =====================================

    async def _ensure_mcp_connection(self):
        """MCP 서버 연결 및 도구 목록 확인"""
        if self.mcp_session is None and MCP_AVAILABLE:
            try:
                # 기존 연결이 있다면 정리
                await self.cleanup()

                server_params = StdioServerParameters(
                    command=YOUTUBE_MCP_SERVER_COMMAND,
                    args=["--api-key", YOUTUBE_API_KEY] if YOUTUBE_API_KEY else [],
                    env={"YOUTUBE_API_KEY": YOUTUBE_API_KEY} if YOUTUBE_API_KEY else {},
                )

                # 새로운 연결 생성
                self.mcp_client = stdio_client(server_params)
                read, write = await self.mcp_client.__aenter__()
                self.mcp_session = ClientSession(read, write)
                await self.mcp_session.__aenter__()
                await self.mcp_session.initialize()

                # 도구 목록 로깅
                tools = await self.mcp_session.list_tools()
                tool_names = [tool.name for tool in tools.tools] if tools.tools else []
                logger.info(
                    f"YouTube MCP 서버에 연결됨. 사용 가능한 도구: {tool_names}"
                )

            except Exception as e:
                logger.error(f"MCP 서버 연결 실패: {e}")
                # 연결 실패 시 정리
                await self.cleanup()

        return self.mcp_session is not None

    # =====================================
    # 핵심 검색 메서드
    # =====================================

    async def _search_youtube_videos(
        self, query: str, max_results: int = 10
    ) -> Dict[str, Any]:
        """searchVideos MCP 도구로 YouTube 비디오 검색"""
        try:
            if not await self._ensure_mcp_connection():
                return {
                    "status": "error",
                    "message": "MCP 서버에 연결할 수 없습니다.",
                    "videos": [],
                }

            # searchVideos 도구 사용
            search_result = await self.mcp_session.call_tool(
                "searchVideos", {"query": query, "maxResults": max_results}
            )

            if getattr(search_result, "is_err", False):
                logger.error(f"YouTube 검색 오류: {search_result.content}")
                return {
                    "status": "error",
                    "message": f"검색 실패: {search_result.content}",
                    "videos": [],
                }

            # 결과 파싱
            videos_data = []
            if hasattr(search_result, "content") and search_result.content:
                try:
                    # MCP 응답 구조 안전하게 처리
                    if isinstance(search_result.content, list):
                        content_text = (
                            search_result.content[0].text
                            if search_result.content
                            else ""
                        )
                    else:
                        content_text = search_result.content

                    # 응답 파싱 (직접 배열 형태)
                    result_data = json.loads(content_text)

                    # 결과가 리스트인지 확인
                    if isinstance(result_data, list):
                        items = result_data
                    elif isinstance(result_data, dict) and "items" in result_data:
                        items = result_data["items"]
                    else:
                        logger.warning(f"예상과 다른 응답 형식: {type(result_data)}")
                        items = []

                    for item in items:
                        if not isinstance(item, dict):
                            continue

                        # YouTube API 표준 응답 구조에 맞춰 파싱
                        video_id = ""
                        if isinstance(item.get("id"), dict):
                            video_id = item.get("id", {}).get("videoId", "")
                        elif isinstance(item.get("id"), str):
                            video_id = item.get("id", "")

                        snippet = item.get("snippet", {})

                        video_info = {
                            "videoId": video_id,
                            "title": snippet.get("title", ""),
                            "description": snippet.get("description", ""),
                            "channelTitle": snippet.get("channelTitle", ""),
                            "publishedAt": snippet.get("publishedAt", ""),
                            "thumbnails": snippet.get("thumbnails", {}),
                        }
                        videos_data.append(video_info)

                except json.JSONDecodeError as e:
                    logger.error(f"JSON 파싱 오류: {e}")
                    return {
                        "status": "error",
                        "message": "응답 데이터 파싱 실패",
                        "videos": [],
                    }
                except Exception as e:
                    logger.error(f"결과 파싱 중 오류: {e}")
                    logger.debug(f"원본 응답: {search_result.content}")
                    return {
                        "status": "error",
                        "message": f"결과 파싱 실패: {str(e)}",
                        "videos": [],
                    }

            return {
                "status": "success",
                "videos": videos_data,
                "search_query": query,
                "total_results": len(videos_data),
            }

        except Exception as e:
            logger.error(f"YouTube 검색 중 오류 발생: {e}")
            # 연결 문제로 인한 오류일 수 있으므로 연결 정리
            try:
                await self.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"오류 후 cleanup 실패: {cleanup_error}")
            return {
                "status": "error",
                "message": f"검색 중 오류가 발생했습니다: {str(e)}",
                "videos": [],
            }

        except Exception as e:
            logger.error(f"YouTube 검색 중 오류 발생: {e}")
            return {
                "status": "error",
                "message": f"검색 중 오류가 발생했습니다: {str(e)}",
                "videos": [],
            }

    # =====================================
    # 결과 포맷팅 메서드
    # =====================================

    def _format_youtube_results(self, videos: List[Dict[str, Any]]) -> str:
        """검색 결과 포맷팅"""
        if not videos:
            return "❌ 검색 결과가 없습니다."

        formatted_results = []
        for i, video in enumerate(videos, 1):
            result_text = f"""
{i}. **{video.get('title', 'N/A')}**
   📺 채널: {video.get('channelTitle', 'N/A')}
   📅 게시일: {video.get('publishedAt', 'N/A')}
   🔗 링크: https://www.youtube.com/watch?v={video.get('videoId', '')}
   📝 설명: {video.get('description', 'N/A')[:150]}...
            """.strip()
            formatted_results.append(result_text)

        return "\n\n".join(formatted_results)

    def _create_video_context(self, videos: List[Dict]) -> str:
        """비디오 정보를 컨텍스트 형태로 변환"""
        context_parts = []

        for i, video in enumerate(videos[:5], 1):  # 상위 5개만 사용
            video_context = f"""
비디오 {i}:
- 제목: {video.get('title', 'N/A')}
- 채널: {video.get('channelTitle', 'N/A')}
- 설명: {video.get('description', 'N/A')[:200]}...
- URL: https://www.youtube.com/watch?v={video.get('videoId', '')}
- 게시일: {video.get('publishedAt', 'N/A')}
            """.strip()
            context_parts.append(video_context)

        return "\n\n".join(context_parts)

    # =====================================
    # Kernel Functions
    # =====================================

    @kernel_function(
        name="search_youtube_videos",
        description="YouTube 비디오를 검색하고 비디오 정보를 컨텍스트로 제공합니다",
    )
    async def search_youtube(
        self,
        query: Annotated[str, "검색할 키워드"],
        max_results: Annotated[Optional[int], "검색할 최대 비디오 수 (기본값: 5)"] = 5,
    ) -> str:
        """YouTube 비디오 검색"""
        # 새로운 세션을 사용하여 안전하게 검색 수행
        local_client = None
        local_session = None

        try:
            if not MCP_AVAILABLE:
                return "❌ MCP 라이브러리가 설치되지 않았습니다."

            if not YOUTUBE_API_KEY:
                return "❌ YOUTUBE_API_KEY 환경 변수가 설정되지 않았습니다."

            # 임시 MCP 연결 생성 (기존 연결과 독립적)
            server_params = StdioServerParameters(
                command=YOUTUBE_MCP_SERVER_COMMAND,
                args=["--api-key", YOUTUBE_API_KEY] if YOUTUBE_API_KEY else [],
                env={"YOUTUBE_API_KEY": YOUTUBE_API_KEY} if YOUTUBE_API_KEY else {},
            )

            local_client = stdio_client(server_params)
            read, write = await local_client.__aenter__()
            local_session = ClientSession(read, write)
            await local_session.__aenter__()
            await local_session.initialize()

            # searchVideos 도구 사용
            search_result = await local_session.call_tool(
                "searchVideos", {"query": query, "maxResults": max_results}
            )

            if getattr(search_result, "is_err", False):
                return f"❌ 검색 오류: {search_result.content}"

            # 결과 파싱
            videos = []
            if hasattr(search_result, "content") and search_result.content:
                try:
                    # MCP 응답 구조 안전하게 처리
                    if isinstance(search_result.content, list):
                        content_text = (
                            search_result.content[0].text
                            if search_result.content
                            else ""
                        )
                    else:
                        content_text = search_result.content

                    # 응답 파싱 (직접 배열 형태)
                    result_data = json.loads(content_text)

                    # 결과가 리스트인지 확인
                    if isinstance(result_data, list):
                        items = result_data
                    elif isinstance(result_data, dict) and "items" in result_data:
                        items = result_data["items"]
                    else:
                        logger.warning(f"예상과 다른 응답 형식: {type(result_data)}")
                        items = []

                    for item in items:
                        if not isinstance(item, dict):
                            continue

                        # YouTube API 표준 응답 구조에 맞춰 파싱
                        video_id = ""
                        if isinstance(item.get("id"), dict):
                            video_id = item.get("id", {}).get("videoId", "")
                        elif isinstance(item.get("id"), str):
                            video_id = item.get("id", "")

                        snippet = item.get("snippet", {})

                        video_info = {
                            "videoId": video_id,
                            "title": snippet.get("title", ""),
                            "description": snippet.get("description", ""),
                            "channelTitle": snippet.get("channelTitle", ""),
                            "publishedAt": snippet.get("publishedAt", ""),
                            "thumbnails": snippet.get("thumbnails", {}),
                        }
                        videos.append(video_info)

                except json.JSONDecodeError as e:
                    logger.error(f"JSON 파싱 오류: {e}")
                    return f"❌ 응답 데이터 파싱 실패: {str(e)}"
                except Exception as e:
                    logger.error(f"결과 파싱 중 오류: {e}")
                    return f"❌ 결과 파싱 실패: {str(e)}"

            # 결과 포맷팅
            template_str = """
🎥 YouTube 검색 결과 (MCP 서버)

🔍 검색어: {{ search_query }}
📊 검색 결과: {{ videos|length }}개 비디오

{% if videos %}
{% for video in videos %}
{{ loop.index }}. **{{ video.title }}**
   📺 채널: {{ video.channelTitle }}
   📅 게시일: {{ video.publishedAt }}
   🔗 링크: https://www.youtube.com/watch?v={{ video.videoId }}
   📝 설명: {{ video.description[:150] if video.description else 'N/A' }}...
   
{% endfor %}

📋 **컨텍스트 정보:**
{{ video_context }}

{% else %}
❌ 검색 결과가 없습니다.
{% endif %}

🕐 검색 시간: {{ datetime.now().strftime('%Y-%m-%d %H:%M:%S') }}
            """

            video_context = self._create_video_context(videos) if videos else ""

            template = Template(template_str)
            format_context = template.render(
                search_query=query,
                videos=videos,
                video_context=video_context,
                datetime=datetime,
            )

            return format_context

        except Exception as e:
            logger.error(f"YouTube 검색 중 오류: {str(e)}")
            return f"YouTube 검색 중 오류: {str(e)}"
        finally:
            # 임시 연결 정리
            try:
                if local_session:
                    await local_session.__aexit__(None, None, None)
                if local_client:
                    await local_client.__aexit__(None, None, None)
            except Exception as cleanup_error:
                logger.warning(f"임시 MCP 연결 정리 중 오류: {cleanup_error}")

    # =====================================
    # 리소스 정리
    # =====================================

    async def close(self):
        """MCP 연결 정리"""
        await self.cleanup()

    async def cleanup(self):
        """MCP 연결 정리 (cleanup 메서드 추가)"""
        try:
            # 세션부터 먼저 정리
            if self.mcp_session:
                try:
                    await self.mcp_session.__aexit__(None, None, None)
                    logger.info("YouTube MCP 세션 종료")
                except Exception as session_error:
                    logger.warning(f"MCP 세션 종료 중 오류: {session_error}")
                finally:
                    self.mcp_session = None

            # 클라이언트 정리
            if self.mcp_client:
                try:
                    await self.mcp_client.__aexit__(None, None, None)
                    logger.info("YouTube MCP 클라이언트 종료")
                except Exception as client_error:
                    logger.warning(f"MCP 클라이언트 종료 중 오류: {client_error}")
                finally:
                    self.mcp_client = None

        except Exception as e:
            logger.warning(f"MCP 연결 종료 중 전체 오류: {str(e)}")
        finally:
            # 강제로 연결 객체들을 None으로 설정
            self.mcp_session = None
            self.mcp_client = None
