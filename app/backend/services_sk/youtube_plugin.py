import os
import logging
from typing import Annotated, Optional, Dict, List, Any
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from dotenv import load_dotenv
from datetime import datetime
from jinja2 import Template
import httpx

load_dotenv(override=True)

# YouTube Data API 설정
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
YOUTUBE_API_BASE_URL = "https://www.googleapis.com/youtube/v3"
TIME_ZONE = os.getenv("TIME_ZONE", "Asia/Seoul")

logger = logging.getLogger(__name__)


# YouTube 플러그인: 순서별로 정리
class YouTubePlugin:
    """YouTube 검색을 위한 플러그인 (YouTube Data API 직접 호출)"""

    def __init__(self):
        # 연결 객체
        self.client = httpx.AsyncClient()
        self.max_results = 10

        # YouTube API 키 확인
        if not YOUTUBE_API_KEY:
            logger.warning(
                "YOUTUBE_API_KEY 환경 변수가 설정되지 않았습니다. YouTube 기능이 비활성화됩니다."
            )
            return

        logger.info("YouTube Plugin 초기화 완료.")

    # =====================================
    # 핵심 검색 메서드
    # =====================================

    async def _search_youtube_videos(
        self, query: str, max_results: int = 10
    ) -> Dict[str, Any]:
        """YouTube Data API를 사용하여 비디오 검색"""
        if not YOUTUBE_API_KEY:
            return {
                "status": "error",
                "message": "YouTube API 키가 설정되지 않았습니다.",
                "videos": [],
            }

        try:
            # YouTube Data API 검색 호출
            search_url = f"{YOUTUBE_API_BASE_URL}/search"
            params = {
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": max_results,
                "order": "relevance",
                "key": YOUTUBE_API_KEY,
            }

            logger.info(f"YouTube API 검색: '{query}'")

            response = await self.client.get(search_url, params=params)

            if response.status_code != 200:
                logger.error(
                    f"YouTube API 오류: {response.status_code} - {response.text}"
                )
                return {
                    "status": "error",
                    "message": f"검색 실패: {response.status_code}",
                    "videos": [],
                }

            data = response.json()

            # 검색 결과를 표준 형식으로 변환
            videos_data = []
            for item in data.get("items", []):
                video_info = {
                    "videoId": item["id"]["videoId"],
                    "title": item["snippet"]["title"],
                    "channelTitle": item["snippet"]["channelTitle"],
                    "description": item["snippet"]["description"],
                    "publishedAt": item["snippet"]["publishedAt"],
                    "thumbnails": item["snippet"].get("thumbnails", {}),
                }
                videos_data.append(video_info)

            return {
                "status": "success",
                "videos": videos_data,
                "search_query": query,
                "total_results": len(videos_data),
            }

        except Exception as e:
            logger.error(f"YouTube API 검색 실패: {str(e)}")
            return {"status": "error", "message": f"검색 실패: {str(e)}", "videos": []}

    async def _get_video_details(self, video_id: str) -> Dict[str, Any]:
        """YouTube Data API를 사용하여 비디오 세부 정보 조회"""
        if not YOUTUBE_API_KEY:
            return {"error": "YouTube API 키가 설정되지 않았습니다."}

        try:
            # YouTube Data API 비디오 정보 호출
            videos_url = f"{YOUTUBE_API_BASE_URL}/videos"
            params = {
                "part": "snippet,statistics,contentDetails",
                "id": video_id,
                "key": YOUTUBE_API_KEY,
            }

            response = await self.client.get(videos_url, params=params)

            if response.status_code != 200:
                return {
                    "error": f"YouTube API 오류: {response.status_code} - {response.text}"
                }

            data = response.json()

            if not data.get("items"):
                return {"error": "비디오를 찾을 수 없습니다."}

            item = data["items"][0]

            # 비디오 정보를 표준 형식으로 변환
            video = {
                "videoId": item["id"],
                "title": item["snippet"]["title"],
                "channelTitle": item["snippet"]["channelTitle"],
                "description": item["snippet"]["description"],
                "publishedAt": item["snippet"]["publishedAt"],
                "tags": item["snippet"].get("tags", []),
                "viewCount": item["statistics"].get("viewCount", "N/A"),
                "likeCount": item["statistics"].get("likeCount", "N/A"),
                "commentCount": item["statistics"].get("commentCount", "N/A"),
                "duration": item["contentDetails"].get("duration", "N/A"),
            }

            return video

        except Exception as e:
            logger.error(f"YouTube API 비디오 세부정보 조회 실패: {str(e)}")
            return {"error": f"비디오 세부정보 조회 실패: {str(e)}"}

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
        try:
            if not YOUTUBE_API_KEY:
                return "❌ YOUTUBE_API_KEY 환경 변수가 설정되지 않았습니다."

            # YouTube 비디오 검색
            result = await self._search_youtube_videos(query, max_results)

            if result.get("status") == "error":
                return f"❌ 오류: {result['message']}"

            # 결과 포맷팅
            template_str = """
🎥 YouTube 검색 결과

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

            videos = result.get("videos", [])
            video_context = self._create_video_context(videos) if videos else ""

            template = Template(template_str)
            format_context = template.render(
                search_query=result.get("search_query", query),
                videos=videos,
                video_context=video_context,
                datetime=datetime,
            )

            return format_context

        except Exception as e:
            return f"YouTube 검색 중 오류: {str(e)}"

    @kernel_function(
        name="get_youtube_video_details",
        description="특정 YouTube 비디오의 세부 정보를 조회합니다",
    )
    async def get_youtube_video_details(
        self, video_id: Annotated[str, "YouTube 비디오 ID (예: 'dQw4w9WgXcQ')"]
    ) -> str:
        """특정 YouTube 비디오의 세부 정보 조회"""
        try:
            if not YOUTUBE_API_KEY:
                return "❌ YOUTUBE_API_KEY 환경 변수가 설정되지 않았습니다."

            # 비디오 세부정보 조회
            result = await self._get_video_details(video_id)

            if "error" in result:
                return f"❌ 오류: {result['error']}"

            # 결과 포맷팅
            template_str = """
🎥 YouTube 비디오 세부정보 (YouTube API)

📺 **{{ video.title }}**

🏷️ 채널: {{ video.channelTitle }}
👀 조회수: {{ video.viewCount }}
👍 좋아요: {{ video.likeCount }}
💬 댓글: {{ video.commentCount }}
📅 게시일: {{ video.publishedAt }}
⏱️ 길이: {{ video.duration }}
🔗 링크: https://www.youtube.com/watch?v={{ video_id }}

📝 **설명:**
{{ video.description }}

🏷️ **태그:**
{% if video.tags %}
{{ video.tags | join(', ') }}
{% else %}
N/A
{% endif %}

🕐 조회 시간: {{ datetime.now().strftime('%Y-%m-%d %H:%M:%S') }}
            """

            template = Template(template_str)
            return template.render(video=result, video_id=video_id, datetime=datetime)

        except Exception as e:
            return f"YouTube 비디오 세부정보 조회 중 오류: {str(e)}"

    # =====================================
    # 리소스 정리
    # =====================================

    async def cleanup(self):
        """HTTP 클라이언트 정리"""
        try:
            if self.client:
                await self.client.aclose()
                logger.info("YouTube API 클라이언트 세션 종료")
        except Exception as e:
            logger.warning(f"클라이언트 종료 중 오류: {str(e)}")
