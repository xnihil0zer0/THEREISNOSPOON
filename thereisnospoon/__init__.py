"""THEREISNOSPOON — Autonomous Live YouTube Chat VJ & Clean 10-Second Neural Video Broadcast Engine."""

from __future__ import annotations

CHANNEL_HANDLE = "@THEREISNOSPOON-t4j"
CHANNEL_ID = "UC9ryqiyaNstEvvsw9EKpccA"
CHANNEL_URL = "https://www.youtube.com/@THEREISNOSPOON-t4j"
CHANNEL_LIVE_URL = "https://www.youtube.com/channel/UC9ryqiyaNstEvvsw9EKpccA/live"
STUDIO_CHANNEL_URL = "https://studio.youtube.com/channel/UC9ryqiyaNstEvvsw9EKpccA"
DEFAULT_RTMP_INGEST = "rtmp://a.rtmp.youtube.com/live2"

__version__ = "0.1.0"

from .chat_vj_director import (
    CHAT_VJ_SYSTEM_PROMPT_NO_CHAT,
    CHAT_VJ_SYSTEM_PROMPT_WITH_CHAT,
    ChatVJDirector,
    ChatVJScene,
    normalize_chat_lines,
)
from .clean_10s_generator import Clean10sClipGenerator
from .quality_metrics import compare_against_study_v2, evaluate_clip_quality
from .rtmp_broadcaster import YouTubeRTMPBroadcaster
from .youtube_chat_poller import YouTubeChatPoller

__all__ = [
    "CHANNEL_HANDLE",
    "CHANNEL_ID",
    "CHANNEL_URL",
    "CHANNEL_LIVE_URL",
    "STUDIO_CHANNEL_URL",
    "DEFAULT_RTMP_INGEST",
    "CHAT_VJ_SYSTEM_PROMPT_WITH_CHAT",
    "CHAT_VJ_SYSTEM_PROMPT_NO_CHAT",
    "ChatVJDirector",
    "ChatVJScene",
    "normalize_chat_lines",
    "Clean10sClipGenerator",
    "evaluate_clip_quality",
    "compare_against_study_v2",
    "YouTubeRTMPBroadcaster",
    "YouTubeChatPoller",
    "__version__",
]
