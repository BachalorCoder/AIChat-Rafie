from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
import os
import re
from urllib.parse import urlparse

import requests

from localagent.web_search import web_search


YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"


@dataclass
class ChannelHit:
    title: str
    channel_id: str
    description: str
    url: str
    score: float


@dataclass
class VideoHit:
    title: str
    published_at: str
    url: str


def maybe_handle_profile_lookup(command: str, config: dict) -> str | None:
    text = " ".join(command.lower().split())

    if not _looks_like_profile_or_video_request(text):
        return None

    if "tiktok" in text:
        name = _extract_subject(command, platform="tiktok")
        if not name:
            return "Which TikTok username or creator should I look up?"
        return _lookup_tiktok_best_effort(name, command, config)

    if _looks_like_youtube_request(text):
        name = _extract_subject(command, platform="youtube")
        if not name:
            return "Which YouTube channel or creator should I look up?"

        api_key = _youtube_api_key(config)
        if not api_key:
            return (
                "I can look up YouTube profiles and recent videos, but I need a YouTube Data API key first. "
                "Set an environment variable named YOUTUBE_API_KEY, then restart me."
            )

        if _wants_recent_videos(text):
            return _youtube_recent_videos(name, api_key)

        return _youtube_profile(name, api_key)

    return None


def _looks_like_profile_or_video_request(text: str) -> bool:
    triggers = [
        "youtube",
        "youtuber",
        "tiktok",
        "profile",
        "channel",
        "recent video",
        "latest video",
        "newest video",
        "recent upload",
        "latest upload",
        "uploaded recently",
        "look up",
        "find user",
        "find creator",
    ]
    return any(trigger in text for trigger in triggers)


def _looks_like_youtube_request(text: str) -> bool:
    return any(
        trigger in text
        for trigger in [
            "youtube",
            "youtuber",
            "channel",
            "recent video",
            "latest video",
            "newest video",
            "recent upload",
            "latest upload",
            "uploaded recently",
        ]
    )


def _wants_recent_videos(text: str) -> bool:
    return any(
        trigger in text
        for trigger in [
            "recent video",
            "latest video",
            "newest video",
            "recent upload",
            "latest upload",
            "uploaded recently",
            "last video",
            "new videos",
        ]
    )


def _extract_subject(command: str, platform: str) -> str:
    text = command.strip()

    patterns = [
        r"(?:latest|recent|newest|last)\s+(?:videos?|uploads?)\s+(?:from|by|of)\s+(?P<name>.+)",
        r"(?:what(?:'s| is)?|show me|find|look up|search for)\s+(?:the\s+)?(?:latest|recent|newest)?\s*(?:videos?|uploads?)?\s*(?:from|by|of)?\s*(?P<name>.+?)\s+(?:on|from)\s+"
        + re.escape(platform),
        r"(?:look up|find|search for|who is)\s+(?P<name>.+?)\s+on\s+"
        + re.escape(platform),
        r"(?:"
        + re.escape(platform)
        + r")\s+(?:user|creator|channel|profile|account)?\s*(?P<name>.+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _clean_subject(match.group("name"))

    cleaned = re.sub(
        r"\b(rafie|rafi|please|can you|could you|look up|find|search for|search|who is|what is|show me|recent|latest|newest|last|videos?|uploads?|from|by|of|on|youtube|youtuber|tiktok|profile|channel|account|creator)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    return _clean_subject(cleaned)


def _clean_subject(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[?.!,]+$", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _youtube_api_key(config: dict) -> str:
    env_name = (
        config.get("integrations", {}).get("youtube_api_key_env")
        or config.get("youtube", {}).get("api_key_env")
        or "YOUTUBE_API_KEY"
    )
    return os.getenv(env_name, "").strip()


def _youtube_get(endpoint: str, params: dict, api_key: str) -> dict:
    params = dict(params)
    params["key"] = api_key

    response = requests.get(
        f"{YOUTUBE_API_BASE}/{endpoint}",
        params=params,
        timeout=15,
    )

    if response.status_code == 403:
        raise RuntimeError("YouTube API refused the request. Check your API key or quota.")

    response.raise_for_status()
    return response.json()


def _find_youtube_channels(name: str, api_key: str, limit: int = 5) -> list[ChannelHit]:
    data = _youtube_get(
        "search",
        {
            "part": "snippet",
            "type": "channel",
            "q": name,
            "maxResults": limit,
        },
        api_key,
    )

    channels: list[ChannelHit] = []

    for item in data.get("items", []):
        item_id = item.get("id", {})
        snippet = item.get("snippet", {})
        channel_id = item_id.get("channelId", "")
        title = snippet.get("title", "").strip()
        description = snippet.get("description", "").strip()

        if not channel_id or not title:
            continue

        channels.append(
            ChannelHit(
                title=title,
                channel_id=channel_id,
                description=description,
                url=f"https://www.youtube.com/channel/{channel_id}",
                score=_name_score(name, title),
            )
        )

    channels.sort(key=lambda channel: channel.score, reverse=True)
    return channels


def _youtube_profile(name: str, api_key: str) -> str:
    try:
        channels = _find_youtube_channels(name, api_key)
    except Exception as exc:
        return f"I tried checking YouTube, but the YouTube lookup failed: {exc}"

    if not channels:
        return f"I could not find a YouTube channel for {name}."

    best = channels[0]

    if best.score < 0.45 and len(channels) > 1:
        suggestions = ", ".join(channel.title for channel in channels[:3])
        return f"I could not find an exact YouTube channel named {name}. Did you mean one of these: {suggestions}?"

    return (
        f"I found the YouTube channel {best.title}. "
        f"Channel link: {best.url}. "
        f"{best.description[:260]}"
    ).strip()


def _youtube_recent_videos(name: str, api_key: str, max_results: int = 5) -> str:
    try:
        channels = _find_youtube_channels(name, api_key)
    except Exception as exc:
        return f"I tried checking YouTube, but the channel lookup failed: {exc}"

    if not channels:
        return f"I could not find a YouTube channel for {name}."

    best = channels[0]

    if best.score < 0.45 and len(channels) > 1:
        suggestions = ", ".join(channel.title for channel in channels[:3])
        return f"I could not find an exact YouTube channel named {name}. Did you mean one of these: {suggestions}?"

    try:
        data = _youtube_get(
            "search",
            {
                "part": "snippet",
                "type": "video",
                "channelId": best.channel_id,
                "order": "date",
                "maxResults": max_results,
            },
            api_key,
        )
    except Exception as exc:
        return f"I found {best.title}, but I could not get their recent videos: {exc}"

    videos: list[VideoHit] = []

    for item in data.get("items", []):
        video_id = item.get("id", {}).get("videoId", "")
        snippet = item.get("snippet", {})
        title = snippet.get("title", "").strip()
        published_at = snippet.get("publishedAt", "").strip()

        if not video_id or not title:
            continue

        videos.append(
            VideoHit(
                title=title,
                published_at=published_at,
                url=f"https://www.youtube.com/watch?v={video_id}",
            )
        )

    if not videos:
        return f"I found {best.title}, but I could not find recent public uploads."

    lines = [f"I found {best.title}. Their recent YouTube videos are:"]

    for index, video in enumerate(videos, start=1):
        ago = _published_ago(video.published_at)
        lines.append(f"{index}. {video.title} — published {ago}. {video.url}")

    return "\n".join(lines)


def _lookup_tiktok_best_effort(name: str, command: str, config: dict) -> str:
    query = f'site:tiktok.com/@ "{name}" TikTok profile'
    wants_recent = _wants_recent_videos(command.lower())

    if wants_recent:
        query = f'site:tiktok.com/@ "{name}" TikTok recent video'

    try:
        results = web_search(query, max_results=5)
    except Exception as exc:
        return f"I tried checking TikTok, but the lookup failed: {exc}"

    profile_hits = []
    for result in results:
        parsed = urlparse(result.url)
        path = parsed.path.strip("/")
        if parsed.netloc.endswith("tiktok.com") and path.startswith("@"):
            profile_hits.append(result)

    if not profile_hits:
        return (
            f"I could not verify a TikTok profile for {name}. "
            "TikTok public lookup is limited without official API access, so I may need the exact username."
        )

    top = profile_hits[:3]
    lines = [
        "TikTok lookup is best-effort unless you connect official TikTok API access.",
        "Closest matches I found:",
    ]

    for index, result in enumerate(top, start=1):
        lines.append(f"{index}. {result.title}: {result.url}")

    return "\n".join(lines)


def _name_score(query: str, title: str) -> float:
    query_norm = _normalize_name(query)
    title_norm = _normalize_name(title)

    if not query_norm or not title_norm:
        return 0.0

    if query_norm == title_norm:
        return 1.0

    if query_norm in title_norm or title_norm in query_norm:
        return 0.85

    return SequenceMatcher(None, query_norm, title_norm).ratio()


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _published_ago(published_at: str) -> str:
    try:
        published = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except Exception:
        return published_at or "an unknown time ago"

    now = datetime.now(timezone.utc)
    delta = now - published

    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "just now"

    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"

    hours = minutes // 60
    if hours < 48:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"

    days = hours // 24
    if days < 60:
        return f"{days} day{'s' if days != 1 else ''} ago"

    months = days // 30
    if months < 24:
        return f"{months} month{'s' if months != 1 else ''} ago"

    years = days // 365
    return f"{years} year{'s' if years != 1 else ''} ago"