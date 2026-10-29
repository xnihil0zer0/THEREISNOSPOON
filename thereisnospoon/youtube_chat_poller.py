"""Read public YouTube live chat or inject local chat without posting to YouTube."""

import json
import os
import re
import threading
import time
from collections import OrderedDict, deque
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlsplit

import requests

CHANNEL_ID = "UC9ryqiyaNstEvvsw9EKpccA"
LIVE_URL = f"https://www.youtube.com/channel/{CHANNEL_ID}/live"
CHAT_URL = "https://www.youtube.com/youtubei/v1/live_chat/get_live_chat"
API_URL = "https://www.googleapis.com/youtube/v3/"


class _Canonical(HTMLParser):
    url = ""

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "link" and "canonical" in attrs.get("rel", "").split():
            self.url = attrs.get("href", "")


def _embedded(page, name, default=None):
    pattern = rf'(?:"{name}"\s*:|{name}\s*=|window\["{name}"\]\s*=)\s*'
    for match in re.finditer(pattern, page):
        try:
            return json.JSONDecoder().raw_decode(page[match.end():])[0]
        except ValueError:
            continue
    return default


def _find(value, key):
    if isinstance(value, dict):
        if key in value:
            yield value[key]
        for child in value.values():
            yield from _find(child, key)
    elif isinstance(value, list):
        for child in value:
            yield from _find(child, key)


def _text(value):
    return value.get("simpleText") or "".join(
        run.get("text", "") or next(iter(run.get("emoji", {}).get("shortcuts", [])), "")
        for run in value.get("runs", [])
    )


class YouTubeChatPoller:
    """Explicit start/stop lifecycle, bounded history, and server-paced polling.

    clear_chat preserves message IDs to avoid replaying previously consumed chat.
    InnerTube is an undocumented endpoint and may change independently of this client.
    """

    def __init__(self, api_key=None, live_url=LIVE_URL, history_size=100, session=None, poll_interval=5.0):
        self.api_key = os.getenv("YOUTUBE_API_KEY", "") if api_key is None else api_key
        self.live_url = live_url
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en"})
        self.poll_interval = max(1.0, float(poll_interval))
        self._history = deque(maxlen=max(10, history_size))
        self._seen = OrderedDict()
        self._lock, self._poll_lock = threading.RLock(), threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._generation = 0
        self._next_poll = self._api_retry = 0.0
        self.video_id = self.innertube_api_key = self.continuation = self.live_chat_id = ""
        self.page_token = self.last_error = ""
        self.is_live = False
        self.broadcast_status = "unknown"
        self.context = {}

    def _request(self, method, url, **kwargs):
        response = getattr(self.session, method)(url, timeout=(5, 15), **kwargs)
        response.raise_for_status()
        return response

    def _append(self, author, message, timestamp, source, badge="VIEWER", message_id="", generation=None):
        with self._lock:
            if generation is not None and generation != self._generation:
                return None
            if message_id and message_id in self._seen:
                return None
            if message_id:
                self._seen[message_id] = True
                if len(self._seen) > 2000:
                    self._seen.popitem(last=False)
            row = dict(author=str(author), message=str(message), timestamp=timestamp, source=source, badge=str(badge))
            self._history.append(row)
            return dict(row)

    def inject_chat_message(self, author, message, badge="VIEWER"):
        return self._append(author, message, datetime.now(timezone.utc).isoformat(), "local", badge)

    def inject_chat_batch(self, messages):
        with self._lock:
            return [self.inject_chat_message(row["author"], row["message"], row.get("badge", "VIEWER"))
                    for row in messages]

    def get_last_10_chat_lines(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in list(self._history)[-10:]]

    def clear_chat(self):
        with self._lock:
            self._history.clear()
            self._generation += 1

    def _advance(self, chat):
        self.continuation = ""
        delay = self.poll_interval
        for entry in chat.get("continuations", []):
            for name in ("timedContinuationData", "invalidationContinuationData", "reloadContinuationData"):
                data = entry.get(name, {})
                if data.get("continuation"):
                    self.continuation = data["continuation"]
                    delay = max(delay, float(data.get("timeoutMs", 0)) / 1000)
                    break
            if self.continuation:
                break
        self._next_poll = time.monotonic() + delay

    def discover_live_chat(self):
        with self._poll_lock:
            page = self._request("get", self.live_url).text
            canonical = _Canonical()
            canonical.feed(page)
            player = _embedded(page, "ytInitialPlayerResponse", {})
            details = player.get("videoDetails", {})
            live = player.get("microformat", {}).get("playerMicroformatRenderer", {}).get("liveBroadcastDetails", {})
            self.video_id = parse_qs(urlsplit(canonical.url).query).get("v", [details.get("videoId", "")])[0]
            self.is_live = bool(live.get("isLiveNow") or details.get("isLive"))
            self.is_live = self.is_live and not (details.get("isUpcoming") or live.get("endTimestamp"))
            self.broadcast_status = "live" if self.is_live else "offline"
            if not self.is_live and (details.get("isUpcoming") or live.get("startTimestamp")):
                self.broadcast_status = "ended" if live.get("endTimestamp") else "upcoming"
            self.innertube_api_key = _embedded(page, "INNERTUBE_API_KEY", "")
            self.context = _embedded(page, "INNERTUBE_CONTEXT", {})
            if not self.context.get("client"):
                self.context = {"client": {"clientName": "WEB",
                                          "clientVersion": _embedded(page, "INNERTUBE_CLIENT_VERSION", "")}}
            self.continuation = self.live_chat_id = self.page_token = ""
            if self.is_live:
                data = _embedded(page, "ytInitialData", {})
                self._advance(next(_find(data, "liveChatRenderer"), {}))
            return dict(video_id=self.video_id, is_live=self.is_live, broadcast_status=self.broadcast_status,
                        chat_active=bool(self.continuation))

    def _api_poll(self, generation):
        if not self.live_chat_id:
            data = self._request("get", API_URL + "videos", params={"key": self.api_key, "id": self.video_id,
                                 "part": "liveStreamingDetails"}).json()
            items = data.get("items", [])
            self.live_chat_id = items[0].get("liveStreamingDetails", {}).get("activeLiveChatId", "") if items else ""
        if not self.live_chat_id:
            self._api_retry = time.monotonic() + 60
            return None
        params = dict(key=self.api_key, liveChatId=self.live_chat_id, part="snippet,authorDetails", maxResults=200)
        if self.page_token:
            params["pageToken"] = self.page_token
        data = self._request("get", API_URL + "liveChat/messages", params=params).json()
        self.page_token = data.get("nextPageToken", "")
        self._next_poll = time.monotonic() + max(self.poll_interval, data.get("pollingIntervalMillis", 5000) / 1000)
        rows = []
        for item in data.get("items", []):
            snippet, author = item.get("snippet", {}), item.get("authorDetails", {})
            if snippet.get("type") == "textMessageEvent":
                row = self._append(author.get("displayName", "Viewer"), snippet.get("displayMessage", ""),
                                   snippet.get("publishedAt") or datetime.now(timezone.utc).isoformat(),
                                   "youtube_api", message_id=item.get("id", ""),
                                   generation=generation)
                if row:
                    rows.append(row)
        if data.get("offlineAt"):
            self.live_chat_id = self.continuation = ""
            self.is_live, self.broadcast_status = False, "ended"
        return rows

    def _inner_poll(self, generation):
        data = self._request("post", CHAT_URL, params={"key": self.innertube_api_key},
                             json={"context": self.context, "continuation": self.continuation}).json()
        chat = data.get("continuationContents", {}).get("liveChatContinuation", {})
        self._advance(chat)
        rows = []
        for item in _find(chat.get("actions", []), "liveChatTextMessageRenderer"):
            timestamp = datetime.fromtimestamp(int(item.get("timestampUsec", time.time() * 1e6)) / 1e6, timezone.utc)
            badges = [badge.get("tooltip", "VIEWER") for badge in _find(item, "liveChatAuthorBadgeRenderer")]
            row = self._append(_text(item.get("authorName", {})), _text(item.get("message", {})),
                               timestamp.isoformat(), "youtube", ", ".join(badges) or "VIEWER",
                               item.get("id", ""), generation)
            if row:
                rows.append(row)
        return rows

    def poll_once(self) -> List[Dict[str, Any]]:
        with self._poll_lock:
            if time.monotonic() < self._next_poll:
                return []
            with self._lock:
                generation = self._generation
            self._next_poll = time.monotonic() + self.poll_interval
            try:
                if not (self.continuation or self.live_chat_id):
                    self.discover_live_chat()
                if not self.is_live:
                    self.last_error = ""
                    self._next_poll = time.monotonic() + 60
                    return []
                if self.api_key and time.monotonic() >= self._api_retry:
                    try:
                        rows = self._api_poll(generation)
                        if rows is not None:
                            self.last_error = ""
                            return rows
                    except (requests.RequestException, ValueError) as exc:
                        self._api_retry = time.monotonic() + 300
                        self.live_chat_id = self.page_token = ""
                        if not self.continuation:
                            raise exc
                rows = self._inner_poll(generation) if self.continuation and self.innertube_api_key else []
                if not (self.continuation or self.live_chat_id):
                    self._next_poll = time.monotonic() + 60
                self.last_error = ""
                return rows
            except (requests.RequestException, ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
                self.last_error = str(exc).replace(self.api_key or "\0", "[redacted]")
                self.continuation = self.live_chat_id = ""
                self._next_poll = time.monotonic() + 30
                return []

    def start(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="youtube-chat", daemon=True)
            self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            self.poll_once()
            self._stop.wait(max(0.1, self._next_poll - time.monotonic()))

    def stop(self, timeout=35):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError("YouTube chat request is still finishing")
