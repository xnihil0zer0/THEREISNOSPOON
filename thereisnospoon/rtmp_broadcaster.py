"""Continuous CPU FFmpeg encoding with a paced, synchronized raw A/V segment feeder."""

import json
import os
import select
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlsplit

VIDEO_BYTES = 864 * 480 * 3 // 2
AUDIO_BYTES = 48000 * 2 * 2 // 24


class YouTubeRTMPBroadcaster:
    """Prepare clips off the playback thread, then switch at exact 240-frame boundaries.

    An empty queue repeats the current clip through the same encoder and connection.
    Each prepared clip uses about 151 MB of temporary disk, plus one active and one preparing clip.
    clips_broadcasted includes repeats, bytes_sent counts muxed bytes, excluding network overhead.
    A network or encoder failure is reported, since no feeder can guarantee network availability.
    """

    def __init__(self, stream_key=None, rtmp_url=None, output_url=None, max_queue_size=4,
                 ffmpeg_bin="ffmpeg", ffprobe_bin="ffprobe"):
        key = os.getenv("YOUTUBE_STREAM_KEY", "") if stream_key is None else stream_key
        self._url = str(output_url or rtmp_url or os.getenv("YOUTUBE_RTMP_URL", ""))
        if not self._url and key:
            self._url = "rtmp://a.rtmp.youtube.com/live2/" + key
        self._key = key
        self.ffmpeg_bin, self.ffprobe_bin = ffmpeg_bin, ffprobe_bin
        self.max_queue_size = max(1, int(max_queue_size))
        self._queue = deque()
        self._lock, self._prepare_lock = threading.RLock(), threading.Lock()
        self._stop = threading.Event()
        self._thread = self._process = self._current = None
        self._generation = 0
        self._started = self._ended = 0.0
        self.current_clip_path = None
        self.clips_broadcasted = self.bytes_sent = 0
        self.last_error = ""
        self._hls_sizes = {}
        self._segment_prefix = ""

    @property
    def rtmp_url_masked(self):
        parts = urlsplit(self._url)
        if parts.scheme in ("rtmp", "rtmps"):
            host = parts.netloc.rsplit("@", 1)[-1]
            return f"{parts.scheme}://{host}{parts.path.rsplit('/', 1)[0]}/[redacted]"
        return self._url

    def _redact(self, text):
        parts = urlsplit(self._url)
        secrets = [self._url, self._key]
        if parts.scheme in ("rtmp", "rtmps"):
            secrets.extend([parts.path.rsplit("/", 1)[-1], parts.password, parts.query])
        for secret in secrets:
            if secret:
                text = text.replace(secret, "[redacted]")
        return text[-1500:]

    @property
    def is_streaming(self):
        return bool(self._process and self._process.poll() is None and self._thread and self._thread.is_alive())

    @property
    def uptime_sec(self):
        return max(0.0, (self._ended or time.monotonic()) - self._started) if self._started else 0.0

    def get_telemetry(self):
        with self._lock:
            return dict(clips_broadcasted=self.clips_broadcasted, bytes_sent=self.bytes_sent,
                        uptime_sec=self.uptime_sec, rtmp_url_masked=self.rtmp_url_masked,
                        is_streaming=self.is_streaming, last_error=self.last_error,
                        current_clip_path=self.current_clip_path, queue_size=len(self._queue))

    def enqueue_clip(self, clip_path):
        """Normalize a completed MP4 before publishing it to the queue. May block during decoding."""
        path = Path(clip_path).resolve(strict=True)
        if not path.is_file() or path.suffix.lower() != ".mp4":
            raise ValueError("Expected a completed .mp4 clip")
        with self._prepare_lock:
            with self._lock:
                if len(self._queue) >= self.max_queue_size:
                    raise BufferError("Broadcast clip queue is full")
                generation = self._generation
            cache = tempfile.TemporaryDirectory(prefix="thereisnospoon-av-")
            try:
                probe = subprocess.run([self.ffprobe_bin, "-v", "error", "-show_streams", "-of", "json", str(path)],
                                       capture_output=True, check=True, timeout=20)
                streams = json.loads(probe.stdout)["streams"]
                audio = "0:a:0" if any(s.get("codec_type") == "audio" for s in streams) else "1:a:0"
                vf = ("fps=24,scale=864:480:force_original_aspect_ratio=decrease,"
                      "pad=864:480:(ow-iw)/2:(oh-ih)/2,setsar=1,tpad=stop_mode=clone:stop_duration=10")
                cmd = [self.ffmpeg_bin, "-nostdin", "-v", "error", "-y", "-threads", "2", "-i", str(path),
                       "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-filter_threads", "1",
                       "-map", "0:v:0", "-vf", vf, "-t", "10", "-pix_fmt", "yuv420p", "-threads", "1",
                       "-f", "rawvideo", cache.name + "/video", "-map", audio, "-af", "apad",
                       "-t", "10", "-ar", "48000", "-ac", "2", "-f", "s16le", cache.name + "/audio"]
                subprocess.run(cmd, capture_output=True, check=True, timeout=90)
                if any((Path(cache.name) / name).stat().st_size != size * 240
                       for name, size in (("video", VIDEO_BYTES), ("audio", AUDIO_BYTES))):
                    raise ValueError("Clip could not be normalized to exactly 10 seconds of audio and video")
                with self._lock:
                    if generation != self._generation:
                        raise RuntimeError("Broadcast stopped while the clip was being prepared")
                    self._queue.append((str(path), cache))
            except Exception as exc:
                cache.cleanup()
                self.last_error = self._redact(str(exc))
                raise
        return str(path)

    queue_clip = enqueue_clip

    def _command(self, video_fd, audio_fd):
        cmd = [self.ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error", "-nostats", "-y",
               "-progress", "pipe:1", "-thread_queue_size", "64", "-f", "rawvideo", "-pixel_format", "yuv420p",
               "-video_size", "864x480", "-framerate", "24", "-i", f"pipe:{video_fd}",
               "-thread_queue_size", "64", "-f", "s16le", "-ar", "48000", "-ac", "2",
               "-probesize", "32", "-analyzeduration", "0", "-i", f"pipe:{audio_fd}",
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "ultrafast",
               "-tune", "zerolatency", "-threads", "2", "-b:v", "2500k", "-maxrate", "2500k",
               "-bufsize", "5000k", "-g", "48", "-keyint_min", "48", "-sc_threshold", "0",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
               "-flush_packets", "1"]
        if self._url.lower().endswith(".m3u8"):
            cmd += ["-f", "hls", "-hls_time", "2", "-hls_list_size", "6", "-hls_flags", "delete_segments+temp_file"]
            cmd += ["-hls_segment_filename", str(Path(self._url).parent / (self._segment_prefix + "%06d.ts"))]
        else:
            cmd += ["-f", "flv", "-flvflags", "no_duration_filesize"]
        return cmd + [self._url]

    def start(self, initial_clip=None):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
        if initial_clip is not None:
            self.enqueue_clip(initial_clip)
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            if not self._url or not self._queue:
                raise ValueError("Configure a stream key or output URL and enqueue a clip before starting")
            if urlsplit(self._url).scheme not in ("rtmp", "rtmps"):
                if urlsplit(self._url).scheme or Path(self._url).suffix.lower() not in (".flv", ".m3u8"):
                    raise ValueError("Output must be RTMP, RTMPS, or a local .flv or .m3u8 path")
                Path(self._url).parent.mkdir(parents=True, exist_ok=True)
            self._stop.clear()
            self._started, self._ended = time.monotonic(), 0.0
            self.clips_broadcasted = self.bytes_sent = 0
            self.last_error, self._hls_sizes = "", {}
            self._segment_prefix = f"{Path(self._url).stem}-{time.time_ns()}-"
            self._thread = threading.Thread(target=self._run, name="youtube-broadcast", daemon=True)
            self._thread.start()

    def _write_frame(self, fds, video, audio):
        pending = dict(zip(fds, (memoryview(video), memoryview(audio))))
        deadline = time.monotonic() + 10
        while pending and not self._stop.is_set():
            if self._process.poll() is not None:
                raise RuntimeError(f"FFmpeg exited with status {self._process.returncode}")
            if time.monotonic() > deadline:
                raise TimeoutError("FFmpeg stopped accepting audio/video")
            for fd in select.select([], list(pending), [], 0.1)[1]:
                try:
                    count = os.write(fd, pending[fd])
                except BlockingIOError:
                    continue
                pending[fd] = pending[fd][count:]
                if not pending[fd]:
                    del pending[fd]

    def _monitor(self):
        for line in self._process.stdout:
            key, separator, value = line.strip().partition("=")
            with self._lock:
                if key == "total_size" and value.isdigit():
                    self.bytes_sent = max(self.bytes_sent, int(value))
                elif key == "out_time_us" and value.lstrip("-").isdigit():
                    self.clips_broadcasted = max(self.clips_broadcasted, int(value) // 10000000)
                elif not separator and line.strip():
                    self.last_error = self._redact(line.strip())
                if key == "progress" and self._url.lower().endswith(".m3u8"):
                    sizes = {}
                    for path in Path(self._url).parent.glob(self._segment_prefix + "*.ts"):
                        try:
                            sizes[str(path)] = path.stat().st_size
                        except FileNotFoundError:
                            continue
                    self.bytes_sent += sum(max(0, size - self._hls_sizes.get(name, 0)) for name, size in sizes.items())
                    self._hls_sizes = sizes

    def _discard_clips(self):
        with self._lock:
            for _, cache in list(self._queue) + ([self._current] if self._current else []):
                cache.cleanup()
            self._queue.clear()
            self._current = self.current_clip_path = None
            self._generation += 1

    def _run(self):
        descriptors, monitor = [], None
        try:
            descriptors.extend(os.pipe())
            descriptors.extend(os.pipe())
            vr, vw, ar, aw = descriptors
            self._process = subprocess.Popen(self._command(vr, ar), pass_fds=(vr, ar), stdin=subprocess.DEVNULL,
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for fd in (vr, ar):
                os.close(fd)
                descriptors.remove(fd)
            for fd in (vw, aw):
                os.set_blocking(fd, False)
            monitor = threading.Thread(target=self._monitor, name="ffmpeg-telemetry", daemon=True)
            monitor.start()
            deadline = time.monotonic()
            while not self._stop.is_set():
                with self._lock:
                    if self._queue:
                        if self._current:
                            self._current[1].cleanup()
                        self._current = self._queue.popleft()
                    self.current_clip_path, cache = self._current
                with open(cache.name + "/video", "rb") as video, open(cache.name + "/audio", "rb") as audio:
                    for _ in range(240):
                        if self._stop.wait(max(0, deadline - time.monotonic())):
                            break
                        self._write_frame((vw, aw), video.read(VIDEO_BYTES), audio.read(AUDIO_BYTES))
                        deadline += 1 / 24
                if self._stop.wait(max(0, deadline - time.monotonic())):
                    break
        except Exception as exc:
            self.last_error = self._redact(str(exc))
        finally:
            for fd in descriptors:
                os.close(fd)
            if self._process:
                for action in (None, self._process.terminate, self._process.kill):
                    try:
                        if action and self._process.poll() is None:
                            action()
                        self._process.wait(timeout=3)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                if self._process.returncode and not self._stop.is_set() and not self.last_error:
                    self.last_error = f"FFmpeg exited with status {self._process.returncode}"
            if monitor:
                monitor.join(2)
                self._process.stdout.close()
            self._ended = time.monotonic()
            self._discard_clips()

    def stop(self, timeout=15):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError("Broadcast worker is still stopping")
        self._discard_clips()

    get_status = get_telemetry
