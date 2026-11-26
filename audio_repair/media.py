import json
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


class RepairError(Exception):
    pass


class ToolExitError(RepairError):
    pass


PLAYLIST_SUFFIXES = {".m3u", ".m3u8", ".ffconcat"}


@dataclass(frozen=True)
class Stream:
    index: int
    kind: str
    codec: str
    sample_rate: int | None
    channels: int | None
    layout: str | None
    time_base: Fraction | None
    start: Fraction | None
    duration: Fraction | None
    dispositions: frozenset[str]
    tags: dict[str, str]
    bits_per_raw_sample: int | None = None
    display: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Media:
    path: Path
    streams: tuple[Stream, ...]
    start: Fraction
    duration: Fraction | None
    size: int
    format_name: str
    chapters: tuple[dict, ...]
    tags: dict[str, str]


def run_tool(args: list[str]) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise RepairError(f"Could not run {args[0]}: {exc}") from exc
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise ToolExitError(f"{args[0]} failed (exit {result.returncode}): {detail}")
    return result.stdout


def require_tools() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RepairError(f"{tool} is required; install FFmpeg and make it available on PATH")


def _fraction(value: object) -> Fraction | None:
    if value is None or value == "N/A":
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("invalid numeric value")
    return Fraction(value)


def _integer(value: object) -> int | None:
    if value is None or value == "N/A":
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("invalid integer")
    text = str(value)
    if not re.fullmatch(r"-?\d+", text):
        raise ValueError("invalid integer")
    return int(text)


def _tags(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in value.items()
    ):
        raise ValueError("invalid tags")
    return dict(value)


def _timestamp(data: dict, ticks: str, seconds: str, time_base: Fraction | None) -> Fraction | None:
    value = _integer(data.get(ticks))
    if value is not None and time_base is not None:
        return value * time_base
    return _fraction(data.get(seconds))


def _stream(data: object) -> Stream:
    if not isinstance(data, dict):
        raise ValueError("invalid stream")
    index = _integer(data.get("index"))
    kind = data.get("codec_type", "unknown")
    codec = data.get("codec_name", "unknown")
    layout = data.get("channel_layout")
    time_base = _fraction(data.get("time_base"))
    sample_rate = _integer(data.get("sample_rate"))
    channels = _integer(data.get("channels"))
    bits = _integer(data.get("bits_per_raw_sample"))
    dispositions = data.get("disposition", {})
    if (
        index is None or index < 0
        or not isinstance(kind, str) or not kind
        or not isinstance(codec, str) or not codec
        or (layout is not None and not isinstance(layout, str))
        or (time_base is not None and time_base <= 0)
        or (sample_rate is not None and sample_rate <= 0)
        or (channels is not None and channels <= 0)
        or (bits is not None and bits < 0)
        or not isinstance(dispositions, dict)
        or any(not isinstance(k, str) or type(v) is not int or v not in (0, 1)
               for k, v in dispositions.items())
    ):
        raise ValueError("invalid stream fields")
    duration = _timestamp(data, "duration_ts", "duration", time_base)
    if duration is not None and duration < 0:
        raise ValueError("negative stream duration")
    display = {}
    for key in ("sample_aspect_ratio", "display_aspect_ratio", "color_primaries", "color_space",
                "color_transfer", "color_range"):
        value = data.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError("invalid display metadata")
        if value not in (None, "N/A", "unknown", "unspecified", "0:1", "0:0"):
            display[key] = value
    for key in ("width", "height"):
        value = _integer(data.get(key))
        if value is not None:
            if value <= 0:
                raise ValueError("invalid video dimensions")
            display[key] = str(value)
    side_data = data.get("side_data_list", [])
    if not isinstance(side_data, list) or any(not isinstance(item, dict) for item in side_data):
        raise ValueError("invalid stream side data")
    for item in side_data:
        if "rotation" in item:
            rotation = _integer(item["rotation"])
            if rotation is None:
                raise ValueError("invalid rotation")
            display["rotation"] = str(rotation)
    return Stream(
        index=index,
        kind=kind,
        codec=codec,
        sample_rate=sample_rate,
        channels=channels,
        layout=layout,
        time_base=time_base,
        start=_timestamp(data, "start_pts", "start_time", time_base),
        duration=duration,
        dispositions=frozenset(k for k, v in dispositions.items() if v),
        tags=_tags(data.get("tags")),
        bits_per_raw_sample=bits or None,
        display=tuple(sorted(display.items())),
    )


def probe(path: Path) -> Media:
    name = str(path)
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", name):
        raise RepairError("Input must be a local file, not a URL")
    path = path.expanduser().absolute()
    if path.suffix.lower() in PLAYLIST_SUFFIXES:
        raise RepairError("HLS playlists and concat inputs are not supported")
    try:
        file_stat = path.stat()
    except OSError as exc:
        raise RepairError(f"Cannot access input {path}: {exc}") from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise RepairError(f"Input must be a regular file: {path}")
    raw = run_tool([
        "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
        "-show_streams", "-show_format", "-show_chapters", "-of", "json",
        str(path),
    ])
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("expected an object")
        raw_streams = data.get("streams")
        info = data.get("format")
        chapters = data.get("chapters", [])
        if (
            not isinstance(raw_streams, list)
            or not isinstance(info, dict)
            or not isinstance(chapters, list)
            or any(not isinstance(chapter, dict) for chapter in chapters)
        ):
            raise ValueError("missing streams, format, or chapters")
        format_name = info.get("format_name")
        if not isinstance(format_name, str) or not format_name:
            raise ValueError("missing format name")
        if {"hls", "concat"} & set(format_name.split(",")):
            raise RepairError("HLS playlists and concat inputs are not supported")
        streams = tuple(_stream(item) for item in raw_streams)
        if len({stream.index for stream in streams}) != len(streams):
            raise ValueError("duplicate stream index")
        start = _fraction(info.get("start_time"))
        duration = _fraction(info.get("duration"))
        if duration is not None and duration < 0:
            raise ValueError("negative format duration")
        return Media(
            path=path, streams=streams, start=start if start is not None else Fraction(0),
            duration=duration, size=file_stat.st_size, format_name=format_name,
            chapters=tuple(chapters), tags=_tags(info.get("tags")),
        )
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        raise RepairError(f"Malformed ffprobe output for {path}: {exc}") from exc


def select_audio(media: Media, override: int | None) -> Stream:
    audio = [stream for stream in media.streams if stream.kind == "audio"]
    if override is not None:
        primary = next((stream for stream in audio if stream.index == override), None)
        if primary is None:
            raise RepairError(f"Audio stream index {override} does not exist or is not audio")
    else:
        if not audio:
            raise RepairError("Input has no audio stream")
        defaults = [stream for stream in audio if "default" in stream.dispositions]
        primary = min(defaults or audio, key=lambda stream: stream.index)
    if primary.codec.lower() in {"unknown", "none", "n/a"}:
        raise RepairError(f"Audio stream {primary.index} has an unknown codec")
    if primary.channels not in (1, 2):
        raise RepairError(f"Audio stream {primary.index} must have one or two channels")
    if primary.layout is not None and primary.layout != {1: "mono", 2: "stereo"}[primary.channels]:
        raise RepairError(f"Audio stream {primary.index} has unsupported layout: {primary.layout}")
    return primary


def validate_video_streams(media: Media) -> None:
    for stream in media.streams:
        if (stream.kind not in {"video", "audio", "subtitle", "attachment"}
                or stream.codec.lower() in {"unknown", "mov_text", "none", "n/a"}):
            raise RepairError(
                f"Stream {stream.index} ({stream.kind}/{stream.codec}) cannot be safely "
                "copied to Matroska; use --audio-only"
            )
