import os
import re
from dataclasses import dataclass
from fractions import Fraction
from math import ceil
from pathlib import Path

from .media import Media, RepairError, Stream, select_audio, validate_video_streams


SAMPLE_RATE = 44100
CHUNK_SECONDS = 6
OVERLAP_SECONDS = 1
CONTEXT_SECONDS = 1
CHUNK_BATCH_SIZE = 1
FLAC_PICTURE_BLOCK_LIMIT = 2**24
MAX_ATTENUATION_DB = 6


@dataclass(frozen=True)
class SampleRequest:
    start: Fraction | None
    duration: Fraction


@dataclass(frozen=True)
class Plan:
    media: Media
    primary: Stream
    output: Path
    video: bool
    sample: SampleRequest | None
    device: str
    allow_attenuation: bool
    duration: Fraction | None
    ram_bytes: int | None
    scratch_bytes: int | None
    sample_from_first_audio: bool = False
    artwork: tuple[Stream, ...] = ()
    sample_picture: Stream | None = None

    @property
    def omitted_streams(self) -> tuple[Stream, ...]:
        if self.video and self.sample_picture is None:
            return ()
        return tuple(stream for stream in self.media.streams
                     if stream.index != self.primary.index and stream != self.sample_picture
                     and stream not in self.artwork)


def seconds_arg(value: str) -> Fraction:
    decimal = r"(?:\d+(?:\.\d+)?|\.\d+)"
    if re.fullmatch(decimal, value):
        return Fraction(value)
    match = re.fullmatch(r"(\d+):([0-5]\d):([0-5]\d(?:\.\d+)?)", value)
    if match:
        hours, minutes, seconds = match.groups()
        return int(hours) * 3600 + int(minutes) * 60 + Fraction(seconds)
    raise ValueError("time must be nonnegative decimal seconds or HH:MM:SS[.fraction]")


def build_plan(
    media: Media, *, output: Path | None = None, audio_only: bool = False,
    sample: SampleRequest | None = None, sample_video: bool = False,
    audio_stream_index: int | None = None,
    device: str = "auto", allow_attenuation: bool | None = None,
) -> Plan:
    primary = select_audio(media, audio_stream_index)
    sample_from_first_audio = sample is not None and sample.start is None
    picture = next(
        (stream for stream in sorted(media.streams, key=lambda item: item.index)
         if stream.kind == "video" and "attached_pic" not in stream.dispositions), None,
    )
    if sample_video and (sample is None or audio_only or picture is None):
        raise RepairError("--sample-video requires a video input and a sample request without --audio-only")
    video = picture is not None and not audio_only and (sample is None or sample_video)
    if video and not sample_video:
        validate_video_streams(media)
    if sample is not None:
        if sample.duration <= 0 or sample.duration > 120:
            raise RepairError("Sample duration must be positive and at most 120 seconds")
        if sample.start is not None and sample.start < 0:
            raise RepairError("Sample start must be nonnegative")
        first_audio = primary.start if primary.start is not None else media.start
        start = sample.start if sample.start is not None else max(Fraction(0), first_audio - media.start)
        sample = SampleRequest(start=start, duration=sample.duration)

    if output is None:
        if sample is not None:
            name = (f"{media.path.stem}.sample-{round(sample.start * 1000)}-"
                    f"{round(sample.duration * 1000)}.repaired.{'mkv' if sample_video else 'flac'}")
        else:
            name = f"{media.path.stem}.repaired.{'mkv' if video else 'flac'}"
        output = media.path.parent / name
    else:
        output = output.expanduser()
    output = output.absolute()
    try:
        parent = output.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RepairError(f"Output parent directory does not exist or is inaccessible: {output.parent}") from exc
    if not parent.is_dir():
        raise RepairError(f"Output parent is not a directory: {parent}")
    output = parent / output.name
    try:
        if output.resolve() == media.path.resolve():
            raise RepairError("Output must not be the input file")
        if os.path.lexists(output):
            if output.exists() and os.path.samefile(output, media.path):
                raise RepairError("Output must not reference the input file")
            raise RepairError(f"Output already exists: {output}")
    except (OSError, RuntimeError) as exc:
        raise RepairError(f"Cannot check output {output}: {exc}") from exc
    suffix = output.suffix.lower()
    if sample_video and suffix != ".mkv":
        raise RepairError("--sample-video output must use .mkv; use --sample for .flac or .wav")
    if video and suffix != ".mkv":
        raise RepairError("Video output must use .mkv; use --audio-only for .flac or .wav")
    if not video and suffix not in {".flac", ".wav"}:
        raise RepairError("Audio output must use .flac or .wav")
    if allow_attenuation and suffix == ".wav":
        raise RepairError("--allow-attenuation is only for integer output, not float WAV")
    if allow_attenuation is None:
        allow_attenuation = suffix != ".wav"
    if device not in {"auto", "mps", "cpu", "cuda"}:
        raise RepairError(f"Unknown device: {device}")

    artwork = ()
    if not (video or audio_only or sample is not None) and suffix == ".flac":
        artwork = tuple(stream for stream in media.streams
                        if stream.kind == "video" and "attached_pic" in stream.dispositions
                        and stream.codec in {"mjpeg", "png"})
    duration = (sample.duration + 2 * CONTEXT_SECONDS
                if sample is not None else primary.duration or media.duration)
    ram_bytes = scratch_bytes = None
    if duration is not None:
        pcm_bytes = ceil(duration * SAMPLE_RATE) * primary.channels * 4
        ram_bytes = 5 * pcm_bytes + 8 * 2**30
        copies = 5 if sample_video else 4 if video or sample is not None else 3
        video_bytes = media.size if video else 0
        if sample_video:
            share = min(1, 2 * duration / media.duration) if media.duration else 1
            video_bytes = 2 * ceil(media.size * share)
        scratch_bytes = (copies * pcm_bytes + video_bytes + 64 * 2**20
                         + len(artwork) * FLAC_PICTURE_BLOCK_LIMIT)
    return Plan(
        media=media, primary=primary, output=output, video=video, sample=sample,
        device=device, allow_attenuation=allow_attenuation, duration=duration,
        ram_bytes=ram_bytes, scratch_bytes=scratch_bytes,
        sample_from_first_audio=sample_from_first_audio, artwork=artwork,
        sample_picture=picture if sample_video else None,
    )


def format_plan(plan: Plan, *, verbose: bool = True) -> str:
    primary = plan.primary
    audio = [stream for stream in plan.media.streams if stream.kind == "audio"]
    language = primary.tags.get("language", "unknown")
    lines = [
        f"Input: {plan.media.path}",
        f"Selected audio: stream {primary.index}, codec {primary.codec}, "
        f"{primary.channels} channel(s), language {language}",
    ]
    if len(audio) > 1:
        defaults = [stream for stream in audio if "default" in stream.dispositions]
        if len(defaults) != 1:
            lines.append("Warning: audio choice is ambiguous; use --audio-stream-index to override.")
        else:
            lines.append("Use --audio-stream-index to choose another audio stream.")
    others = [stream for stream in plan.media.streams if stream.index != primary.index]
    if plan.sample_picture is not None:
        picture = plan.sample_picture
        lines.append(f"Video to copy: stream {picture.index} ({picture.codec})")
        omitted = ", ".join(f"{s.index} ({s.kind}/{s.codec})" for s in plan.omitted_streams) or "none"
        lines.append(f"Other streams omitted: {omitted}; chapters omitted: {len(plan.media.chapters)}")
        lines.append("Output profile: Matroska with 24-bit FLAC audio")
        lines.append("Video sample boundaries depend on keyframes and complete frames inside the requested range.")
    elif plan.video:
        copied = ", ".join(f"{s.index} ({s.kind}/{s.codec})" for s in others) or "none"
        lines.append(f"Other streams to copy: {copied}; chapters: {len(plan.media.chapters)}")
        lines.append("Output profile: Matroska with 24-bit FLAC audio")
    else:
        if plan.artwork:
            artwork = ", ".join(f"{s.index} ({s.codec})" for s in plan.artwork)
            lines.append(f"Artwork to copy: {artwork}")
        omitted = ", ".join(f"{s.index} ({s.kind}/{s.codec})" for s in plan.omitted_streams) or "none"
        lines.append(f"Other streams omitted: {omitted}; chapters omitted: {len(plan.media.chapters)}")
        profile = "32-bit float WAV" if plan.output.suffix.lower() == ".wav" else "24-bit FLAC"
        lines.append(f"Output profile: {profile}")
    lines.append(f"Destination: {plan.output}")
    if plan.sample is not None:
        lines.append(f"Sample request: start {plan.sample.start} s, "
                     f"duration {plan.sample.duration} s on source playback timeline")
        lines.append("Actual sample range and neighboring context are determined during decode.")
    if plan.output.suffix.lower() != ".wav":
        policy = f"up to {MAX_ATTENUATION_DB} dB" if plan.allow_attenuation else "off"
        lines.append(f"Automatic attenuation: {policy}")
    if verbose:
        if primary.sample_rate == SAMPLE_RATE:
            lines.append(f"Sample rate: {SAMPLE_RATE} Hz (no conversion expected)")
        else:
            source = f"{primary.sample_rate} Hz" if primary.sample_rate else "unknown"
            lines.append(f"Sample rate: {source} to {SAMPLE_RATE} Hz (conversion may be required)")
        lines.append(f"Requested device: {plan.device} (actual device determined at runtime)")
        lines.append(f"Chunking: {CHUNK_SECONDS} s, overlap {OVERLAP_SECONDS} s, "
                     f"context {CONTEXT_SECONDS} s per side, batch {CHUNK_BATCH_SIZE}")
        if plan.ram_bytes is None:
            lines.append("Estimated RAM and scratch: unknown (duration unavailable)")
        else:
            lines.append(f"Estimated RAM: {plan.ram_bytes} bytes; scratch: {plan.scratch_bytes} bytes "
                         "(provisional, not a resource guarantee)")
        lines.append("Deferred checks: decoded timeline and sample range, stream copy compatibility, "
                     "device availability, resource limits, and output validation.")
    if plan.duration is None:
        lines.append("Known blocker: input duration is unknown; use --sample for a bounded repair.")
    return "\n".join(lines)
