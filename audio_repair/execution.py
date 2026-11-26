import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from .media import RepairError, probe
from .planner import CONTEXT_SECONDS, FLAC_PICTURE_BLOCK_LIMIT, MAX_ATTENUATION_DB, SAMPLE_RATE, Plan
from .timeline import AudioSpan, audio_span, require_aligned_audio_starts


Backend = Callable[[Path, Path, str], str]
MAX_WAV_BYTES = 2**32 - 1
DECODE_WAV_LIMIT = MAX_WAV_BYTES - 1024 * 1024
MIN_MODEL_FRAMES = 442
BLOCK_FRAMES = 65536
FLAC_MAX = 1.0 - 2**-23


@dataclass(frozen=True)
class RunResult:
    output: Path
    device: str
    frames: int
    gain: float
    sample_range: tuple[Fraction, Fraction] | None


def _tool(args: list[str]) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise RepairError(f"Could not run {args[0]}: {exc}") from exc
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise RepairError(f"{args[0]} failed (exit {result.returncode}): {detail}")
    return result


def _physical_memory() -> int | None:
    try:
        pages, page_size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return pages * page_size
    except (ValueError, OSError, AttributeError):
        pass
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False,
        )
        value = int(result.stdout.strip()) if result.returncode == 0 else 0
        return value if value > 0 else None
    except (OSError, ValueError):
        return None


def check_resources(plan: Plan) -> None:
    if plan.duration is None:
        raise RepairError("Input duration is unknown; use --sample for a bounded repair")
    if plan.ram_bytes is None or plan.scratch_bytes is None:
        raise RepairError("Cannot estimate resources for this input; use --sample")
    try:
        free = shutil.disk_usage(plan.output.parent).free
    except OSError as exc:
        raise RepairError(f"Cannot check free space at {plan.output.parent}: {exc}") from exc
    if free < plan.scratch_bytes:
        raise RepairError(
            f"Insufficient disk space: need an estimated {plan.scratch_bytes} bytes, have {free}"
        )
    physical = _physical_memory()
    if physical is not None and plan.ram_bytes > physical:
        raise RepairError(
            f"Estimated RAM {plan.ram_bytes} bytes exceeds physical memory {physical} bytes; "
            "use --sample"
        )


def _check_actual_resources(plan: Plan, frames: int, *, sample_video_bytes: int = 0) -> None:
    pcm_bytes = frames * plan.primary.channels * 4
    ram_bytes = 5 * pcm_bytes + 8 * 2**30
    physical = _physical_memory()
    if physical is not None and ram_bytes > physical:
        raise RepairError(
            f"Decoded audio requires an estimated {ram_bytes} bytes of RAM, exceeding "
            f"physical memory {physical} bytes; use --sample"
        )
    copies = 4 if plan.sample_picture is not None else 3 if plan.sample is not None or plan.video else 2
    remaining = copies * pcm_bytes + 64 * 2**20
    remaining += len(plan.artwork) * FLAC_PICTURE_BLOCK_LIMIT
    if plan.video:
        remaining += sample_video_bytes if plan.sample_picture is not None else plan.media.size
    try:
        free = shutil.disk_usage(plan.output.parent).free
    except OSError as exc:
        raise RepairError(f"Cannot check free space at {plan.output.parent}: {exc}") from exc
    if free < remaining:
        raise RepairError(
            f"Insufficient disk space for decoded audio: need an estimated "
            f"{remaining} more bytes, have {free}"
        )


def publish(staged: Path, output: Path) -> None:
    try:
        with staged.open("rb") as source:
            os.fsync(source.fileno())
        os.link(staged, output)
    except FileExistsError as exc:
        raise RepairError(f"Output already exists; refusing to overwrite: {output}") from exc
    except OSError as exc:
        raise RepairError(f"Could not publish {output} without replacing an existing file: {exc}") from exc


def _info(path: Path):
    try:
        return sf.info(str(path))
    except (OSError, RuntimeError, ValueError) as exc:
        raise RepairError(f"Cannot read decoded audio {path}: {exc}") from exc


def _check_shape(path: Path, channels: int, *, expected_frames: int | None = None) -> int:
    info = _info(path)
    if info.samplerate != SAMPLE_RATE or info.channels != channels:
        raise RepairError(f"Audio shape mismatch in {path}: expected {SAMPLE_RATE} Hz and {channels} channels")
    if info.frames * channels * 4 + 4096 >= MAX_WAV_BYTES:
        raise RepairError("Decoded audio exceeds the classic WAV size limit; use --sample")
    if expected_frames is not None and info.frames != expected_frames:
        raise RepairError(f"Audio frame count mismatch in {path}: expected {expected_frames}, got {info.frames}")
    return info.frames


def _scan(path: Path, frames: int) -> tuple[float, float]:
    minimum, maximum, count = math.inf, -math.inf, 0
    try:
        with sf.SoundFile(str(path)) as source:
            for block in source.blocks(blocksize=BLOCK_FRAMES, dtype="float32", always_2d=True):
                if not np.isfinite(block).all():
                    raise RepairError(f"Audio contains nonfinite samples: {path}")
                minimum = min(minimum, float(block.min()))
                maximum = max(maximum, float(block.max()))
                count += len(block)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RepairError(f"Cannot read decoded audio {path}: {exc}") from exc
    if count != frames:
        raise RepairError(f"Decoded audio is incomplete: expected {frames} frames, read {count}")
    return minimum, maximum


def _decode_wav(command: list[str], output: Path) -> subprocess.CompletedProcess:
    result = _tool([*command, "-c:a", "pcm_f32le", "-f", "wav",
                    "-fs", str(DECODE_WAV_LIMIT), str(output)])
    # FFmpeg can stop at -fs and return success with a readable, truncated WAV.
    if output.stat().st_size >= DECODE_WAV_LIMIT:
        raise RepairError("Decoded audio reached the classic WAV size limit; use --sample")
    return result


def _decode_full(plan: Plan, scratch: Path, source_span: AudioSpan | None = None) -> Path:
    audio_count = sum(stream.kind == "audio" for stream in plan.media.streams)
    real_video = any(
        stream.kind == "video" and "attached_pic" not in stream.dispositions
        for stream in plan.media.streams
    )
    formats = set(plan.media.format_name.split(","))
    if audio_count == 1 and not real_video and plan.primary.sample_rate == SAMPLE_RATE and formats & {"wav", "flac"}:
        try:
            info = _info(plan.media.path)
        except RepairError:
            pass
        else:
            if info.samplerate == SAMPLE_RATE and info.channels == plan.primary.channels:
                return plan.media.path
    decoded = scratch / "decoded.wav"
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-n",
        "-protocol_whitelist", "file,pipe", "-i", str(plan.media.path),
        "-map", f"0:{plan.primary.index}",
    ]
    if source_span is not None and source_span.discard_tail:
        command.extend(["-af", f"atrim=end_sample={source_span.frames}"])
    command.extend(["-ar", str(SAMPLE_RATE)])
    _decode_wav(command, decoded)
    return decoded


def _showinfo(stderr: str, time_base: Fraction | None) -> tuple[int, int]:
    first = None
    total = 0
    tolerance = max(1, math.ceil((time_base or Fraction(1, SAMPLE_RATE)) * SAMPLE_RATE))
    for line in stderr.splitlines():
        if "ashowinfo" not in line or not re.search(r"\bn:\s*\d+", line):
            continue
        match = re.search(r"\bpts:\s*(-?\d+)\b.*\bnb_samples:\s*(\d+)\b", line)
        if not match:
            raise RepairError("Sample audio has missing or malformed presentation timestamps")
        pts, count = map(int, match.groups())
        if count <= 0:
            raise RepairError("Sample audio has an empty frame")
        if first is None:
            first = pts
        elif abs(pts - first - total) > tolerance:
            raise RepairError("Sample audio has discontinuous presentation timestamps")
        total += count
    if first is None:
        raise RepairError("Sample range contains no audio")
    return first, total


def _seek_seconds(value: Fraction) -> str:
    microseconds = math.floor(value * 1_000_000)
    return f"{microseconds // 1_000_000}.{microseconds % 1_000_000:06d}"


def prepare_sample(plan: Plan, scratch: Path) -> tuple[Path, int, int, tuple[Fraction, Fraction]]:
    request = plan.sample
    if request is None:
        raise RepairError("A sample request is required")
    start = math.ceil((plan.media.start + request.start) * SAMPLE_RATE)
    end = math.ceil((plan.media.start + request.start + request.duration) * SAMPLE_RATE)
    context = CONTEXT_SECONDS * SAMPLE_RATE
    output = scratch / "sample-context.wav"
    command = [
        "ffmpeg", "-nostdin", "-v", "info", "-xerror", "-n", "-copyts",
        "-protocol_whitelist", "file,pipe",
    ]
    seek = Fraction(start - context, SAMPLE_RATE) - 2
    seek_enabled = seek > max(Fraction(0), plan.media.start)
    if seek_enabled:
        command.extend(["-seek_timestamp", "1", "-ss", _seek_seconds(seek)])
    if plan.sample_from_first_audio:
        trim_start = start - context if seek_enabled else math.ceil(plan.media.start * SAMPLE_RATE)
        filtergraph = (
            f"aresample={SAMPLE_RATE},asettb=1/{SAMPLE_RATE},"
            f"atrim=start_pts={trim_start},ashowinfo,asetpts=PTS-STARTPTS,"
            f"atrim=end_sample={math.ceil(request.duration * SAMPLE_RATE) + context}"
        )
    else:
        trim_start = start - context
        filtergraph = (
            f"aresample={SAMPLE_RATE},asettb=1/{SAMPLE_RATE},"
            f"atrim=start_pts={trim_start}:end_pts={end + context},"
            "ashowinfo,asetpts=PTS-STARTPTS"
        )
    track = plan.primary
    if (track.codec == "aac" and "mp4" in plan.media.format_name.split(",")
            and track.start is not None and track.duration is not None and track.sample_rate):
        # The decoder may return padding past the end of the presented track.
        end_sample = (track.start + track.duration) * track.sample_rate
        if end_sample.denominator == 1:
            filtergraph = f"asettb=1/{track.sample_rate},atrim=end_pts={int(end_sample)},{filtergraph}"
    command.extend([
        "-i", str(plan.media.path), "-map", f"0:{plan.primary.index}",
        "-af", filtergraph,
    ])
    result = _decode_wav(command, output)
    first, shown_frames = _showinfo(result.stderr, plan.primary.time_base)
    frames = _check_shape(output, plan.primary.channels)
    if frames > shown_frames:
        raise RepairError("Sample decode produced audio without verified timestamps")
    if plan.sample_from_first_audio:
        tolerance = max(1, math.ceil(
            (plan.primary.time_base or Fraction(1, SAMPLE_RATE)) * SAMPLE_RATE
        ))
        if seek_enabled and first <= trim_start + tolerance:
            raise RepairError(
                "The first audio sample could not be determined from the estimated start; "
                "specify --start explicitly"
            )
        start = first
        end = start + math.ceil(request.duration * SAMPLE_RATE)
    keep_start, keep_end = max(start, first), min(end, first + frames)
    if keep_end <= keep_start:
        raise RepairError("Sample range contains no audio")
    sample_range = (
        Fraction(keep_start, SAMPLE_RATE) - plan.media.start,
        Fraction(keep_end, SAMPLE_RATE) - plan.media.start,
    )
    return output, keep_start - first, keep_end - keep_start, sample_range


def _trim(source: Path, destination: Path, start: int, frames: int) -> None:
    try:
        with sf.SoundFile(str(source)) as infile:
            infile.seek(start)
            with sf.SoundFile(str(destination), "x", samplerate=SAMPLE_RATE,
                              channels=infile.channels, format="WAV", subtype="FLOAT") as outfile:
                remaining = frames
                while remaining:
                    block = infile.read(min(remaining, BLOCK_FRAMES), dtype="float32", always_2d=True)
                    if not len(block):
                        raise RepairError("Repaired audio ended before the sample range")
                    outfile.write(block)
                    remaining -= len(block)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RepairError(f"Cannot trim repaired audio: {exc}") from exc


def _metadata_args(plan: Plan) -> list[str]:
    args = ["-map_metadata", "-1"]
    for key, value in plan.media.tags.items():
        lowered = key.lower()
        if any(word in lowered for word in ("replaygain", "r128", "peak")):
            continue
        if "=" in key or any(ord(char) < 32 for char in key):
            continue
        args.extend(["-metadata", f"{key}={value}"])
    return args


def _encode(plan: Plan, source: Path, staged: Path, gain: float, frames: int) -> None:
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-n",
        "-protocol_whitelist", "file,pipe", "-i", str(source),
    ]
    if plan.artwork:
        command.extend(["-protocol_whitelist", "file,pipe", "-i", str(plan.media.path)])
    command.extend(["-map", "0:a:0"])
    command.extend(["-map_metadata", "-1"] if plan.video else _metadata_args(plan))
    for index, cover in enumerate(plan.artwork):
        command.extend(["-map", f"1:{cover.index}", f"-map_metadata:s:v:{index}", f"1:s:{cover.index}"])
    if plan.artwork:
        command.extend(["-c:v", "copy", "-disposition:v", "attached_pic"])
    if gain != 1.0:
        command.extend(["-af", f"volume={gain:.17g}:precision=double"])
    if plan.video or plan.output.suffix.lower() == ".flac":
        command.extend(["-c:a", "flac", "-sample_fmt", "s32",
                        "-bits_per_raw_sample", "24", "-f", "flac"])
        subtype = "PCM_24"
    else:
        command.extend(["-c:a", "pcm_f32le", "-f", "wav"])
        subtype = "FLOAT"
    command.append(str(staged))
    _tool(command)
    _check_shape(staged, plan.primary.channels, expected_frames=frames)
    if _info(staged).subtype != subtype:
        raise RepairError(f"Unexpected encoded audio subtype in {staged}; expected {subtype}")
    if plan.artwork:
        covers = [stream for stream in probe(staged).streams if "attached_pic" in stream.dispositions]
        if [stream.codec for stream in covers] != [stream.codec for stream in plan.artwork]:
            raise RepairError("FLAC did not preserve all cover images; use -o out.wav to omit artwork")


def execute(
    plan: Plan, *, backend: Backend | None = None,
    progress: Callable[[str], None] | None = None,
    chunk_progress: Callable[[int, int], None] | None = None,
    verbose: bool = False,
) -> RunResult:
    check_resources(plan)
    notify = progress or (lambda message: None)
    try:
        with tempfile.TemporaryDirectory(prefix=".audio-repair-", dir=plan.output.parent) as directory:
            scratch = Path(directory)
            sample_range = None
            source_span = None
            sample_clip = None
            if plan.video:
                notify("Checking video and audio timing")
                try:
                    require_aligned_audio_starts(plan.media)
                    if plan.sample_picture is None:
                        source_span = audio_span(plan.media, plan.primary, trim_aac_tail=True)
                except RepairError as exc:
                    detail = str(exc)
                    if "--audio-only" not in detail:
                        detail += "; use --sample for audio only" if plan.sample else "; use --audio-only"
                    raise RepairError(f"Video audio timeline cannot be verified: {detail}") from exc
            notify("Reading audio")
            if plan.sample is not None:
                source, keep_start, keep_frames, sample_range = prepare_sample(plan, scratch)
                frames = _check_shape(source, plan.primary.channels)
                if plan.sample_picture is not None:
                    from .video import prepare_video_sample

                    notify("Preparing video preview")
                    audio_start, audio_end = (plan.media.start + time for time in sample_range)
                    sample_tick = Fraction(1, SAMPLE_RATE)
                    requested_start = (audio_start - sample_tick if plan.sample_from_first_audio
                                       else plan.media.start + plan.sample.start)
                    requested_end = (audio_start + plan.sample.duration if plan.sample_from_first_audio
                                     else requested_start + plan.sample.duration)
                    sample_clip, clip_start, clip_end = prepare_video_sample(
                        plan, max(requested_start, audio_start - sample_tick),
                        min(requested_end, audio_end + sample_tick), scratch / "clip.mkv",
                    )
                    first = max(0, round((clip_start - audio_start) * SAMPLE_RATE))
                    last = min(keep_frames, round((clip_end - audio_start) * SAMPLE_RATE))
                    keep_start += first
                    keep_frames = last - first
                    if keep_frames <= 0 or abs(Fraction(keep_frames, SAMPLE_RATE) - (clip_end - clip_start)) > sample_tick:
                        raise RepairError("Video sample has no matching audio range")
                    sample_range = (clip_start - plan.media.start, clip_end - plan.media.start)
            else:
                source = _decode_full(plan, scratch, source_span)
                keep_start = 0
                frames = keep_frames = _check_shape(source, plan.primary.channels)
            if source_span is not None:
                expected = Fraction(source_span.frames * SAMPLE_RATE, source_span.rate)
                if abs(frames - expected) > 1:
                    raise RepairError(
                        "Decoded audio sample count does not match its original timeline; use --audio-only"
                    )
            if frames < MIN_MODEL_FRAMES:
                raise RepairError(f"Audio is too short for Apollo: at least {MIN_MODEL_FRAMES} frames are required")
            video_bytes = sample_clip.size if sample_clip is not None else 0
            _check_actual_resources(plan, frames, sample_video_bytes=video_bytes)
            _scan(source, frames)
            repaired = scratch / "repaired.wav"
            if backend is None:
                from .backend import repair

                notify("Loading repair model")
                device = repair(source, repaired, plan.device, chunk_progress=chunk_progress, verbose=verbose)
            else:
                notify("Repairing audio")
                device = backend(source, repaired, plan.device)
            notify("Checking repaired audio")
            _check_shape(repaired, plan.primary.channels, expected_frames=frames)
            minimum, maximum = _scan(repaired, frames)
            if plan.sample is not None:
                trimmed = scratch / "trimmed.wav"
                _trim(repaired, trimmed, keep_start, keep_frames)
                repaired = trimmed
                minimum, maximum = _scan(repaired, keep_frames)
            gain = 1.0
            if (plan.video or plan.output.suffix.lower() == ".flac") and (minimum < -1.0 or maximum > FLAC_MAX):
                gain = 1.0 / max(-minimum, maximum / FLAC_MAX)
                reduction_db = -20 * math.log10(gain)
                limit_db = MAX_ATTENUATION_DB if plan.allow_attenuation else 0
                if reduction_db > limit_db:
                    wav_option = "--sample -o out.wav" if plan.sample else "--audio-only -o out.wav"
                    hint = f"Export float WAV with {wav_option}."
                    if not plan.allow_attenuation and reduction_db <= MAX_ATTENUATION_DB:
                        hint = f"Retry without --no-attenuation or export float WAV with {wav_option}."
                    raise RepairError(
                        f"Audio exceeds the 24-bit FLAC range and needs {reduction_db:.2f} dB of "
                        f"attenuation (limit {limit_db} dB). {hint}"
                    )
                gain = math.nextafter(gain, 0)
            notify("Encoding audio")
            if plan.video:
                from .video import remux, remux_video_sample
                encoded = scratch / "repaired.flac"
                _encode(plan, repaired, encoded, gain, keep_frames)
                staged = scratch / "staged.mkv"
                notify("Assembling and checking video")
                if sample_clip is not None:
                    remux_video_sample(plan, sample_clip, encoded, staged, keep_frames)
                else:
                    remux(plan, encoded, staged, source_span, keep_frames)
            else:
                staged = scratch / ("staged" + plan.output.suffix.lower())
                _encode(plan, repaired, staged, gain, keep_frames)
                _scan(staged, keep_frames)
            notify("Saving output")
            publish(staged, plan.output)
            return RunResult(plan.output, device, keep_frames, gain, sample_range)
    except RepairError:
        raise
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        raise RepairError(f"Audio repair failed: {exc}") from exc
