from contextlib import closing
from dataclasses import dataclass
from fractions import Fraction
from math import floor
from pathlib import Path

from .media import Media, RepairError, Stream, probe, run_tool
from .planner import SAMPLE_RATE, Plan
from .timeline import (
    AudioSpan, _integer, _packet_duration, _records, _time_base, audio_span, first_packet_pts,
    require_aligned_audio_starts, verify_copy_timing,
)


def _tag(tags: dict[str, str], key: str) -> str:
    return next((value for name, value in tags.items() if name.lower() == key), "")


def _language(tags: dict[str, str]) -> str:
    value = _tag(tags, "language")
    return "" if value.lower() == "und" else value


def _offset(value: Fraction) -> str:
    microseconds = round(value * 1_000_000)
    sign = "-" if microseconds < 0 else ""
    whole, fraction = divmod(abs(microseconds), 1_000_000)
    return f"{sign}{whole}.{fraction:06d}"


def _chapter_time(chapter: dict, key: str) -> tuple[Fraction, Fraction]:
    try:
        tick = Fraction(str(chapter["time_base"]))
        if tick <= 0:
            raise ValueError("invalid chapter time base")
        return Fraction(int(chapter[key])) * tick, tick
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
        raise RepairError(f"Chapter has no usable {key} timestamp") from exc


def _verify_chapters(source: Media, result: Media, shift: Fraction) -> None:
    if len(source.chapters) != len(result.chapters):
        raise RepairError("Chapter count changed")
    for before, after in zip(source.chapters, result.chapters):
        for key in ("start", "end"):
            old, old_tick = _chapter_time(before, key)
            new, new_tick = _chapter_time(after, key)
            if abs(new - old - shift) > Fraction(1, 1000) + old_tick + new_tick:
                raise RepairError(f"Chapter {key} timing changed")
        old_tags = before.get("tags", {})
        new_tags = after.get("tags", {})
        if not isinstance(old_tags, dict) or not isinstance(new_tags, dict):
            raise RepairError("Invalid chapter metadata")
        if (_tag(old_tags, "title") != _tag(new_tags, "title")
                or _language(old_tags) != _language(new_tags)):
            raise RepairError("Chapter title or language changed")


def _verify_stream_metadata(original: Stream, output: Stream, primary: bool) -> None:
    if original.kind != output.kind:
        raise RepairError(f"Stream {original.index} changed type or order")
    if not primary and original.codec != output.codec:
        raise RepairError(f"Copied stream {original.index} changed codec")
    if original.dispositions != output.dispositions:
        raise RepairError(f"Stream {original.index} changed dispositions")
    if (_tag(original.tags, "title") != _tag(output.tags, "title")
            or _language(original.tags) != _language(output.tags)):
        raise RepairError(f"Stream {original.index} changed title or language")
    if original.kind == "attachment":
        for key in ("filename", "mimetype"):
            if _tag(original.tags, key) != _tag(output.tags, key):
                raise RepairError(f"Attachment {original.index} changed {key}")
    if original.kind == "video":
        actual = dict(output.display)
        expected = dict(original.display)
        if (not {"width", "height"} <= expected.keys()
                or any(actual.get(key) != value for key, value in expected.items())):
            raise RepairError(f"Video stream {original.index} changed display metadata")


def _verify_global_metadata(original: Media, output: Media) -> None:
    if (_tag(original.tags, "title") != _tag(output.tags, "title")
            or _language(original.tags) != _language(output.tags)):
        raise RepairError("Global title or language changed")
    if any(any(word in key.lower() for word in ("replaygain", "r128", "peak"))
           for key in output.tags):
        raise RepairError("Output retained stale global loudness or peak metadata")


def _verify(plan: Plan, result: Media, source_span: AudioSpan, frames: int) -> None:
    originals = sorted(plan.media.streams, key=lambda stream: stream.index)
    outputs = sorted(result.streams, key=lambda stream: stream.index)
    if len(originals) != len(outputs):
        raise RepairError("Stream count changed")
    require_aligned_audio_starts(result)
    pairs = list(zip(originals, outputs))
    real_video = next(
        ((before, after) for before, after in pairs
         if before.kind == "video" and "attached_pic" not in before.dispositions),
        None,
    )
    if real_video is None:
        raise RepairError("No real video stream to establish the common timeline")
    before_video, after_video = real_video
    if before_video.time_base is None or after_video.time_base is None:
        raise RepairError("Video has no usable time base")
    shift = first_packet_pts(result, after_video) - first_packet_pts(plan.media, before_video)
    if abs(shift) > before_video.time_base + after_video.time_base:
        raise RepairError("Video timeline was shifted by more than a timestamp tick")
    _verify_global_metadata(plan.media, result)
    _verify_chapters(plan.media, result, shift)
    for original, copied in pairs:
        primary = original.index == plan.primary.index
        _verify_stream_metadata(original, copied, primary)
        if primary:
            if any(any(word in key.lower() for word in ("replaygain", "r128", "peak"))
                   for key in copied.tags):
                raise RepairError("Repaired audio retained stale loudness or peak metadata")
            if (copied.codec != "flac" or copied.sample_rate != SAMPLE_RATE
                    or copied.channels != original.channels or copied.bits_per_raw_sample != 24):
                raise RepairError("Repaired audio is not 24-bit FLAC with the expected shape")
            output_span = audio_span(result, copied)
            if output_span.frames != frames or copied.time_base is None:
                raise RepairError("Repaired audio frame count or time base changed")
            tolerance = copied.time_base + Fraction(1, SAMPLE_RATE)
            if (abs(output_span.start - source_span.start - shift) > tolerance
                    or abs(output_span.end - source_span.end - shift) > tolerance):
                raise RepairError("Repaired audio changed its start or end timing")
        elif original.kind != "attachment":
            verify_copy_timing(plan.media, original, result, copied, shift)


def remux(
    plan: Plan, repaired_flac: Path, staged: Path, source_span: AudioSpan, frames: int,
) -> None:
    originals = sorted(plan.media.streams, key=lambda stream: stream.index)
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-n", "-copyts",
        "-protocol_whitelist", "file,pipe", "-i", str(plan.media.path),
        "-itsoffset", _offset(source_span.start),
        "-protocol_whitelist", "file,pipe", "-i", str(repaired_flac),
    ]
    for stream in originals:
        command.extend(["-map", "1:a:0" if stream.index == plan.primary.index else f"0:{stream.index}"])
    command.extend([
        "-c", "copy", "-map_metadata", "0", "-map_chapters", "0",
        "-avoid_negative_ts", "disabled", "-default_mode", "passthrough",
    ])
    for index, stream in enumerate(originals):
        command.extend([f"-disposition:{index}", "+".join(sorted(stream.dispositions)) or "0"])
        if stream.index == plan.primary.index:
            for key in ("language", "title"):
                value = _tag(stream.tags, key)
                if value:
                    command.extend([f"-metadata:s:{index}", f"{key}={value}"])
    for key in plan.media.tags:
        if any(word in key.lower() for word in ("replaygain", "r128", "peak")):
            command.extend(["-metadata", f"{key}="])
    command.extend(["-f", "matroska", str(staged)])
    try:
        run_tool(command)
        _verify(plan, probe(staged), source_span, frames)
    except RepairError as exc:
        detail = str(exc)
        if "--audio-only" not in detail:
            detail += "; use --audio-only"
        raise RepairError(f"Video preservation failed: {detail}") from exc


@dataclass(frozen=True)
class SamplePacket:
    pts: Fraction
    end: Fraction
    digest: str


def _sample_packets(
    media: Media, stream: Stream, start: Fraction, end: Fraction,
) -> tuple[list[SamplePacket], Fraction]:
    tick = _time_base(stream)
    packets = []
    fields = "pts,duration,flags,data_hash"
    with closing(_records(media, stream, "packets", fields,
                          interval=f"{_offset(start)}%", hash_data=True)) as records:
        for record in records:
            pts = _integer(record, "pts") * tick
            if not packets and (pts < start or "K" not in record.get("flags", "")):
                if pts >= end:
                    break
                continue
            duration = _packet_duration(record, tick)
            if duration is None:
                raise RepairError("Video sample needs a known duration for each copied frame")
            if pts + duration > end or (packets and pts < packets[0].pts):
                break
            digest = record.get("data_hash")
            if not digest:
                raise RepairError("Cannot verify video sample packet data")
            packets.append(SamplePacket(pts, pts + duration, digest))
    if not packets:
        raise RepairError("No complete video frames starting at a keyframe inside the sample range; "
                          "choose a different start or a longer duration")

    # A usable decode-order prefix must also contain every source frame in
    # presentation order, including when B-frames are reordered or frame rate varies.
    first = packets[0].pts
    by_pts = {packet.pts: index for index, packet in enumerate(packets)}
    if len(by_pts) != len(packets):
        raise RepairError("Video sample contains duplicate picture timestamps")
    count = 0
    last_index = -1
    presented = 0
    clip_end = first
    with closing(_records(media, stream, "frames", "best_effort_timestamp",
                          interval=f"{_offset(first)}%")) as records:
        for record in records:
            pts = _integer(record, "best_effort_timestamp") * tick
            if pts < first:
                continue
            index = by_pts.get(pts)
            if index is None:
                break
            presented += 1
            last_index = max(last_index, index)
            if presented == last_index + 1:
                count = presented
                clip_end = packets[index].end
            if presented == len(packets):
                break
    if not count:
        raise RepairError("Cannot find a continuous video sample inside the requested range")
    return packets[:count], clip_end


def prepare_video_sample(
    plan: Plan, start: Fraction, end: Fraction, clip: Path,
) -> tuple[Media, Fraction, Fraction]:
    picture = plan.sample_picture
    if picture is None:
        raise RepairError("A video stream is required for --sample-video")
    packets, clip_end = _sample_packets(plan.media, picture, start, end)
    clip_start = packets[0].pts
    seek = Fraction(floor(clip_start * 1_000_000), 1_000_000)
    run_tool([
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-n",
        "-protocol_whitelist", "file,pipe", "-seek_timestamp", "1", "-ss", _offset(seek),
        "-i", str(plan.media.path), "-map", f"0:{picture.index}", "-map_chapters", "-1",
        "-c:v", "copy", "-frames:v", str(len(packets)),
        "-disposition:v:0", "+".join(sorted(picture.dispositions)) or "0",
        "-default_mode", "passthrough", "-f", "matroska", str(clip),
    ])
    media = probe(clip)
    copied, = media.streams
    _verify_stream_metadata(picture, copied, False)
    tick = _time_base(copied)
    tolerance = tick + _time_base(picture)
    fields = "pts,duration,data_hash"
    with closing(_records(media, copied, "packets", fields, hash_data=True)) as records:
        for packet in packets:
            record = next(records, None)
            if record is None or record.get("data_hash") != packet.digest:
                raise RepairError("Stream copy did not preserve the selected video packets")
            pts = _integer(record, "pts") * tick
            if abs(pts - (packet.pts - clip_start)) > tolerance:
                raise RepairError("Stream copy changed the video sample timeline")
            if packet.end == clip_end:
                duration = _packet_duration(record, tick)
                if duration is None or abs(pts + duration - (clip_end - clip_start)) > tolerance:
                    raise RepairError("Stream copy changed the video sample end")
        if next(records, None) is not None:
            raise RepairError("Stream copy produced extra video packets")
    with closing(_records(media, copied, "frames", "best_effort_timestamp")) as records:
        decoded = [_integer(record, "best_effort_timestamp") * tick for record in records]
    timeline = sorted(packet.pts - clip_start for packet in packets)
    if len(decoded) != len(timeline) or any(abs(a - b) > tolerance for a, b in zip(decoded, timeline)):
        raise RepairError("Video sample did not decode to the expected frames")
    if decoded[0] != 0:
        raise RepairError("Video sample did not start at zero; cannot guarantee audio alignment")
    return media, clip_start, clip_end


def remux_video_sample(plan: Plan, clip: Media, audio: Path, staged: Path, frames: int) -> None:
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-n",
        "-protocol_whitelist", "file,pipe", "-i", str(clip.path),
        "-protocol_whitelist", "file,pipe", "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-map_chapters", "-1",
        "-disposition:a:0", "+".join(sorted(plan.primary.dispositions)) or "0",
        "-avoid_negative_ts", "disabled", "-default_mode", "passthrough",
    ]
    for key in ("language", "title"):
        value = _tag(plan.primary.tags, key)
        if value:
            command.extend(["-metadata:s:a:0", f"{key}={value}"])
    for key in clip.tags:
        if any(word in key.lower() for word in ("replaygain", "r128", "peak")):
            command.extend(["-metadata", f"{key}="])
    run_tool(command + ["-f", "matroska", str(staged)])
    result = probe(staged)
    picture, primary = result.streams
    _verify_stream_metadata(clip.streams[0], picture, False)
    _verify_stream_metadata(plan.primary, primary, True)
    _verify_global_metadata(plan.media, result)
    require_aligned_audio_starts(result)
    verify_copy_timing(clip, clip.streams[0], result, picture, Fraction(0))
    span = audio_span(result, primary)
    if (span.start != 0 or span.frames != frames or primary.codec != "flac"
            or primary.sample_rate != SAMPLE_RATE or primary.channels != plan.primary.channels
            or primary.bits_per_raw_sample != 24):
        raise RepairError("Remux changed the repaired audio sample timeline or format")
