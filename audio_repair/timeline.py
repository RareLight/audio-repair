import itertools
import subprocess
import tempfile
from contextlib import closing
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator

from .media import Media, RepairError, Stream


@dataclass(frozen=True)
class AudioSpan:
    start: Fraction
    frames: int
    rate: int
    discard_tail: int = 0

    @property
    def end(self) -> Fraction:
        return self.start + Fraction(self.frames, self.rate)


def _records(
    media: Media, stream: Stream, kind: str, fields: str, *,
    interval: str | None = None, hash_data: bool = False,
) -> Iterator[dict[str, str]]:
    args = [
        "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
        "-select_streams", str(stream.index), f"-show_{kind}",
        "-show_entries", f"{kind[:-1]}={fields}", "-of", "compact=p=0:nk=0",
    ]
    if interval is not None:
        args.extend(["-read_intervals", interval])
    if hash_data:
        args.extend(["-show_data_hash", "sha256"])
    args.append(str(media.path))
    with tempfile.TemporaryFile(mode="w+b") as errors:
        try:
            process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=errors, text=True)
        except OSError as exc:
            raise RepairError(f"Cannot run ffprobe for stream {stream.index}: {exc}") from exc
        try:
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                record = {}
                for field in line.split("|"):
                    if not field:
                        continue
                    key, separator, value = field.partition("=")
                    if not separator or not key:
                        raise RepairError(f"Malformed ffprobe {kind} record for stream {stream.index}")
                    record[key] = value
                if record:
                    yield record
            code = process.wait()
            if code:
                errors.seek(0, 2)
                errors.seek(max(0, errors.tell() - 8192))
                detail = errors.read().decode("utf-8", "replace").strip() or "no diagnostic"
                raise RepairError(f"ffprobe failed for stream {stream.index}: {detail}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()


def _integer(record: dict[str, str], field: str) -> int:
    try:
        value = record[field]
        if not value or value == "N/A":
            raise ValueError("missing value")
        return int(value)
    except (KeyError, ValueError) as exc:
        raise RepairError(f"Missing or invalid {field} in ffprobe timeline") from exc


def _time_base(stream: Stream) -> Fraction:
    if stream.time_base is None or stream.time_base <= 0:
        raise RepairError(f"Stream {stream.index} has no usable time base")
    return stream.time_base


def audio_span(media: Media, stream: Stream, *, trim_aac_tail: bool = False) -> AudioSpan:
    rate = stream.sample_rate
    tick = _time_base(stream)
    if stream.kind != "audio" or rate is None or rate <= 0:
        raise RepairError(f"Stream {stream.index} has no usable audio sample rate")
    first = None
    count = 0
    tolerance = max(tick, Fraction(1, rate))
    fields = "best_effort_timestamp,nb_samples,duration,pkt_duration"
    with closing(_records(media, stream, "frames", fields)) as records:
        for record in records:
            timestamp = _integer(record, "best_effort_timestamp") * tick
            frames = _integer(record, "nb_samples")
            if frames <= 0:
                raise RepairError(f"Stream {stream.index} has an empty decoded audio frame")
            if first is None:
                first = timestamp
            elif abs(timestamp - (first + Fraction(count, rate))) > tolerance:
                raise RepairError(f"Stream {stream.index} has discontinuous or drifting audio timestamps")
            count += frames
    if first is None or not count:
        raise RepairError(f"Stream {stream.index} has no decoded audio frames")
    span = AudioSpan(first, count, rate)
    if "mp4" in media.format_name.split(",") and stream.duration is not None:
        replace_aac = trim_aac_tail and stream.codec == "aac"
        if replace_aac and stream.start == first:
            declared_frames = stream.duration * rate
            padding = count - declared_frames
            if declared_frames.denominator == 1 and 0 < padding < frames:
                # Older demuxers can expose a shortened final AAC packet
                # without telling the decoder to discard the trailing samples.
                duration_field = "duration" if record.get("duration") not in (None, "N/A") else "pkt_duration"
                last_duration = _integer(record, duration_field) * tick
                expected_end = first + stream.duration
                if last_duration == Fraction(frames - padding, rate) and timestamp + last_duration == expected_end:
                    return AudioSpan(first, int(declared_frames), rate, int(padding))
        duration_tolerance = tolerance if replace_aac else max(tick, Fraction(1, 1000))
        if abs(Fraction(count, rate) - stream.duration) > duration_tolerance:
            raise RepairError(
                f"Stream {stream.index} has unverifiable MP4 edits or decoder priming; "
                "use --audio-only"
            )
    return span


def first_packet_pts(media: Media, stream: Stream) -> Fraction:
    tick = _time_base(stream)
    with closing(_records(media, stream, "packets", "pts,duration")) as records:
        try:
            first = next(records)
        except StopIteration as exc:
            raise RepairError(f"Stream {stream.index} has no packets") from exc
    return _integer(first, "pts") * tick


def _first_frame_pts(media: Media, stream: Stream) -> Fraction:
    tick = _time_base(stream)
    with closing(_records(media, stream, "frames", "best_effort_timestamp")) as records:
        first = next(records, None)
    if first is None:
        raise RepairError(f"Stream {stream.index} has no decoded frames")
    return _integer(first, "best_effort_timestamp") * tick


def require_aligned_audio_starts(media: Media) -> None:
    video = next(
        (stream for stream in sorted(media.streams, key=lambda item: item.index)
         if stream.kind == "video" and "attached_pic" not in stream.dispositions), None,
    )
    if video is None:
        raise RepairError("No real video stream to establish the common timeline")
    video_start = _first_frame_pts(media, video)
    for stream in media.streams:
        if stream.kind == "audio":
            offset = _first_frame_pts(media, stream) - video_start
            if offset:
                raise RepairError(
                    f"Audio stream {stream.index} starts {offset} s relative to video stream "
                    f"{video.index}; preservation-mode video requires aligned starts"
                )


def _packet_duration(packet: dict[str, str], tick: Fraction) -> Fraction | None:
    if packet.get("duration") in (None, "N/A"):
        return None
    value = _integer(packet, "duration") * tick
    return value if value > 0 else None


def verify_copy_timing(
    source: Media, original: Stream, result: Media, copied: Stream, shift: Fraction,
) -> None:
    source_tick, output_tick = _time_base(original), _time_base(copied)
    tolerance = source_tick + output_tick
    if original.kind == "audio":
        before, after = audio_span(source, original), audio_span(result, copied)
        if (before.rate != after.rate or before.frames != after.frames
                or abs(after.start - before.start - shift) > tolerance
                or abs(after.end - before.end - shift) > tolerance):
            raise RepairError(f"Copied audio stream {original.index} changed its timeline")
        return
    last_video_pts = None
    last_video_end = None
    with closing(_records(source, original, "packets", "pts,duration")) as left, \
         closing(_records(result, copied, "packets", "pts,duration")) as right:
        for source_packet, result_packet in itertools.zip_longest(left, right):
            if source_packet is None or result_packet is None:
                raise RepairError(f"Copied stream {original.index} changed its packet count")
            before = _integer(source_packet, "pts") * source_tick
            after = _integer(result_packet, "pts") * output_tick
            if abs(after - before - shift) > tolerance:
                raise RepairError(f"Copied stream {original.index} changed packet timing")
            if original.kind == "subtitle":
                source_duration = _packet_duration(source_packet, source_tick)
                if source_duration is not None:
                    output_duration = _packet_duration(result_packet, output_tick)
                    if output_duration is None or abs(output_duration - source_duration) > tolerance:
                        raise RepairError(f"Copied subtitle stream {original.index} changed packet duration")
            if original.kind == "video" and (last_video_pts is None or before > last_video_pts):
                # Packet order can differ from presentation order when B-frames are present.
                last_video_pts = before
                source_duration = _packet_duration(source_packet, source_tick)
                output_duration = _packet_duration(result_packet, output_tick)
                last_video_end = None if source_duration is None else (
                    before + source_duration, None if output_duration is None else after + output_duration,
                )
    if original.kind == "video":
        if last_video_pts is None:
            raise RepairError(f"Copied video stream {original.index} has no packets")
        if last_video_end is not None:
            before, after = last_video_end
            if after is None or abs(after - before - shift) > tolerance:
                raise RepairError(f"Copied video stream {original.index} changed its end timing")
