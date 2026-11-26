import argparse
import math
import sys
from fractions import Fraction
from glob import glob, has_magic
from pathlib import Path

from .media import PLAYLIST_SUFFIXES, Media, RepairError, ToolExitError, probe, require_tools
from .planner import MAX_ATTENUATION_DB, SAMPLE_RATE, Plan, SampleRequest, build_plan, format_plan, seconds_arg


class TerminalStatus:
    def __init__(self, stream, *, verbose: bool):
        self.stream = stream
        self.interactive = stream.isatty() and not verbose
        self.active_line = False
        self.milestone = -1

    def finish(self) -> None:
        if self.active_line:
            print(file=self.stream, flush=True)
            self.active_line = False

    def phase(self, message: str) -> None:
        self.finish()
        print(f"{message}…", file=self.stream, flush=True)

    def chunks(self, completed: int, total: int) -> None:
        percent = completed * 100 // total
        if self.interactive:
            width = 20
            filled = completed * width // total
            bar = "#" * filled + "-" * (width - filled)
            count = f"{completed:>{len(str(total))}}/{total}"
            print(f"\rRepairing audio [{bar}] {percent:3d}% ({count} chunks)",
                  end="", file=self.stream, flush=True)
            self.active_line = True
            if completed == total:
                self.finish()
        elif percent // 25 > self.milestone:
            self.milestone = percent // 25
            print(f"Repairing audio: {percent}% ({completed}/{total} chunks)",
                  file=self.stream, flush=True)


def collect_inputs(inputs: list[Path], *, verbose: bool) -> tuple[list[Media], int]:
    candidates: dict[Path, tuple[Path, bool]] = {}
    failures = 0
    for argument in inputs:
        try:
            path = argument.expanduser()
            matches = [path]
            if has_magic(str(path)) and not path.exists() and not path.is_symlink():
                matches = [Path(match) for match in sorted(glob(str(path)))]
                if not matches:
                    raise RepairError("No paths matched the wildcard")
            for match in matches:
                if not match.is_dir():
                    candidates[match.absolute()] = (match, False)
                    continue
                for entry in sorted(match.iterdir()):
                    if (entry.suffix.lower() not in PLAYLIST_SUFFIXES
                            and not entry.stem.lower().endswith(".repaired") and entry.is_file()):
                        candidates.setdefault(entry.absolute(), (entry, True))
        except (RepairError, OSError, RuntimeError) as exc:
            report_error(exc, verbose=verbose, input_path=argument)
            failures += 1

    queue = []
    for path, discovered in candidates.values():
        try:
            media = probe(path)
        except ToolExitError as exc:
            if discovered:
                if verbose:
                    print(f"Skipping {path}: {exc}", file=sys.stderr)
                continue
            report_error(exc, verbose=verbose, input_path=path)
            failures += 1
        except RepairError as exc:
            report_error(exc, verbose=verbose, input_path=path)
            failures += 1
        else:
            if not discovered or any(stream.kind == "audio" for stream in media.streams):
                queue.append(media)
    return queue, failures


def report_error(exc: Exception, *, verbose: bool, input_path: Path | None = None) -> None:
    detail = str(exc)
    if not verbose and "\n" in detail:
        detail = detail.splitlines()[0] + " Run with --verbose for details."
    source = f"{input_path}: " if input_path is not None else ""
    print(f"audio-repair: {source}{detail}", file=sys.stderr)


def repair_file(plan: Plan, *, verbose: bool) -> None:
    try:
        from .execution import execute
    except ImportError as exc:
        raise RepairError(f"Cannot load audio execution dependencies: {exc}") from exc
    if verbose:
        print(f"Selected audio: stream {plan.primary.index} ({plan.primary.codec})", file=sys.stderr)
    omitted = plan.omitted_streams
    chapters = len(plan.media.chapters)
    if (not plan.video or plan.sample_picture is not None) and (omitted or chapters):
        print(f"Omitting {len(omitted)} other stream(s) and {chapters} chapter(s).", file=sys.stderr)
    status = TerminalStatus(sys.stderr, verbose=verbose)
    try:
        result = execute(plan, progress=status.phase, chunk_progress=status.chunks, verbose=verbose)
    finally:
        status.finish()
    print(f"Saved: {result.output}", flush=True)
    if verbose:
        print(f"Device: {result.device}; audio stream: {plan.primary.index}; "
              f"frames: {result.frames} ({result.frames / SAMPLE_RATE:.6f} s)")
    if result.sample_range is not None:
        start, end = map(float, result.sample_range)
        decimals = 6 if verbose else 3
        print(f"Source excerpt: {start:.{decimals}f}–{end:.{decimals}f} s")
    if result.gain != 1.0:
        print(f"Reduced audio level by {-20 * math.log10(result.gain):.2f} dB to prevent clipping.")
        if verbose:
            print(f"Applied gain: {result.gain:.17g}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repair local audio and video recordings one at a time.")
    parser.add_argument("inputs", type=Path, nargs="+", metavar="INPUT",
                        help="local files, wildcards, or top-level directories")
    parser.add_argument("-o", "--output", type=Path, help="output path for a single input")
    parser.add_argument("--dry-run", action="store_true", help="probe and print the plan only")
    samples = parser.add_mutually_exclusive_group()
    samples.add_argument("--sample", action="store_true", help="repair a short audio-only sample")
    samples.add_argument("--sample-video", action="store_true", help="remux a short sample with repaired audio")
    parser.add_argument("--start", type=seconds_arg, help="sample start on the source playback timeline")
    parser.add_argument("--duration", type=seconds_arg, help="sample duration (default 30 seconds; maximum 120)")
    parser.add_argument("--audio-only", action="store_true", help="omit video and other streams")
    parser.add_argument("--audio-stream-index", type=int, help="absolute input audio stream index")
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    attenuation = parser.add_mutually_exclusive_group()
    attenuation.add_argument(
        "--allow-attenuation", dest="allow_attenuation", action="store_true", default=None,
        help=f"allow up to {MAX_ATTENUATION_DB} dB of automatic attenuation (default for FLAC)",
    )
    attenuation.add_argument("--no-attenuation", dest="allow_attenuation", action="store_false",
                             help="require FLAC output at the repaired level")
    parser.add_argument("-v", "--verbose", action="store_true", help="show technical details and warnings")
    tokens = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(tokens) if "--" in tokens else parser.parse_intermixed_args(tokens)
    sampling = args.sample or args.sample_video
    if not sampling and (args.start is not None or args.duration is not None):
        parser.error("--start and --duration require --sample or --sample-video")
    if args.sample_video and args.audio_only:
        parser.error("--sample-video is incompatible with --audio-only")
    duration = args.duration if args.duration is not None else Fraction(30)
    if sampling and not 0 < duration <= 120:
        parser.error("sample duration must be positive and at most 120 seconds")
    try:
        require_tools()
        queue, failures = collect_inputs(args.inputs, verbose=args.verbose)
        if not queue:
            if not failures:
                report_error(RepairError("No eligible audio or video files found"), verbose=args.verbose)
            return 1
        if args.output is not None and len(queue) != 1:
            parser.error("-o requires exactly one input file")
        sample = SampleRequest(args.start, duration) if sampling else None
        batch = len(queue) + failures > 1
        completed = 0
        destinations = set()
        for index, media in enumerate(queue, 1):
            if batch and not args.dry_run:
                print(f"[{index}/{len(queue)}] {media.path}", file=sys.stderr, flush=True)
            try:
                plan = build_plan(
                    media, output=args.output, audio_only=args.audio_only, sample=sample,
                    sample_video=args.sample_video, audio_stream_index=args.audio_stream_index,
                    device=args.device, allow_attenuation=args.allow_attenuation,
                )
                if args.dry_run:
                    if plan.output in destinations:
                        raise RepairError(f"Output conflicts with another queued input: {plan.output}")
                    destinations.add(plan.output)
                    print(format_plan(plan, verbose=args.verbose), end="\n\n" if batch else "\n", flush=batch)
                else:
                    repair_file(plan, verbose=args.verbose)
                completed += 1
            except RepairError as exc:
                report_error(exc, verbose=args.verbose, input_path=media.path if batch else None)
                failures += 1
        if batch:
            label = "planned" if args.dry_run else "repaired"
            print(f"{completed} {label}, {failures} failed.", file=sys.stderr)
        return int(failures > 0)
    except RepairError as exc:
        report_error(exc, verbose=args.verbose)
        return 1
    except KeyboardInterrupt:
        print("audio-repair: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
