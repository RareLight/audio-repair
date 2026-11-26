import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import wave
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from unittest import mock

from audio_repair import cli
from audio_repair.media import Media, RepairError, Stream, probe, require_tools, select_audio
from audio_repair.planner import SampleRequest, build_plan, format_plan, seconds_arg


ROOT = Path(__file__).resolve().parents[2]


def write_test_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
        audio.writeframes(bytes(6 * 44100 * 2 * 2))


def stream(index=0, kind="audio", **changes):
    values = dict(
        index=index, kind=kind, codec="pcm_s16le", sample_rate=48000,
        channels=2, layout="stereo", time_base=Fraction(1, 48000),
        start=Fraction(0), duration=Fraction(60),
        dispositions=frozenset(), tags={},
    )
    values.update(changes)
    return Stream(**values)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.mp4"
        self.source.touch()
        self.media = Media(
            path=self.source, streams=(stream(),), start=Fraction(0),
            duration=Fraction(100), size=1000, format_name="mov",
            chapters=(), tags={},
        )

    def test_selection_default_ties_fallback_and_override(self):
        first = stream(2, tags={"language": "eng"})
        second = stream(5, dispositions=frozenset({"default"}), tags={"language": "fra"})
        third = stream(7, dispositions=frozenset({"default"}))
        media = replace(self.media, streams=(third, first, second))
        self.assertEqual(select_audio(media, None), second)
        self.assertEqual(select_audio(media, 2), first)
        self.assertIn("ambiguous", format_plan(build_plan(media)))
        self.assertIn("--audio-stream-index", format_plan(build_plan(media)))
        self.assertIn("language fra", format_plan(build_plan(media)))
        one_default = replace(media, streams=(first, second))
        self.assertEqual(select_audio(one_default, None), second)
        no_defaults = replace(media, streams=(third, first))
        no_defaults = replace(no_defaults, streams=tuple(
            replace(item, dispositions=frozenset()) for item in no_defaults.streams
        ))
        self.assertEqual(select_audio(no_defaults, None), first)
        for index in (0, 99):
            with self.subTest(index=index), self.assertRaisesRegex(RepairError, "not audio"):
                select_audio(media, index)
        video = stream(9, kind="video")
        with self.assertRaisesRegex(RepairError, "not audio"):
            select_audio(replace(media, streams=media.streams + (video,)), 9)

    def test_missing_unsupported_and_unknown_audio(self):
        with self.assertRaisesRegex(RepairError, "no audio"):
            select_audio(replace(self.media, streams=(stream(kind="video"),)), None)
        for changes in (
            {"channels": 6, "layout": "5.1"},
            {"channels": None},
            {"layout": "5.1"},
            {"channels": 1, "layout": "stereo"},
            {"codec": "unknown"},
        ):
            with self.subTest(changes=changes), self.assertRaises(RepairError):
                select_audio(replace(self.media, streams=(stream(**changes),)), None)
        self.assertEqual(select_audio(replace(
            self.media, streams=(stream(layout=None),)), None).channels, 2)

    def test_attached_picture_is_not_video_and_real_video_is(self):
        picture = stream(3, kind="video", codec="mjpeg",
                         dispositions=frozenset({"attached_pic"}))
        media = replace(self.media, streams=(picture, stream()))
        plan = build_plan(media)
        self.assertFalse(plan.video)
        self.assertEqual(plan.output.name, "source.repaired.flac")
        self.assertEqual(plan.artwork, (picture,))
        self.assertIn("Artwork to copy: 3 (mjpeg)", format_plan(plan))
        self.assertEqual(plan.omitted_streams, ())
        self.assertEqual(build_plan(media, audio_only=True).omitted_streams, (picture,))
        self.assertEqual(build_plan(media, output=self.root / "out.wav").omitted_streams, (picture,))
        self.assertEqual(build_plan(media, sample=SampleRequest(None, Fraction(30))).omitted_streams,
                         (picture,))
        video = stream(4, kind="video", codec="h264")
        media = replace(media, streams=media.streams + (video,))
        plan = build_plan(media)
        self.assertTrue(plan.video)
        self.assertEqual(plan.output.name, "source.repaired.mkv")
        self.assertIn("4 (video/h264)", format_plan(plan))
        self.assertFalse(build_plan(media, audio_only=True).video)
        self.assertFalse(build_plan(media, sample=SampleRequest(None, Fraction(30))).video)

    def test_sample_start_naming_cap_and_estimates(self):
        audio = stream(start=Fraction(0), duration=Fraction(1))
        media = replace(self.media, start=Fraction(-2), streams=(audio,))
        plan = build_plan(media, sample=SampleRequest(None, Fraction(30)))
        self.assertEqual(plan.sample, SampleRequest(Fraction(2), Fraction(30)))
        self.assertEqual(plan.output.name, "source.sample-2000-30000.repaired.flac")
        self.assertEqual(plan.duration, Fraction(32))
        pcm = 32 * 44100 * 2 * 4
        self.assertEqual(plan.ram_bytes, 5 * pcm + 8 * 2**30)
        self.assertEqual(plan.scratch_bytes, 4 * pcm + 64 * 2**20)
        explicit = build_plan(media, sample=SampleRequest(Fraction(0), Fraction(3, 2)))
        self.assertEqual(explicit.output.name, "source.sample-0-1500.repaired.flac")
        self.assertIn("Actual sample range", format_plan(explicit))
        self.assertIn("duration 3/2 s", format_plan(explicit))
        huge_start = Fraction(10**400)
        huge = build_plan(media, output=self.root / "huge.flac",
                          sample=SampleRequest(huge_start, Fraction(1)))
        self.assertIn(f"start {huge_start} s", format_plan(huge))
        missing_start = replace(media, streams=(replace(audio, start=None),))
        self.assertEqual(build_plan(missing_start, sample=SampleRequest(None, Fraction(1))).sample.start, 0)
        for duration in (Fraction(0), Fraction(-1), Fraction(121)):
            with self.subTest(duration=duration), self.assertRaisesRegex(RepairError, "duration"):
                build_plan(media, sample=SampleRequest(None, duration))
        self.assertEqual(build_plan(media, sample=SampleRequest(None, Fraction(120))).sample.duration, 120)
        with self.assertRaisesRegex(RepairError, "start"):
            build_plan(media, sample=SampleRequest(Fraction(-1), Fraction(1)))

    def test_video_sample_plan_needs_picture_and_mkv(self):
        sample = SampleRequest(Fraction(1), Fraction(30))
        with self.assertRaisesRegex(RepairError, "requires a video input"):
            build_plan(self.media, sample=sample, sample_video=True)
        picture = stream(1, kind="video", codec="h264")
        secondary = stream(2)
        media = replace(self.media, streams=(stream(), picture, secondary))
        plan = build_plan(media, sample=sample, sample_video=True)
        self.assertTrue(plan.video)
        self.assertEqual(plan.sample_picture, picture)
        self.assertEqual(plan.omitted_streams, (secondary,))
        self.assertEqual(plan.output.name, "source.sample-1000-30000.repaired.mkv")
        self.assertIn("Video to copy: stream 1", format_plan(plan))
        self.assertIn("Other streams omitted: 2", format_plan(plan))
        with self.assertRaisesRegex(RepairError, "output must use .mkv"):
            build_plan(media, sample=sample, sample_video=True, output=self.root / "out.flac")

    def test_full_estimates_and_unknown_duration(self):
        video = stream(1, kind="video", codec="h264")
        media = replace(self.media, streams=(stream(), video))
        plan = build_plan(media)
        pcm = 60 * 44100 * 2 * 4
        self.assertEqual(plan.duration, 60)
        self.assertEqual(plan.ram_bytes, 5 * pcm + 8 * 2**30)
        self.assertEqual(plan.scratch_bytes, 4 * pcm + 1000 + 64 * 2**20)
        self.assertEqual(build_plan(self.media).scratch_bytes, 3 * pcm + 64 * 2**20)
        no_duration = replace(media, duration=None, streams=(stream(duration=None), video))
        plan = build_plan(no_duration)
        self.assertIsNone(plan.ram_bytes)
        self.assertIsNone(plan.scratch_bytes)
        self.assertIn("Known blocker: input duration is unknown", format_plan(plan))
        long_media = replace(self.media, streams=(stream(duration=Fraction(7201)),))
        long_plan = build_plan(long_media)
        self.assertEqual(long_plan.duration, 7201)
        self.assertNotIn("Known blocker", format_plan(long_plan))

    def test_output_extensions_and_attenuation(self):
        video = replace(self.media, streams=(stream(), stream(1, kind="video")))
        with self.assertRaisesRegex(RepairError, "Video output must"):
            build_plan(video, output=self.root / "out.mp4")
        with self.assertRaisesRegex(RepairError, "Audio output must"):
            build_plan(self.media, output=self.root / "out.mkv")
        self.assertEqual(build_plan(video, audio_only=True, output=self.root / "out.wav").output.suffix, ".wav")
        with self.assertRaisesRegex(RepairError, "float WAV"):
            build_plan(self.media, output=self.root / "out.wav", allow_attenuation=True)
        self.assertTrue(build_plan(self.media, allow_attenuation=True).allow_attenuation)
        self.assertTrue(build_plan(self.media).allow_attenuation)
        self.assertFalse(build_plan(self.media, output=self.root / "out.wav").allow_attenuation)

    def test_path_conflicts_and_missing_parent(self):
        with self.assertRaisesRegex(RepairError, "input"):
            build_plan(self.media, output=self.source)
        link = self.root / "hardlink.flac"
        os.link(self.source, link)
        with self.assertRaisesRegex(RepairError, "input"):
            build_plan(self.media, output=link)
        dangling = self.root / "dangling.flac"
        dangling.symlink_to(self.root / "missing")
        with self.assertRaisesRegex(RepairError, "already exists"):
            build_plan(self.media, output=dangling)
        existing = self.root / "existing.flac"
        existing.touch()
        with self.assertRaisesRegex(RepairError, "already exists"):
            build_plan(self.media, output=existing)
        with self.assertRaisesRegex(RepairError, "parent"):
            build_plan(self.media, output=self.root / "missing" / "out.flac")
        loop = self.root / "loop"
        loop.symlink_to(loop)
        with self.assertRaisesRegex(RepairError, "parent"):
            build_plan(self.media, output=loop / "out.flac")

    def test_output_uses_lexical_input_name_and_resolved_parent(self):
        alias = self.root / "alias.wav"
        alias.symlink_to(self.source)
        media = replace(self.media, path=alias)
        self.assertEqual(build_plan(media).output, self.root.resolve() / "alias.repaired.flac")
        subdir = self.root / "real"
        subdir.mkdir()
        (self.root / "alias-dir").symlink_to(subdir)
        plan = build_plan(media, output=self.root / "alias-dir" / "out.flac")
        self.assertEqual(plan.output, subdir.resolve() / "out.flac")


class TimeTests(unittest.TestCase):
    def test_time_values_are_exact(self):
        for value, expected in (
            ("0", Fraction(0)), (".25", Fraction(1, 4)),
            ("00:12:30.125", Fraction(6001, 8)),
            ("100:59:59.5", Fraction(727199, 2)),
        ):
            with self.subTest(value=value):
                self.assertEqual(seconds_arg(value), expected)

    def test_reject_bad_time_values(self):
        for value in ("-1", "+1", "nan", "inf", "1e2", "1.", "1:60:00",
                      "1:00:60", "1:2:03", "", " 1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                seconds_arg(value)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "input.wav"
        self.path.touch()
        self.data = {
            "streams": [{
                "index": 2, "codec_type": "audio", "codec_name": "flac",
                "sample_rate": "48000", "channels": 2, "channel_layout": "stereo",
                "time_base": "1/48000", "start_pts": 4800, "start_time": "9.0",
                "duration_ts": 24000, "duration": "10.0",
                "disposition": {"default": 1, "forced": 0},
                "tags": {"language": "eng"},
            }],
            "format": {"format_name": "wav", "start_time": "0.2", "duration": "2.5",
                       "tags": {"title": "test"}},
            "chapters": [{"id": 0}],
        }

    def test_probe_parses_rational_fields_and_invokes_tool_safely(self):
        with mock.patch("audio_repair.media.run_tool", return_value=json.dumps(self.data)) as run:
            media = probe(self.path)
        args = run.call_args.args[0]
        self.assertEqual(args, [
            "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
            "-show_streams", "-show_format", "-show_chapters", "-of", "json",
            str(self.path.absolute()),
        ])
        self.assertEqual(media.streams[0].start, Fraction(1, 10))
        self.assertEqual(media.streams[0].duration, Fraction(1, 2))
        self.assertEqual(media.streams[0].dispositions, frozenset({"default"}))
        self.assertEqual(media.streams[0].tags, {"language": "eng"})
        self.assertEqual(media.start, Fraction(1, 5))
        self.assertEqual(media.duration, Fraction(5, 2))
        self.assertEqual(media.tags, {"title": "test"})
        self.assertEqual(media.chapters, ({"id": 0},))

    def test_probe_fallback_and_na(self):
        item = self.data["streams"][0]
        item.update(start_pts="N/A", duration_ts="N/A", start_time="0.125", duration="N/A",
                    channel_layout=None)
        self.data["format"]["start_time"] = "N/A"
        with mock.patch("audio_repair.media.run_tool", return_value=json.dumps(self.data)):
            media = probe(self.path)
        self.assertEqual(media.streams[0].start, Fraction(1, 8))
        self.assertIsNone(media.streams[0].duration)
        self.assertEqual(media.start, 0)

    def test_probe_rejects_malformed_output(self):
        for value in (
            "{", "[]", "{}", json.dumps({**self.data, "streams": [{}]}),
            json.dumps({**self.data, "streams": [self.data["streams"][0]] * 2}),
            json.dumps({**self.data, "format": {"format_name": "wav", "duration": "NaN"}}),
        ):
            with self.subTest(value=value), mock.patch("audio_repair.media.run_tool", return_value=value):
                with self.assertRaisesRegex(RepairError, "Malformed ffprobe"):
                    probe(self.path)

    def test_rejects_urls_nonregular_playlist_and_playlist_format(self):
        for path in (Path("https://example.com/a.wav"), Path(self.temp.name), self.path.with_suffix(".m3u8")):
            with self.subTest(path=path), self.assertRaises(RepairError):
                probe(path)
        self.data["format"]["format_name"] = "concat"
        with mock.patch("audio_repair.media.run_tool", return_value=json.dumps(self.data)):
            with self.assertRaisesRegex(RepairError, "concat"):
                probe(self.path)

    def test_missing_tools_diagnostic(self):
        with mock.patch("audio_repair.media.shutil.which", return_value=None):
            with self.assertRaisesRegex(RepairError, "ffmpeg is required"):
                require_tools()


class CliTests(unittest.TestCase):
    def test_directory_queues_only_top_level_media_and_keeps_sibling_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            for path in (root / "a.wav", root / "b.recording", root / "old.repaired.wav",
                         nested / "hidden.wav"):
                write_test_wav(path)
            (root / "notes.txt").write_text("standalone notes", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as output, \
                 contextlib.redirect_stderr(io.StringIO()) as error:
                self.assertEqual(cli.main([str(root), str(root / "a.wav"), "--dry-run"]), 0)
            destinations = [line.removeprefix("Destination: ") for line in output.getvalue().splitlines()
                            if line.startswith("Destination: ")]
            self.assertEqual(destinations, [str(root.resolve() / f"{stem}.repaired.flac")
                                            for stem in ("a", "b")])
            self.assertIn("2 planned, 0 failed.", error.getvalue())

    def test_quoted_wildcard_and_explicit_file_run_serially_after_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            other = root / "other"
            other.mkdir()
            paths = [root / "a.wav", root / "b.wav", other / "c.wav"]
            for path in paths:
                write_test_wav(path)
            visited = []

            def execute(plan, **kwargs):
                visited.append(plan.media.path)
                if plan.media.path == paths[0]:
                    raise RepairError("Injected repair failure")
                self.assertEqual(plan.output.parent, plan.media.path.parent)
                plan.output.touch()
                return mock.Mock(output=plan.output, frames=264600, gain=1.0, sample_range=None)

            with mock.patch("audio_repair.execution.execute", side_effect=execute), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as error:
                self.assertEqual(cli.main([str(root / "*.wav"), str(paths[2])]), 1)
            self.assertEqual(visited, paths)
            self.assertFalse((root / "a.repaired.flac").exists())
            self.assertTrue((root / "b.repaired.flac").is_file())
            self.assertTrue((other / "c.repaired.flac").is_file())
            self.assertIn("[3/3]", error.getvalue())
            self.assertIn("2 repaired, 1 failed.", error.getvalue())

    def test_output_override_rejects_multiple_inputs_before_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a.wav", "b.wav"):
                write_test_wav(root / name)
            with mock.patch("audio_repair.execution.execute") as execute, \
                 contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as raised:
                    cli.main([str(root), "-o", str(root / "out.flac")])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("-o requires exactly one input file", error.getvalue())
            execute.assert_not_called()

    def test_sample_flags_and_syntax_errors(self):
        for args in (
            ["in.wav", "--start", "1"], ["in.wav", "--duration", "30"],
            ["in.wav", "--sample", "--duration", "121"],
            ["in.wav", "--sample", "--duration", "0"],
            ["in.wav", "--sample", "--start", "nan"],
            ["in.mp4", "--sample-video", "--sample"],
            ["in.mp4", "--sample-video", "--audio-only"],
            ["in.mp4", "--sample-video", "--duration", "121"],
            ["in.wav", "--allow-attenuation", "--no-attenuation"],
        ):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    cli.main(args)
                self.assertEqual(raised.exception.code, 2)

    def test_execution_error_has_no_traceback(self):
        fake_media = mock.Mock()
        fake_media.streams = []
        fake_media.chapters = ()
        fake_plan = mock.Mock(omitted_streams=(), media=fake_media)
        with mock.patch.object(cli, "require_tools"), mock.patch.object(cli, "probe", return_value=fake_media), \
             mock.patch.object(cli, "build_plan", return_value=fake_plan), \
             mock.patch("audio_repair.execution.execute", side_effect=RepairError("execution failed")), \
             contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(cli.main(["input.wav"]), 1)
        self.assertIn("execution failed", error.getvalue())
        self.assertNotIn("Traceback", error.getvalue())

    def test_execution_reports_selected_stream_and_sample_range(self):
        fake_media = mock.Mock(streams=[], chapters=())
        fake_plan = mock.Mock(omitted_streams=(), media=fake_media)
        fake_plan.primary.index = 2
        fake_plan.primary.codec = "aac"
        fake_plan.video = False
        result = mock.Mock(output=Path("/tmp/output.wav"), device="cpu", frames=44100,
                           sample_range=(Fraction(1, 3), Fraction(4, 3)), gain=1.0)
        for verbose in (False, True):
            with self.subTest(verbose=verbose), mock.patch.object(cli, "require_tools"), \
                 mock.patch.object(cli, "probe", return_value=fake_media), \
                 mock.patch.object(cli, "build_plan", return_value=fake_plan), \
                 mock.patch("audio_repair.execution.execute", return_value=result) as execute, \
                 contextlib.redirect_stdout(io.StringIO()) as output, \
                 contextlib.redirect_stderr(io.StringIO()) as error:
                self.assertEqual(cli.main(["input.wav"] + (["--verbose"] if verbose else [])), 0)
            self.assertIs(execute.call_args.kwargs["verbose"], verbose)
            self.assertEqual("Selected audio: stream 2 (aac)" in error.getvalue(), verbose)
            self.assertEqual("frames: 44100" in output.getvalue(), verbose)
            excerpt = "0.333333–1.333333" if verbose else "0.333–1.333"
            self.assertIn(f"Source excerpt: {excerpt} s", output.getvalue())

    def test_progress_bar_and_redirected_milestones(self):
        for tty in (False, True):
            with self.subTest(tty=tty):
                stream = io.StringIO()
                with mock.patch.object(stream, "isatty", return_value=tty):
                    status = cli.TerminalStatus(stream, verbose=False)
                for index in range(11):
                    status.chunks(index, 10)
                status.phase("Encoding audio")
                output = stream.getvalue()
                self.assertIn("100% (10/10 chunks)\nEncoding audio", output)
                self.assertEqual("\r" in output, tty)
                self.assertEqual(output.count("Repairing audio"), 11 if tty else 5)

    def test_fixture_dry_run_is_read_only_without_torch_or_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "input.wav"
            write_test_wav(source)
            target = Path(temp) / "preview.flac"
            env = os.environ.copy()
            env.update(PYTHONPATH=str(ROOT), HF_HOME=str(Path(temp) / "empty-cache"))
            code = """
import builtins
import socket
import sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'huggingface_hub'}:
        raise AssertionError('unexpected model import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
def no_network(*args, **kwargs):
    raise AssertionError('unexpected network access')
socket.socket.connect = no_network
from audio_repair.cli import main
status = main(sys.argv[1:])
assert 'torch' not in sys.modules
sys.exit(status)
"""
            result = subprocess.run(
                [sys.executable, "-B", "-c", code, str(source),
                 "--dry-run", "--sample", "--start", "1", "-o", str(target)],
                cwd=temp, env=env, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Selected audio: stream", result.stdout)
            self.assertIn("Sample request: start 1 s", result.stdout)
            self.assertNotIn("Chunking:", result.stdout)
            self.assertEqual(list(Path(temp).iterdir()), [source])


if __name__ == "__main__":
    unittest.main()
