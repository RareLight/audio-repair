import contextlib
import hashlib
import io
import os
import subprocess
import tempfile
import threading
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import soundfile as sf

from audio_repair import backend, execution
from audio_repair.media import RepairError, probe
from audio_repair.planner import SAMPLE_RATE, SampleRequest, build_plan


def identity(source, destination, device):
    samples, rate = sf.read(str(source), dtype="float32", always_2d=True)
    sf.write(str(destination), samples, rate, subtype="FLOAT")
    return "identity"


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patch = mock.patch("audio_repair.execution._physical_memory", return_value=64 * 2**30)
        patch.start()
        self.addCleanup(patch.stop)

    def source(self, name="input.wav", *, rate=SAMPLE_RATE, seconds=2,
               subtype="PCM_16", values=None):
        path = self.root / name
        if values is None:
            time = np.arange(rate * seconds, dtype=np.float64) / rate
            values = np.column_stack((0.25 * np.sin(time * 8), 0.2 * np.cos(time * 5)))
        sf.write(str(path), values, rate, subtype=subtype)
        return path

    def test_native_integer_flac_roundtrip_and_source_unchanged(self):
        path = self.source()
        original_hash = hashlib.sha256(path.read_bytes()).digest()
        seen = []

        def recording_backend(source, destination, device):
            seen.append(source)
            return identity(source, destination, device)

        plan = build_plan(probe(path))
        result = execution.execute(plan, backend=recording_backend)
        self.assertEqual(seen, [path])
        self.assertEqual(result.frames, 2 * SAMPLE_RATE)
        self.assertEqual(result.device, "identity")
        self.assertEqual(result.gain, 1.0)
        self.assertEqual(sf.info(str(result.output)).subtype, "PCM_24")
        source_pcm, _ = sf.read(str(path), dtype="int16", always_2d=True)
        result_pcm, _ = sf.read(str(result.output), dtype="int16", always_2d=True)
        np.testing.assert_array_equal(source_pcm, result_pcm)
        self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), original_hash)
        self.assertEqual(sorted(item.name for item in self.root.iterdir()),
                         ["input.repaired.flac", "input.wav"])

    def test_full_ffmpeg_resample(self):
        path = self.source(rate=48000)
        seen = []

        def recording_backend(source, destination, device):
            seen.append((source.name, sf.info(str(source)).samplerate))
            return identity(source, destination, device)

        result = execution.execute(build_plan(probe(path)), backend=recording_backend)
        self.assertEqual(seen, [("decoded.wav", SAMPLE_RATE)])
        self.assertEqual(result.frames, SAMPLE_RATE * 2)
        self.assertEqual(sf.info(str(result.output)).frames, SAMPLE_RATE * 2)

    def test_native_info_failure_uses_ffmpeg_decode(self):
        path = self.source()
        original_info = execution._info
        seen = []

        def info(candidate):
            if candidate == path:
                raise RepairError("native decoder cannot read source")
            return original_info(candidate)

        def recording_backend(source, destination, device):
            seen.append(source.name)
            return identity(source, destination, device)

        plan = build_plan(probe(path))
        with mock.patch("audio_repair.execution._info", side_effect=info):
            result = execution.execute(plan, backend=recording_backend)
        self.assertEqual(seen, ["decoded.wav"])
        self.assertEqual(result.frames, SAMPLE_RATE * 2)

    def test_float_wav_out_of_range_and_flac_policy(self):
        values = np.tile(np.array([-1.3, 1.2], dtype=np.float32), (SAMPLE_RATE, 1))
        path = self.source(subtype="FLOAT", values=values)
        media = probe(path)
        wav = execution.execute(build_plan(media, output=self.root / "out.wav"), backend=identity)
        actual, _ = sf.read(str(wav.output), dtype="float32", always_2d=True)
        np.testing.assert_array_equal(values, actual)
        with self.assertRaisesRegex(RepairError, "exceeds the 24-bit FLAC range"):
            execution.execute(
                build_plan(media, output=self.root / "rejected.flac", allow_attenuation=False),
                backend=identity,
            )
        self.assertFalse((self.root / "rejected.flac").exists())
        result = execution.execute(
            build_plan(media, output=self.root / "attenuated.flac"),
            backend=identity,
        )
        self.assertLess(result.gain, 1.0)
        self.assertAlmostEqual(result.gain, 1 / 1.3, places=5)
        actual, _ = sf.read(str(result.output), dtype="float32", always_2d=True)
        self.assertGreaterEqual(float(actual.min()), -1.0)
        self.assertLessEqual(float(actual.max()), execution.FLAC_MAX)
        np.testing.assert_allclose(actual, values * result.gain, atol=2**-23)

    def test_flac_attenuation_limit_cannot_be_overridden(self):
        for peak, passes in ((1.99, True), (2.0, False)):
            with self.subTest(peak=peak):
                values = np.full((SAMPLE_RATE, 2), peak, dtype=np.float32)
                source = self.source(name=f"{peak}.wav", subtype="FLOAT", values=values)
                plan = build_plan(probe(source), allow_attenuation=True)
                if passes:
                    result = execution.execute(plan, backend=identity)
                    self.assertGreater(result.gain, 10 ** (-6 / 20))
                else:
                    with self.assertRaisesRegex(RepairError, r"needs 6\.02 dB.*limit 6 dB"):
                        execution.execute(plan, backend=identity)
                    self.assertFalse(plan.output.exists())

    def test_nonfinite_input_output_and_shape_mismatch(self):
        values = np.zeros((SAMPLE_RATE, 2), dtype=np.float32)
        values[20, 0] = np.nan
        bad_input = self.source(name="bad.wav", subtype="FLOAT", values=values)
        called = mock.Mock(side_effect=AssertionError("must not run"))
        with self.assertRaisesRegex(RepairError, "nonfinite"):
            execution.execute(build_plan(probe(bad_input)), backend=called)
        called.assert_not_called()
        path = self.source()
        media = probe(path)

        def broken(kind):
            def do_repair(source, destination, device):
                samples, rate = sf.read(str(source), dtype="float32", always_2d=True)
                if kind == "nan":
                    samples[0, 0] = np.inf
                elif kind == "frames":
                    samples = samples[:-1]
                else:
                    samples = samples[:, :1]
                sf.write(str(destination), samples, rate, subtype="FLOAT")
                return "identity"
            return do_repair

        for kind in ("nan", "frames", "channels"):
            output = self.root / f"{kind}.wav"
            with self.subTest(kind=kind), self.assertRaises(RepairError):
                execution.execute(build_plan(media, output=output), backend=broken(kind))
            self.assertFalse(output.exists())

    def test_too_short_and_resource_guards_before_backend(self):
        path = self.source(values=np.zeros((441, 2), dtype=np.float32))
        plan = build_plan(probe(path))
        called = mock.Mock()
        with self.assertRaisesRegex(RepairError, "too short"):
            execution.execute(plan, backend=called)
        called.assert_not_called()
        with mock.patch("audio_repair.execution.shutil.disk_usage",
                        return_value=mock.Mock(free=0)):
            with self.assertRaisesRegex(RepairError, "Insufficient disk space"):
                execution.execute(plan, backend=called)
        self.assertEqual([item.name for item in self.root.iterdir()], ["input.wav"])
        with self.assertRaisesRegex(RepairError, "unknown"):
            execution.check_resources(replace(plan, duration=None))
        with mock.patch("audio_repair.execution._physical_memory", return_value=1):
            with self.assertRaisesRegex(RepairError, "physical memory"):
                execution.check_resources(plan)

    def test_capped_ffmpeg_decode_fails_without_publishing_a_partial_file(self):
        path = self.source(rate=48000, seconds=2)
        plan = build_plan(probe(path))
        called = mock.Mock()
        with mock.patch.object(execution, "DECODE_WAV_LIMIT", 80_000):
            with self.assertRaisesRegex(RepairError, "reached the classic WAV size limit"):
                execution.execute(plan, backend=called)
        called.assert_not_called()
        self.assertFalse(plan.output.exists())

    def test_classic_wav_size_boundary(self):
        limit = (execution.MAX_WAV_BYTES - 4096 + 7) // 8
        with mock.patch.object(execution, "_info") as info:
            info.return_value = SimpleNamespace(samplerate=44100, channels=2, frames=limit - 1)
            self.assertEqual(execution._check_shape(Path("stub.wav"), 2), limit - 1)
            info.return_value.frames = limit
            with self.assertRaisesRegex(RepairError, "classic WAV size limit"):
                execution._check_shape(Path("stub.wav"), 2)

    def test_actual_resource_estimate_catches_short_headers(self):
        path = self.source()
        plan = replace(build_plan(probe(path)), ram_bytes=1)
        called = mock.Mock()
        with mock.patch("audio_repair.execution._physical_memory", return_value=8 * 2**30 + 1):
            with self.assertRaisesRegex(RepairError, "Decoded audio requires"):
                execution.execute(plan, backend=called)
        with mock.patch("audio_repair.execution.shutil.disk_usage",
                        side_effect=[mock.Mock(free=10**12), mock.Mock(free=0)]):
            with self.assertRaisesRegex(RepairError, "Insufficient disk space for decoded"):
                execution.execute(plan, backend=called)
        called.assert_not_called()
        self.assertFalse(plan.output.exists())

    def test_tag_copy_omits_stale_peak_metadata(self):
        path = self.source()
        tagged = self.root / "tagged.flac"
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(path),
            "-metadata", "title=Concert", "-metadata", "REPLAYGAIN_TRACK_GAIN=-5 dB",
            "-metadata", "R128_TRACK_GAIN=42", "-c:a", "flac", str(tagged),
        ], check=True, capture_output=True)
        result = execution.execute(build_plan(probe(tagged)), backend=identity)
        tags = {key.lower(): value for key, value in probe(result.output).tags.items()}
        self.assertEqual(tags.get("title"), "Concert")
        self.assertFalse(any("replaygain" in key or "r128" in key or "peak" in key
                             for key in tags))

    def test_flac_copies_attached_artwork_without_reencoding(self):
        path = self.source()
        covered = self.root / "covered.flac"
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
            "-f", "lavfi", "-i", "color=orange:s=32x32:d=0.04",
            "-map", "0:a", "-map", "1:v", "-c:a", "flac", "-c:v", "png",
            "-metadata:s:v:0", "title=Artwork", "-metadata:s:v:0", "comment=Cover (front)",
            "-disposition:v", "attached_pic", str(covered),
        ], check=True, capture_output=True)
        plan = build_plan(probe(covered))
        result = execution.execute(plan, backend=identity)
        cover, = [stream for stream in probe(result.output).streams
                  if "attached_pic" in stream.dispositions]
        self.assertEqual(cover.tags, plan.artwork[0].tags)
        images = []
        for media in (covered, result.output):
            images.append(subprocess.run([
                "ffmpeg", "-nostdin", "-v", "error", "-i", str(media),
                "-map", "0:v:0", "-c:v", "copy", "-f", "image2pipe", "pipe:1",
            ], check=True, capture_output=True).stdout)
        self.assertTrue(images[0])
        self.assertEqual(images[0], images[1])

    def delayed_media(self, delay=2):
        duration = 12
        samples = np.arange(SAMPLE_RATE * duration, dtype=np.float32)
        samples = np.column_stack((samples / len(samples) * 0.4 - 0.2,
                                   samples / len(samples) * 0.2 - 0.1))
        source = self.source(name="source.wav", values=samples)
        container = self.root / "delayed.mkv"
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=black:s=16x16:r=2:d={duration + delay}",
            "-itsoffset", str(delay), "-i", str(source),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "mpeg4",
            "-c:a", "pcm_s16le", str(container),
        ], check=True, capture_output=True)
        return container, source

    def test_late_sample_seeks_and_retains_context_and_source(self):
        container, source = self.delayed_media()
        digest = hashlib.sha256(container.read_bytes()).digest()
        media = probe(container)
        plan = build_plan(media, sample=SampleRequest(Fraction(10), Fraction(3)),
                          output=self.root / "sample.wav")
        observed = []
        commands = []
        original_tool = execution._tool

        def recording_tool(args):
            commands.append(args)
            return original_tool(args)

        def recording_backend(path, destination, device):
            observed.append(sf.info(str(path)).frames)
            return identity(path, destination, device)

        with mock.patch("audio_repair.execution._tool", side_effect=recording_tool):
            result = execution.execute(plan, backend=recording_backend)
        self.assertIn("-seek_timestamp", commands[0])
        self.assertIn("-ss", commands[0])
        self.assertEqual(len(observed), 1)
        self.assertLessEqual(abs(observed[0] - 5 * SAMPLE_RATE), 45)
        self.assertEqual(result.frames, 3 * SAMPLE_RATE)
        self.assertEqual(result.sample_range, (Fraction(10), Fraction(13)))
        expected, _ = sf.read(str(source), start=8 * SAMPLE_RATE, frames=3 * SAMPLE_RATE,
                              dtype="float32", always_2d=True)
        actual, _ = sf.read(str(result.output), dtype="float32", always_2d=True)
        np.testing.assert_allclose(actual, expected, atol=1 / 32768)
        self.assertEqual(hashlib.sha256(container.read_bytes()).digest(), digest)

    def test_default_first_sample_and_early_late_out_of_range(self):
        container, _ = self.delayed_media()
        media = probe(container)
        cases = [
            (None, Fraction(3), (Fraction(2), Fraction(5)), 3),
            (Fraction(0), Fraction(4), (Fraction(2), Fraction(4)), 2),
            (Fraction(25, 2), Fraction(4), (Fraction(25, 2), Fraction(14)), Fraction(3, 2)),
        ]
        for index, (start, duration, expected_range, expected_duration) in enumerate(cases):
            with self.subTest(start=start):
                plan = build_plan(media, sample=SampleRequest(start, duration),
                                  output=self.root / f"sample-{index}.wav")
                result = execution.execute(plan, backend=identity)
                self.assertEqual(result.sample_range[0], expected_range[0])
                self.assertLessEqual(abs(result.sample_range[1] - expected_range[1]),
                                     Fraction(1, 1000))
                self.assertLessEqual(abs(result.frames - expected_duration * SAMPLE_RATE), 45)
                self.assertEqual(Fraction(result.frames, SAMPLE_RATE),
                                 result.sample_range[1] - result.sample_range[0])
        empty = build_plan(media, sample=SampleRequest(Fraction(30), Fraction(2)),
                           output=self.root / "empty.wav")
        with self.assertRaisesRegex(RepairError, "no audio"):
            execution.execute(empty, backend=identity)
        self.assertFalse(empty.output.exists())

    def test_default_sample_seeks_to_late_first_audio_and_rejects_bad_estimate(self):
        container, _ = self.delayed_media(delay=8)
        media = probe(container)
        media = replace(media, streams=tuple(
            replace(item, start=Fraction(8)) if item.kind == "audio" else item
            for item in media.streams
        ))
        plan = build_plan(media, sample=SampleRequest(None, Fraction(3)),
                          output=self.root / "late-default.wav")
        commands = []
        original_tool = execution._tool

        def recording_tool(args):
            commands.append(args)
            return original_tool(args)

        with mock.patch("audio_repair.execution._tool", side_effect=recording_tool):
            result = execution.execute(plan, backend=identity)
        self.assertIn("-seek_timestamp", commands[0])
        self.assertEqual(result.sample_range, (Fraction(8), Fraction(11)))
        wrong = replace(plan, sample=SampleRequest(Fraction(10), Fraction(3)),
                        output=self.root / "wrong-default.wav")
        with self.assertRaisesRegex(RepairError, "first audio sample could not be determined"):
            execution.execute(wrong, backend=identity)
        self.assertFalse(wrong.output.exists())

    def test_sample_timestamp_continuity_rejected(self):
        lines = (
            "[Parsed_ashowinfo] n:0 pts:0 pts_time:0 nb_samples:100\n"
            "[Parsed_ashowinfo] n:1 pts:200 pts_time:0.004 nb_samples:100\n"
        )
        with self.assertRaisesRegex(RepairError, "discontinuous"):
            execution._showinfo(lines, Fraction(1, SAMPLE_RATE))
        with self.assertRaisesRegex(RepairError, "missing"):
            execution._showinfo("[Parsed_ashowinfo] n:0 pts:NOPTS nb_samples:100",
                                Fraction(1, SAMPLE_RATE))
        drift = "\n".join(
            f"[Parsed_ashowinfo] n:{i} pts:{pts} nb_samples:100"
            for i, pts in enumerate((0, 101, 202))
        )
        with self.assertRaisesRegex(RepairError, "discontinuous"):
            execution._showinfo(drift, Fraction(1, SAMPLE_RATE))
        with self.assertRaisesRegex(RepairError, "empty frame"):
            execution._showinfo("[Parsed_ashowinfo] n:0 pts:0 nb_samples:0",
                                Fraction(1, SAMPLE_RATE))

    def test_cancellation_and_encoding_failure_leave_no_output(self):
        path = self.source()
        digest = hashlib.sha256(path.read_bytes()).digest()
        plan = build_plan(probe(path))

        def interrupted(source, destination, device):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            execution.execute(plan, backend=interrupted)
        self.assertFalse(plan.output.exists())
        self.assertEqual(list(self.root.iterdir()), [path])
        self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), digest)

        with mock.patch("audio_repair.execution._encode", side_effect=OSError("encoder failed")):
            with self.assertRaisesRegex(RepairError, "encoder failed"):
                execution.execute(plan, backend=identity)
        self.assertFalse(plan.output.exists())
        self.assertEqual(list(self.root.iterdir()), [path])
        self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), digest)

    def test_video_remux_failure_does_not_publish(self):
        container, _ = self.delayed_media()
        plan = build_plan(probe(container))
        with mock.patch("audio_repair.video.remux", side_effect=RepairError("remux failed")):
            with self.assertRaises(RepairError):
                execution.execute(plan, backend=identity)
        self.assertFalse(plan.output.exists())

    def test_two_publishers_never_replace(self):
        first, second = self.root / "first", self.root / "second"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        output = self.root / "out"
        barrier = threading.Barrier(2)

        def publish(path):
            barrier.wait()
            try:
                execution.publish(path, output)
                return "success"
            except RepairError as exc:
                return str(exc)

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(publish, (first, second)))
        self.assertEqual(results.count("success"), 1)
        self.assertTrue(any("refusing to overwrite" in result for result in results))
        self.assertIn(output.read_bytes(), (b"first", b"second"))
        self.assertEqual(first.read_bytes(), b"first")
        self.assertEqual(second.read_bytes(), b"second")


class CheckpointTests(unittest.TestCase):
    def test_checksum_failure_before_upstream_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint"
            path.write_bytes(b"bad")
            inference = mock.Mock()
            with mock.patch("huggingface_hub.hf_hub_download", return_value=str(path)) as download, \
                 mock.patch.dict("sys.modules", {"inference": inference}):
                with self.assertRaisesRegex(RepairError, "checksum mismatch"):
                    backend.repair(Path("in.wav"), Path("out.wav"), "cpu")
            inference.run_inference.assert_not_called()
            self.assertTrue(download.call_args.kwargs["local_files_only"])
            self.assertFalse(download.call_args.kwargs["token"])
            self.assertEqual(download.call_args.kwargs["revision"], backend.CHECKPOINT_REVISION)

    def test_cache_miss_fallback_and_digest_each_time(self):
        from huggingface_hub.errors import LocalEntryNotFoundError

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint"
            path.write_bytes(b"trusted")
            expected = hashlib.sha256(b"trusted").hexdigest()
            with mock.patch.object(backend, "CHECKPOINT_SHA256", expected), \
                 mock.patch("huggingface_hub.hf_hub_download",
                            side_effect=[LocalEntryNotFoundError("missing"), str(path), str(path)]) as download:
                self.assertEqual(backend.verified_checkpoint(), path)
                self.assertTrue(download.call_args_list[0].kwargs["local_files_only"])
                self.assertNotIn("local_files_only", download.call_args_list[1].kwargs)
                path.write_bytes(b"changed")
                with self.assertRaisesRegex(RepairError, "checksum mismatch"):
                    backend.verified_checkpoint()

    def test_backend_wraps_device_error_and_suppresses_upstream_stdout(self):
        def fail(*args, **kwargs):
            print("upstream bands")
            raise RuntimeError("MPS was requested but is not available")

        inference = mock.Mock(run_inference=fail)
        with mock.patch.object(backend, "verified_checkpoint", return_value=Path("checkpoint")), \
             mock.patch.dict("sys.modules", {"inference": inference}), \
             mock.patch("sys.stdout") as stdout:
            with self.assertRaisesRegex(RepairError, "MPS was requested but is not available"):
                backend.repair(Path("in.wav"), Path("out.wav"), "mps")
        stdout.write.assert_not_called()

    def test_verbose_shows_upstream_diagnostics(self):
        warning = "An output with one or more elements was resized since it had shape []"

        def run(*args, **kwargs):
            print("upstream bands")
            warnings.warn_explicit(warning, UserWarning, "apollo.py", 1,
                                   module="look2hear.models.apollo")
            warnings.warn("unrelated warning", UserWarning)
            return "mps"

        inference = mock.Mock(run_inference=run)
        for verbose in (False, True):
            with self.subTest(verbose=verbose), \
                 mock.patch.object(backend, "verified_checkpoint", return_value=Path("checkpoint")), \
                 mock.patch.dict("sys.modules", {"inference": inference}), \
                 contextlib.redirect_stderr(io.StringIO()) as stderr, warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                backend.repair(Path("in.wav"), Path("out.wav"), "mps", verbose=verbose)
            self.assertEqual("upstream bands" in stderr.getvalue(), verbose)
            messages = [str(item.message) for item in seen]
            self.assertIn("unrelated warning", messages)
            self.assertEqual(warning in messages, verbose)


if __name__ == "__main__":
    unittest.main()
