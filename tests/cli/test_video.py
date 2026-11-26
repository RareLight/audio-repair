import hashlib
import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from audio_repair import execution, timeline, video
from audio_repair.media import RepairError, probe
from audio_repair.planner import SAMPLE_RATE, SampleRequest, build_plan


def identity(source, destination, device):
    samples, rate = sf.read(str(source), dtype="float32", always_2d=True)
    sf.write(str(destination), samples, rate, subtype="FLOAT")
    return "identity"


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode:
        raise AssertionError(f"Command failed: {args}\n{result.stderr}")
    return result.stdout


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patch = mock.patch("audio_repair.execution._physical_memory", return_value=64 * 2**30)
        patch.start()
        self.addCleanup(patch.stop)

    def mov(self, *, primary_offset="0", alternate_offset="0", video_offset="0"):
        output = self.root / "concert.mov"
        tone = (
            "if(lt(t,0.2),0.2*sin(2*PI*440*t),"
            "if(between(t,2,2.2),0.2*sin(2*PI*880*t),0))"
        )
        command([
            "ffmpeg", "-nostdin", "-v", "error", "-y",
            "-itsoffset", video_offset, "-f", "lavfi", "-i", "testsrc2=s=48x48:r=8:d=4",
            "-itsoffset", alternate_offset, "-f", "lavfi", "-i",
            "sine=frequency=300:sample_rate=48000:duration=4",
            "-itsoffset", primary_offset, "-f", "lavfi", "-i",
            f"aevalsrc='{tone}|{tone}':s=48000:d=3",
            "-map", "0:v", "-map", "1:a", "-map", "2:a",
            "-c:v", "mpeg4", "-q:v", "3", "-c:a", "pcm_s16le",
            "-disposition:a:0", "0", "-disposition:a:1", "default",
            "-metadata:s:a:0", "language=eng", "-metadata:s:a:1", "language=fra",
            str(output),
        ])
        return output

    def vfr(self):
        output = self.root / "vfr.mov"
        command([
            "ffmpeg", "-nostdin", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=s=48x48:r=12:d=4",
            "-f", "lavfi", "-i", "sine=frequency=420:sample_rate=48000:duration=2",
            "-map", "0:v", "-map", "1:a",
            "-vf", "select=not(mod(n\\,2))+not(mod(n\\,3))",
            "-fps_mode", "vfr", "-c:v", "mpeg4", "-q:v", "3",
            "-c:a", "pcm_s16le", str(output),
        ])
        return output

    def mkv_with_chapters(self):
        subtitle = self.root / "captions.srt"
        subtitle.write_text(
            "1\n00:00:00,400 --> 00:00:00,900\nFirst\n\n"
            "2\n00:00:01,400 --> 00:00:02,100\nSecond\n",
            encoding="utf-8",
        )
        metadata = self.root / "chapters.txt"
        metadata.write_text(
            ";FFMETADATA1\ntitle=Concert\nREPLAYGAIN_TRACK_GAIN=-4 dB\n"
            "[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=1200\n"
            "title=Intro\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=1200\nEND=3000\n"
            "title=Finale\n",
            encoding="utf-8",
        )
        output = self.root / "chapters.mkv"
        command([
            "ffmpeg", "-nostdin", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=s=48x48:r=8:d=3",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=3",
            "-i", str(subtitle), "-i", str(metadata),
            "-map", "0:v", "-map", "1:a", "-map", "2:s",
            "-map_metadata", "3", "-map_chapters", "3",
            "-c:v", "mpeg4", "-c:a", "pcm_s16le", "-c:s", "srt",
            "-metadata:s:s:0", "language=eng", "-metadata:s:s:0", "title=Captions",
            "-disposition:s:0", "default", str(output),
        ])
        return output, subtitle

    def hashes(self, path, index):
        raw = command([
            "ffprobe", "-v", "error", "-select_streams", str(index),
            "-show_packets", "-show_data_hash", "sha256",
            "-show_entries", "packet=data_hash", "-of", "json", str(path),
        ])
        return [packet["data_hash"] for packet in json.loads(raw)["packets"]]

    def video_packets(self, path, index):
        raw = command([
            "ffprobe", "-v", "error", "-select_streams", str(index),
            "-show_packets", "-show_entries", "packet=pts_time,duration_time",
            "-of", "json", str(path),
        ])
        return json.loads(raw)["packets"]

    def test_aligned_default_with_alternate_audio_and_lossless_copy(self):
        path = self.mov()
        digest = hashlib.sha256(path.read_bytes()).digest()
        media = probe(path)
        plan = build_plan(media)
        self.assertEqual(plan.primary.index, 2)
        self.assertEqual(plan.primary.sample_rate, 48000)
        span = timeline.audio_span(media, plan.primary)
        self.assertEqual(span.start, 0)
        self.assertEqual(span.frames, 144000)
        result = execution.execute(plan, backend=identity)
        self.assertEqual(result.frames, 132300)
        output = probe(result.output)
        self.assertEqual([s.kind for s in output.streams], ["video", "audio", "audio"])
        self.assertEqual(output.streams[2].codec, "flac")
        self.assertEqual(output.streams[2].bits_per_raw_sample, 24)
        self.assertIn("default", output.streams[2].dispositions)
        self.assertEqual(output.streams[2].tags.get("language"), "fra")
        repaired = timeline.audio_span(output, output.streams[2])
        self.assertEqual(repaired.frames, 132300)
        self.assertEqual(repaired.start, span.start)
        self.assertEqual(self.hashes(path, 0), self.hashes(result.output, 0))
        self.assertEqual(self.hashes(path, 1), self.hashes(result.output, 1))
        self.assertEqual(len(self.video_packets(path, 0)), len(self.video_packets(result.output, 0)))
        decoded = subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(result.output),
            "-map", "0:2", "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1",
        ], capture_output=True, check=True).stdout
        samples = np.frombuffer(decoded, dtype="<f4").reshape(-1, 2)[:, 0]
        for start, frequency in ((0.02, 440), (2.02, 880)):
            marker = samples[int(start * SAMPLE_RATE):int((start + 0.16) * SAMPLE_RATE)]
            spectrum = abs(np.fft.rfft(marker))
            peak = np.fft.rfftfreq(len(marker), 1 / SAMPLE_RATE)[spectrum.argmax()]
            self.assertAlmostEqual(peak, frequency, delta=4)
        self.assertLess(np.max(np.abs(samples[SAMPLE_RATE:3 * SAMPLE_RATE // 2])), 1e-6)
        self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), digest)
        self.assertEqual(sorted(item.name for item in self.root.iterdir()),
                         ["concert.mov", "concert.repaired.mkv"])

    def test_audio_start_offsets_fail_before_inference(self):
        for track, index in (("primary", 2), ("alternate", 1), ("video", 1)):
            with self.subTest(track=track):
                path = self.mov(**{f"{track}_offset": "0.253"})
                before = hashlib.sha256(path.read_bytes()).digest()
                plan = build_plan(probe(path))
                called = mock.Mock()
                with self.assertRaisesRegex(RepairError, f"Audio stream {index} starts .*--audio-only"):
                    execution.execute(plan, backend=called)
                called.assert_not_called()
                self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), before)
                self.assertFalse(plan.output.exists())
                self.assertFalse(any(self.root.glob(".audio-repair-*")))

    def test_remux_cannot_publish_a_one_tick_audio_delay(self):
        plan = build_plan(probe(self.mov()))
        run_tool = video.run_tool

        def introduce_delay(args):
            args = args.copy()
            args[args.index("-itsoffset") + 1] = "0.001"
            return run_tool(args)

        with mock.patch.object(video, "run_tool", side_effect=introduce_delay):
            with self.assertRaisesRegex(RepairError, "requires aligned starts"):
                execution.execute(plan, backend=identity)
        self.assertFalse(plan.output.exists())

    def test_video_sample_copies_complete_b_frame_sequence_and_matches_audio(self):
        source = self.root / "b-frames.mp4"
        command([
            "ffmpeg", "-nostdin", "-v", "error",
            "-f", "lavfi", "-i", "testsrc2=s=96x64:r=10:d=8",
            "-f", "lavfi", "-i", "sine=sample_rate=48000:duration=8",
            "-c:v", "libx264", "-g", "20", "-sc_threshold", "0", "-bf", "3",
            "-c:a", "aac", str(source),
        ])
        media = probe(source)
        start, duration = Fraction("2.2"), Fraction("3.5")
        plan = build_plan(media, sample=SampleRequest(start, duration), sample_video=True)
        result = execution.execute(plan, backend=identity)
        clip_start, clip_end = result.sample_range
        self.assertEqual(clip_start, 4)
        self.assertLessEqual(clip_end, start + duration)
        self.assertGreaterEqual(clip_end, Fraction("5.4"))
        output = probe(result.output)
        self.assertEqual([stream.codec for stream in output.streams], ["h264", "flac"])
        self.assertEqual(result.frames, (clip_end - clip_start) * SAMPLE_RATE)
        self.assertEqual(timeline.audio_span(output, output.streams[1]).start, 0)
        source_packets = self.video_packets(source, 0)
        source_hashes = self.hashes(source, 0)
        covered = [(Fraction(packet["pts_time"]), digest)
                   for packet, digest in zip(source_packets, source_hashes)
                   if clip_start <= Fraction(packet["pts_time"]) < clip_end]
        self.assertEqual(self.hashes(result.output, 0), [digest for _, digest in covered])
        self.assertEqual(
            [Fraction(packet["pts_time"]) for packet in self.video_packets(result.output, 0)],
            [pts - clip_start for pts, _ in covered],
        )

        no_keyframe = build_plan(media, sample=SampleRequest(start, Fraction(1)), sample_video=True)
        called = mock.Mock()
        with self.assertRaisesRegex(RepairError, "keyframe inside the sample range"):
            execution.execute(no_keyframe, backend=called)
        called.assert_not_called()
        self.assertFalse(no_keyframe.output.exists())

    def test_video_sample_scope_and_delayed_audio(self):
        source, _ = self.mkv_with_chapters()
        request = SampleRequest(None, Fraction(1))
        plan = build_plan(probe(source), sample=request, sample_video=True)
        result = execution.execute(plan, backend=identity)
        self.assertEqual(result.sample_range, (0, 1))
        output = probe(result.output)
        self.assertEqual([stream.kind for stream in output.streams], ["video", "audio"])
        self.assertFalse(output.chapters)

        delayed = self.mov(primary_offset="0.253")
        plan = build_plan(probe(delayed), sample=request, sample_video=True)
        called = mock.Mock()
        with self.assertRaisesRegex(RepairError, "requires aligned starts.*--sample"):
            execution.execute(plan, backend=called)
        called.assert_not_called()
        self.assertFalse(plan.output.exists())

    def test_vfr_and_early_audio_end_preserved(self):
        path = self.vfr()
        before = probe(path)
        packets = self.video_packets(path, 0)
        times = [Fraction(packet["pts_time"]) for packet in packets]
        self.assertGreater(len(set(b - a for a, b in zip(times, times[1:]))), 1)
        result = execution.execute(build_plan(before), backend=identity)
        after = probe(result.output)
        self.assertEqual(result.frames, 2 * SAMPLE_RATE)
        self.assertEqual(self.hashes(path, 0), self.hashes(result.output, 0))
        source_pts = [Fraction(packet["pts_time"]) for packet in packets]
        output_pts = [Fraction(packet["pts_time"]) for packet in self.video_packets(result.output, 0)]
        self.assertEqual(len(source_pts), len(output_pts))
        self.assertTrue(all(abs(a - b) <= Fraction(2, 1000)
                            for a, b in zip(source_pts, output_pts)))
        self.assertGreater(max(output_pts), Fraction(3))
        span = timeline.audio_span(after, next(s for s in after.streams if s.kind == "audio"))
        self.assertEqual(span.frames, 2 * SAMPLE_RATE)

    def test_subtitles_chapters_and_global_metadata(self):
        path, _ = self.mkv_with_chapters()
        before = probe(path)
        result = execution.execute(build_plan(before), backend=identity)
        after = probe(result.output)
        self.assertEqual(next(value for key, value in after.tags.items() if key.lower() == "title"),
                         "Concert")
        self.assertTrue(any("replaygain" in key.lower() for key in before.tags))
        self.assertFalse(any("replaygain" in key.lower() for key in after.tags))
        self.assertEqual(len(after.chapters), 2)
        self.assertEqual([chapter["tags"]["title"] for chapter in after.chapters],
                         ["Intro", "Finale"])
        subtitle = next(stream for stream in after.streams if stream.kind == "subtitle")
        self.assertEqual(subtitle.codec, "subrip")
        self.assertEqual(subtitle.tags["title"], "Captions")
        self.assertEqual(subtitle.tags["language"], "eng")
        self.assertIn("default", subtitle.dispositions)
        original_subtitle = next(stream for stream in before.streams if stream.kind == "subtitle")
        self.assertEqual(self.hashes(path, original_subtitle.index),
                         self.hashes(result.output, subtitle.index))

    def test_mov_text_and_data_rejected_before_execution(self):
        path, subtitle = self.mkv_with_chapters()
        source = self.root / "subtitle.mp4"
        command([
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(path),
            "-i", str(subtitle), "-map", "0:v", "-map", "0:a", "-map", "1:s",
            "-c:v", "copy", "-c:a", "aac", "-c:s", "mov_text", str(source),
        ])
        media = probe(source)
        with self.assertRaisesRegex(RepairError, "mov_text.*--audio-only"):
            build_plan(media)
        self.assertFalse(build_plan(media, audio_only=True).video)
        data = replace(media.streams[-1], index=9, kind="data", codec="bin_data")
        clean = probe(self.mov())
        with self.assertRaisesRegex(RepairError, "data/bin_data.*--audio-only"):
            build_plan(replace(clean, streams=clean.streams + (data,)))

    def test_discontinuous_frame_timestamps_are_rejected(self):
        media = probe(self.mov())
        audio = next(stream for stream in media.streams if stream.index == 2)
        records = [
            {"best_effort_timestamp": str(12144 + offset), "nb_samples": "480"}
            for offset in (0, 480, 961, 1442)
        ]
        with mock.patch("audio_repair.timeline._records",
                        return_value=(record for record in records)):
            with self.assertRaisesRegex(RepairError, "discontinuous or drifting"):
                timeline.audio_span(media, audio)

    def test_video_end_is_verified_in_presentation_order(self):
        media = probe(self.mov())
        stream = replace(media.streams[0], time_base=Fraction(1, 1000))
        source_packets = [
            {"pts": "0", "duration": "100"},
            {"pts": "200", "duration": "100"},
            {"pts": "100", "duration": "100"},
        ]
        output_packets = [dict(packet) for packet in source_packets]
        output_packets[1]["duration"] = "50"
        with mock.patch("audio_repair.timeline._records", side_effect=[
            (packet for packet in source_packets), (packet for packet in output_packets),
        ]):
            with self.assertRaisesRegex(RepairError, "changed its end timing"):
                timeline.verify_copy_timing(media, stream, media, stream, Fraction(0))

    def test_real_discontinuity_and_verification_failure_leave_no_output(self):
        path = self.root / "gap.mkv"
        command([
            "ffmpeg", "-nostdin", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=s=48x48:r=8:d=3",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
            "-map", "0:v", "-map", "1:a", "-c:v", "mpeg4",
            "-af", "asetpts=PTS+if(gte(T\\,1)\\,0.5/TB\\,0)",
            "-c:a", "pcm_s16le", str(path),
        ])
        plan = build_plan(probe(path))
        called = mock.Mock()
        with self.assertRaisesRegex(RepairError, "discontinuous or drifting"):
            execution.execute(plan, backend=called)
        called.assert_not_called()
        self.assertFalse(plan.output.exists())
        clean = self.mov()
        plan = build_plan(probe(clean))
        with mock.patch("audio_repair.video._verify", side_effect=RepairError("injected mismatch")):
            with self.assertRaisesRegex(RepairError, "injected mismatch"):
                execution.execute(plan, backend=identity)
        self.assertFalse(plan.output.exists())
        self.assertFalse(any(item.name.startswith(".audio-repair-") for item in self.root.iterdir()))

    def test_aac_primary_accepts_aligned_audio_and_rejects_delay(self):
        for offset in ("0", "0.261"):
            with self.subTest(offset=offset):
                self.check_aac_primary(offset)

    def check_aac_primary(self, offset):
        path = self.root / f"aac-{offset}.mp4"
        command([
            "ffmpeg", "-nostdin", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=s=48x48:r=10:d=2",
            "-itsoffset", offset, "-f", "lavfi", "-i",
            "sine=frequency=440:sample_rate=48000:duration=1.37",
            "-map", "0:v", "-map", "1:a", "-c:v", "mpeg4", "-c:a", "aac", str(path),
        ])
        plan = build_plan(probe(path))
        if offset == "0":
            result = execution.execute(plan, backend=identity)
            self.assertLessEqual(abs(result.frames - plan.primary.duration * SAMPLE_RATE), 1)
            output = probe(result.output)
            span = timeline.audio_span(output, output.streams[1])
            self.assertEqual(span.frames, result.frames)
            self.assertEqual(span.start, 0)
            self.assertEqual(self.hashes(path, 0), self.hashes(result.output, 0))
        else:
            called = mock.Mock()
            with self.assertRaisesRegex(RepairError, "requires aligned starts"):
                execution.execute(plan, backend=called)
            called.assert_not_called()
            self.assertFalse(plan.output.exists())
            audio_only = build_plan(plan.media, audio_only=True)
            self.assertTrue(execution.execute(audio_only, backend=identity).output.exists())
        preview = build_plan(plan.media, sample=SampleRequest(None, Fraction(2)),
                             output=self.root / f"aac-{offset}-sample.wav")
        sample = execution.execute(preview, backend=identity)
        end = plan.primary.start + plan.primary.duration - plan.media.start
        self.assertLessEqual(abs(sample.sample_range[1] - end), Fraction(1, SAMPLE_RATE))
        self.assertLessEqual(abs(sample.frames - plan.primary.duration * SAMPLE_RATE), 1)

    def test_aac_tail_requires_last_frame_duration_and_declared_end_to_agree(self):
        media = probe(self.mov())
        audio = replace(media.streams[2], codec="aac", start=Fraction(0),
                        duration=Fraction(1248, 48000))
        records = [
            {"best_effort_timestamp": "0", "nb_samples": "1024"},
            {"best_effort_timestamp": "1024", "nb_samples": "1024", "pkt_duration": "224"},
        ]
        with mock.patch("audio_repair.timeline._records",
                        side_effect=lambda *args: (record for record in records)):
            span = timeline.audio_span(media, audio, trim_aac_tail=True)
            self.assertEqual((span.frames, span.discard_tail), (1248, 800))
            with self.assertRaisesRegex(RepairError, "unverifiable"):
                timeline.audio_span(media, audio)
            records[-1]["pkt_duration"] = "225"
            with self.assertRaisesRegex(RepairError, "unverifiable"):
                timeline.audio_span(media, audio, trim_aac_tail=True)

    def test_stream_metadata_display_and_offset_gates(self):
        path = self.mov()
        media = probe(path)
        source = media.streams[0]
        with self.assertRaisesRegex(RepairError, "display metadata"):
            video._verify_stream_metadata(replace(source, display=(("color_range", "tv"),)),
                                          replace(source, display=(("color_range", "pc"),)), False)
        self.assertEqual(video._offset(Fraction(-1, 4)), "-0.250000")
        self.assertEqual(video._offset(Fraction(253, 1000)), "0.253000")
        with self.assertRaisesRegex(RepairError, "Global title"):
            video._verify_global_metadata(replace(media, tags={"TITLE": "original"}),
                                          replace(media, tags={"title": "changed"}))
        with self.assertRaisesRegex(RepairError, "stale global"):
            video._verify_global_metadata(media, replace(media, tags={"R128_TRACK_GAIN": "4"}))

    def test_video_error_suggests_audio_only_once(self):
        plan = mock.Mock()
        plan.media.streams = ()
        plan.media.tags = {}
        span = timeline.AudioSpan(Fraction(0), 100, SAMPLE_RATE)
        for message in ("mux failed", "mux failed; use --audio-only"):
            with self.subTest(message=message), \
                 mock.patch("audio_repair.video.run_tool", side_effect=RepairError(message)):
                with self.assertRaises(RepairError) as raised:
                    video.remux(plan, self.root / "in.flac", self.root / "out.mkv", span, 100)
                self.assertEqual(str(raised.exception).count("--audio-only"), 1)


if __name__ == "__main__":
    unittest.main()
