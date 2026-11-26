# Audio Repair: v1 implementation brief

**Status: v1 implemented and mechanically verified.** Apollo baseline: `e84bcac`. The remaining release gates are listening on representative recordings and long-duration resource calibration with the real checkpoint on MPS.

## Outcome and scope

Given a local recording, repair its primary audio track with Apollo without introducing another lossy codec generation. An audio input produces 24-bit FLAC by default. A video input produces Matroska (`.mkv`) with the repaired FLAC track in place of the original primary audio and the other compatible streams copied. Never overwrite the source or an existing destination.

The first supported platform is macOS on Apple Silicon; CPU should remain a functional fallback. Support mono and stereo, including a mono/stereo primary track alongside untouched multichannel alternatives. Support continuous, verifiable timelines. Target ordinary MP3, WAV, FLAC, AAC/M4A audio and MP4, MOV, MKV video first; support depends on codecs, track layout and the installed FFmpeg build, not extensions alone. A decodable recording is not necessarily one Apollo will improve. The short sample is how the user decides.

Not in v1: AAC/MP3 output, ALAC/MOV delivery, video transcoding, surround repair or implicit downmix, GUI, live streams/URLs, DRM, loudness mastering, or unlimited-duration streaming inference. Preserve the existing upstream CLI and tests.

## CLI contract

Examples of the proposed interface:

```text
audio-repair concert.mp4 --dry-run
audio-repair concert.mp4 --sample --start 00:12:30 --duration 30
audio-repair concert.mp4 --sample-video --start 00:12:30 --duration 30
audio-repair concert.mp4
audio-repair concert.mp4 --audio-only -o concert.flac
audio-repair audience.mp3 -o audience.repaired.wav
```

| Option | Contract |
| --- | --- |
| `INPUT...`, `-o PATH` | Accept files, wildcards and top-level directory scans. Run queued jobs sequentially and report failures per file. Default output is a sibling named `<stem>.repaired.flac` for audio or `<stem>.repaired.mkv` for video. `-o` changes the destination for a single file. Reject incompatible extensions with an actionable message. No `--force`. |
| `--dry-run` | Probe and print the selected stream, copy/omission plan, output format and destination. Add `--verbose` for resampling, chunk settings, resource estimates and deferred checks. The dry run creates no files and requires no inference or model download. Works with either sample option. |
| `--sample` | Run **real inference** on one short passage. Always produce **audio only**, even from video. Same stream choice, sample rate, chunk settings, and export policy as a full repair. Default: 30 seconds from the first presented audio sample. Default output: `<stem>.sample-<start_ms>-<duration_ms>.repaired.flac`, with the resolved requested start/duration in milliseconds. It cannot collide with a full repair's default name. |
| `--sample-video` | Use the same time options as `--sample` and deliver the first video stream with the selected repaired FLAC audio in an MKV. Omit alternate streams and chapters. Start at a usable keyframe inside the request and finish at a verified frame boundary. Report the actual common source interval or fail if no safe cut exists. Apply the source audio start-alignment check and verify the output before publishing. Incompatible with `--sample` and `--audio-only`. |
| `--start TIME`, `--duration TIME` | Sample mode only; accept nonnegative decimal seconds or `HH:MM:SS[.fraction]`. Duration must be positive; proposed sample limit: 120 seconds. Explicit starts use the source playback timeline, not the raw timestamp of the first audio packet. Intersect the request with available audio; report the actual range and fail if empty. |
| `--audio-only` | Extract and repair just the selected audio from a video, with no remux. This intentionally omits the other streams. Redundant but harmless with `--sample`. |
| `--audio-stream-index N` | Override the primary-track choice using the **absolute** input stream index shown by the dry-run. Reject a non-audio or missing index. |
| `--device auto\|mps\|cpu\|cuda` | Delegate to Apollo's selection policy. Report the actual device when running; do not silently use CPU when an explicit accelerator was requested. |
| `--allow-attenuation`, `--no-attenuation` | The integer-output default applies up to 6 dB of attenuation only when needed to fit the 24-bit PCM range. Report the reduction and fail above the cap. Apply the same gain across channels and time. `--allow-attenuation` selects the default explicitly. `--no-attenuation` requires the repaired level to fit as-is. |
| `-v`, `--verbose` | Include technical details and upstream diagnostics. Default output reports phases and inference progress. |

Audio-only destinations support `.flac` (24-bit) or `.wav` (float32). Video destinations support `.mkv` in v1: `-o out.mp4` must not silently select AAC or write Matroska under the wrong name. Preserve the original recording as the A/B reference. For the sample, print the exact source passage to compare at matched listening level. Read real neighboring audio for inference context, then trim it away; don't spend ten minutes repairing the prefix to sample minute ten. A preview is representative, not a promise of bit identity with a full-run slice at a different chunk boundary.

## Architecture and processing

The `audio-repair` entry point lives in [audio_repair/](../audio_repair/), with
its tests in [tests/cli/](../tests/cli/). The
[backend adapter](../audio_repair/backend.py) calls the
[Apollo inference entry point](../inference.py). The model and training
code keep their upstream paths for easier merges. Use one checkout for the
wrapper and backend. Preserve upstream attribution and the [license](../LICENSE),
document the tested checkpoint, and test the installed CLI from outside the
checkout. Tests generate media in temporary directories. Local recordings,
public demo downloads, and smoke-test output go in the ignored `media/`
directory. Use the OS credential store or tool-specific configuration
outside the checkout for credentials.

1. **Probe and plan.** Use the installed `ffprobe` and `ffmpeg` executables via argument arrays, not a shell. Read stream inventory, dispositions, channel layouts, rational time bases, codec/container information, and chapters. An attached cover image is not a movie. Choose the sole audio track; otherwise choose the lowest-index default audio track, falling back to the lowest-index audio track if no default is marked. Explain ambiguous choices and show the override. Fail early on no audio, unsupported primary layout, unreadable source, output conflicts, or known-incompatible copy streams.
2. **Decode only as necessary.** Eligible 44.1 kHz audio-only sources can use Apollo's native SoundFile path when it reads the intended audio and its decoded length/gapless handling is verified for that format. Otherwise explicitly select the stream with FFmpeg, decode once and resample once to 44.1 kHz float PCM. Do not detour through a lossy codec. A lossless temporary file is acceptable for the upstream file-based adapter. Keep source timing separately: a standalone WAV does not describe where its audio belonged in a movie.
3. **Repair.** Resolve and verify the official checkpoint against a release-pinned digest before handing its local path to Apollo's deserializer. Use the cache offline when available; never download on dry-run or upload user media. Use `run_inference` with 6 s chunks, 1 s overlap, 1 s of neighbor context per side, batch size 1. Preserve channels and frame count. Validate finite output, matching sample rate, channel count and sample count before encoding. Give a clear error for inputs too short for the model rather than exposing a tensor traceback.
4. **Encode once.** Keep repaired float PCM until final encoding. FLAC is lossless relative to the **quantized integer** signal, not necessarily the model's floats. Check for out-of-range values before integer conversion. Automatically apply a constant gain within the 6 dB cap, or stop and suggest float WAV if the required gain exceeds the allowed reduction. Do not clip silently. Preserve source tags or artwork where representable; do not copy stale ReplayGain/peak metadata onto the changed audio. Report material omissions; fail on unrepresentable streams in a preservation-mode video job.
5. **Remux video without transcoding.** Explicitly map the repaired track into the selected track's position. Copy all other supported streams, preserving order, language, titles, default/forced dispositions, chapters, attachments, and relevant display metadata. Do not select just the first video and audio and discard the rest. If preservation is not possible, explain the blocker and offer audio-only output instead.
6. **Validate, then publish.** Probe the staged result and compare it with the plan. Report the output path, selected stream, processing device, duration, and any gain change. Publish a complete, validated file using a same-filesystem, atomic **no-replace** operation. A preflight existence check followed by an overwriting rename is not sufficient. Errors, cancellation, and concurrent jobs must not overwrite a destination, alter input bytes, or leave a partial file at the final name. Clean up only job-owned scratch files.

## Timing and resource limits

- **Single clock:** Derive the selected track's audible start and end relative to the other streams from decoded presentation timing, accounting for edit lists and decoder priming. In video preservation mode every audio track must start at the same presentation timestamp as the first real video stream, in both the input and the staged output. Audio-only exports and previews can use tracks with an offset. Use rational timestamps and integer sample counts. Do not treat container duration, first packet PTS, or WAV time zero as the audible start. If muxing shifts the origin, use one common shift for all streams. Never use `-shortest`, time stretching, or per-track zeroing to conceal a discrepancy.
- **Verification:** Measure decoded frames through a single resampling pass; Apollo must return exactly that many frames. The FLAC track must decode to the same frame count. For video, compare original and output relative start/end timing, timeline continuity and stream inventory. Allow only documented sample-grid rounding and destination time-base quantization, not a loose lip-sync tolerance. Check for accumulating drift rather than only comparing the first timestamp. Flag or reject missing/discontinuous timestamps, nontrivial edits, or unsupported timing layouts rather than flattening them silently.
- **Bounded inference is not bounded system RAM:** Upstream reads the whole input and allocates full-length CPU buffers. At 44.1 kHz, stereo float PCM uses about 1.27 GB/hour **per copy**. Budget for input, accumulators, model headroom, intermediates and staged output; check scratch and destination volumes separately. Display estimates as estimates. Derive and document a conservative supported-duration/resource limit from target-machine measurements before release. Reject jobs that cannot safely fit. Do not pass an oversized intermediate or output to a classic RIFF/WAV writer; add a tested large-file path before claiming support above its limit.
- **Local operation:** Resolve input/output path and file-identity aliases, including hard links and symlinks. Check destination collisions again at publication. Emit concise phase progress and calculate the inference percentage from completed chunks. Missing FFmpeg, unavailable MPS, offline cache misses and insufficient space should each have a distinct, actionable diagnostic.

## Step-by-step implementation checklist

The checked milestones cover the implemented and mechanically tested behavior. Long-job calibration and the listening-dependent release gate remain open.

- [x] **0. Approval gate.** Agree on CLI, sample semantics, output defaults, and failure policies.
- [x] **1. Establish the wrapper boundary.** Add package/CLI scaffolding and an injectable repair adapter; preserve the original `inference.py` interface. Verify the installed command works outside the checkout and existing upstream tests still pass.
- [x] **2. Implement probe, plan, and dry-run before inference.** Test single and multiple audio tracks (including default second track, ties, and override), attached cover art, no audio, unsupported surround, bad extensions, and alias/collision checks. A dry-run with an empty weight cache succeeds without Torch inference, downloads, scratch files, or output.
- [x] **3. Ship one audio-only vertical slice.** Select the native or FFmpeg path, verify the official checkpoint, run chunked Apollo, stage one FLAC/float WAV and publish safely. With an identity backend, check lengths and exact round-trip of the chosen integer reference; inject over-range and nonfinite audio to verify failure/attenuation policies. With real Apollo, verify shape and finite output on the public MPS fixture.
- [x] **4. Implement the short sample.** Decode just the requested region plus context, execute the real adapter and trim by sample count. Test middle, first, last, shorter-than-requested and out-of-range intervals, including a nonzero video/audio offset. Verify it does not run inference over the full file, create a video, normalize for loudness, or collide with a full-run result.
- [x] **5. Implement preservation-mode video remux.** Use small synthetic fixtures with timestamped markers, an alternate/default audio track, chapters, and subtitles. Reject audio start offsets and test 48-to-44.1 kHz conversion, variable-frame-rate video, and intentional audio/video duration differences. Assert relative A/V sync near the start, middle, and end; verify copied content (packet hashes where comparable, otherwise decoded hashes plus presentation times), track order and metadata. Include a discontinuity and an unrepresentable stream that must fail cleanly.
- [ ] **6. Harden long jobs and failure recovery (partial).** Resource guards, the WAV-size boundary, interruption, encoder and verification failures, and no-replace publication are tested. Full jobs are bounded by resources and intermediate format capacity. A two-hour stereo run passed with Apollo's chunk loop and an identity model. The real-checkpoint MPS measurement remains two minutes.
- [ ] **7. Release gate.** The offline CPU/FFmpeg suite, upstream tests, wheel installation outside the checkout, and a real-checkpoint CPU preview pass. The earlier MPS smoke tests still need a follow-up on the current wrapper in a session with Metal access. Audition a level-matched preview and chunk seams on representative music. Install, sample, full repair, format limits, checkpoint updates and upstream merge checks are documented in [AUDIO_REPAIR.md](AUDIO_REPAIR.md).

**Definition of done:** An ordinary user can inspect a concert file with no inference, audition one chosen passage without processing the whole concert, and then create a validated lossless delivery file with one command. If the utility cannot preserve timing, streams, or output safety under the documented contract, it explains why and leaves the source and final destination untouched.
