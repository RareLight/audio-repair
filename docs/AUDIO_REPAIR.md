# Audio Repair

This wrapper uses Apollo to repair a selected audio track from a local file.
It keeps the input unchanged and uses lossless delivery formats; it cannot
recover a guaranteed original signal from a lossy recording. Audition a
level-matched preview and chunk seams before relying on the result. Human
listening validation is still pending.

## Install on Apple Silicon

Run the setup from the repository root. Use native Python 3.10 or later and
an isolated environment. FFmpeg and
FFprobe must be on `PATH`; if needed, install them with Homebrew:

```bash
brew install ffmpeg
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/macos-arm64.txt
python -m pip install --no-deps -e .
```

The macOS dependencies are pinned in [requirements/macos-arm64.txt](../requirements/macos-arm64.txt). The
wrapper defaults to `--device auto` (CUDA, then MPS, then CPU); use
`--device mps`, `--device cpu`, or `--device cuda` to request a specific
device. Explicit unavailable devices fail instead of silently falling back.

## Inspect, preview, repair

Inspect the selected stream and destination without downloading weights or
running inference. Add `--verbose` for resource estimates and technical
details:

```bash
audio-repair concert.mp4 --dry-run
```

During a repair the terminal shows each phase and an inference progress bar
based on completed chunks. The bar reaches 100% when inference finishes.
Encoding, video assembly, and verification follow as separate phases.
Redirected output reports progress milestones on separate lines. Use
`-v` or `--verbose` to see device, stream and frame details, upstream
diagnostics, and the exact gain.

Run real inference on a short, audio-only passage before processing the file:

```bash
audio-repair concert.mp4 --sample --start 00:12:30 --duration 30
```

Sample start is relative to the source playback timeline. The default is 30
seconds from the first presented audio sample; duration must be positive and
at most 120 seconds. The actual available range is reported after decoding.
Time values can be decimal seconds or `HH:MM:SS[.fraction]`. Samples use the
same 6-second chunks, 1-second overlap, 1-second neighboring context per side
and batch size 1 as a full repair. A sample may differ from the corresponding
full-run slice because chunk boundaries differ.

To preview the repaired audio with the corresponding picture:

```bash
audio-repair concert.mp4 --sample-video --start 00:12:30 --duration 30
```

`--sample-video` uses the same time options as `--sample` and writes an MKV
with the first video stream copied and the selected audio repaired as 24-bit
FLAC. It omits alternate streams and chapters. Stream copy starts at the
first usable keyframe inside the requested interval and can end early to
keep complete frames. Both tracks use the resulting interval, which the
command reports on the source playback timeline. If the interval has no
usable keyframe, choose a different start or a longer duration. Use
`--dry-run` to inspect the planned streams and output path.

Repair a video while preserving compatible streams:

```bash
audio-repair concert.mp4
```

Or intentionally omit video and all other streams and write float WAV:

```bash
audio-repair concert.mp4 --audio-only -o concert.repaired.wav
```

### Queue multiple files

```bash
audio-repair *.mp4
audio-repair take-1.wav take-2.wav
audio-repair .
audio-repair ~/Recordings ~/Downloads/sets
```

Each directory contributes eligible files from its top level in filename
order. Directory scans look for audio tracks and skip standard
`*.repaired.*` outputs. Paths supplied explicitly are checked as inputs.
Options apply to every queued file. Jobs run one at a time and continue
after a file fails. The command reports the
totals and exits with a failure status if any job failed. Use `--dry-run`
to inspect the queue before starting.

The default audio output is `<stem>.repaired.flac`; video defaults to
`<stem>.repaired.mkv`. Each output goes beside its source file. Use `-o`
to set an output path when the queue contains a single file. Sample names use
`<stem>.sample-<start_ms>-<duration_ms>.repaired.flac` or
`<stem>.sample-<start_ms>-<duration_ms>.repaired.mkv` for a video preview.
The sample filename uses the requested times. For audio-only output, `-o`
accepts `.flac` or `.wav`. Video outputs require `.mkv`. Existing outputs,
source aliases, and dangling symlink
destinations are refused. There is no force/overwrite option.

If multiple audio streams exist, the lowest-index default audio stream is
selected; if none is marked default, the lowest audio index is selected.
Use the absolute index shown by `--dry-run` with `--audio-stream-index N` to
choose another. Only mono or stereo selected audio is supported; alternatives
may be multichannel when copied in a video.

## Formats, timing and limits

Apollo operates at 44.1 kHz; selected audio at another rate is resampled once.
Audio-only delivery is 24-bit FLAC or 32-bit float WAV. Video delivery is
Matroska with a 24-bit FLAC replacement track; it does not produce AAC, MP3,
MP4 or MOV output. FLAC is lossless relative to the quantized 24-bit signal,
not necessarily to the model's floats. Matroska/FLAC support varies across
stock players and editing applications; keep the source for compatibility.
Full audio-to-FLAC repairs copy supported JPEG and PNG cover images. Samples,
float WAV, and explicit `--audio-only` omit cover art.

Float WAV preserves values outside the integer PCM range. If repaired
samples exceed the FLAC range, the default applies one constant gain across
all samples and channels and reports the reduction in dB. A repair that
needs more than 6 dB of attenuation fails and reports the required reduction.
This guardrail limits the whole-track level change to roughly a halving of
amplitude, about one bit of quantization margin at 24-bit output.
`--no-attenuation` requires FLAC output at the repaired level. The existing
`--allow-attenuation` flag explicitly selects the default and uses the same
limit.

The wrapper attempts to copy supported video, alternate audio, subtitles,
attachments, chapters and display metadata while checking their timelines.
It refuses publication if it cannot verify preservation. For example,
`mov_text` captions, data streams, unsupported rotations or display metadata,
discontinuous timestamps and complex edits may prevent a preservation-mode
job. The repaired primary AAC track can be trimmed when decoded timing and
track metadata agree on the presented end. Other AAC timing layouts may
still be refused. `--audio-only` omits the other streams and chapters.

Video repair and video previews require every source audio track to
start at the first presented frame of the first video stream. An audio
track that starts earlier or later is rejected before inference. The output
must pass the same check before publication, along with the existing checks
for drift and end timing. Audio-only exports and `--sample` previews can
use tracks with a start offset.

Full jobs are limited by memory, disk space, and the intermediate WAV format.
RAM and scratch/destination space are estimated and checked dynamically,
including after decode. Intermediate files and the staged output use the
destination volume. Apollo uses full-length CPU buffers, and chunking
bounds model batch size. Classic WAV intermediates are guarded against the
4 GiB limit. Use a shorter sample or split the source when a limit is reached.

On an Apple Silicon Mac with 32 GiB of unified memory, a two-hour stereo
48 kHz fixture was resampled to 44.1 kHz and exported as 24-bit FLAC with
317,520,000 frames per channel. The process reached 6.84 GiB peak RSS
with Apollo's production chunk loop and an identity model. The full Apollo
model was verified separately on a four-second stereo preview on CPU.
The longest documented run with the real checkpoint on MPS was two minutes.

## Checkpoint and maintenance

Only local media paths are accepted; media is not uploaded. Execution first
tries the local Hugging Face cache, then downloads the official checkpoint if
needed and network access is allowed. The wrapper pins and verifies the file
on every run before model deserialization:

| Field | Value |
| --- | --- |
| Repository | `JusperLee/Apollo` |
| Revision | `c68bd80fdd9c0d93d2f4a833cb154624f660a561` |
| File | `pytorch_model.bin` |
| SHA-256 | `99d9af7f1ff20e63c393035513a655392818d66b4d7fc23d658175c1f15e8d76` |

`HF_HUB_OFFLINE=1` requires that exact revision in the cache. To update the
checkpoint, independently verify the trusted official snapshot and digest,
update both constants in [audio_repair/backend.py](../audio_repair/backend.py), and rerun the checksum,
offline-cache and failure tests, the full unit suite and real-device smoke
tests. Do not accept an unexpected digest or deserialize untrusted weights.

For an upstream merge, preserve the original Apollo source, attribution and
license. Review changes since the wrapper baseline (`e84bcac`), especially
`inference.run_inference`, its device/chunk behavior and dependencies; adapt
the wrapper boundary rather than forking the model algorithm. Run
`python -m unittest discover -s tests -v`, test a built installation outside
the checkout, repeat the real-device smoke tests, and audition representative
previews and seams before release.

See the [implementation brief](IMPLEMENTATION_BRIEF.md) for the pipeline and
release checklist and the [Apple Silicon notes](MACOS_ARM64.md) for direct
Apollo inference and accelerator benchmarks.
