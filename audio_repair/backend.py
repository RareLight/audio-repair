import contextlib
import hashlib
import io
import sys
import warnings
from pathlib import Path
from typing import Callable

from .media import RepairError
from .planner import CHUNK_BATCH_SIZE, CHUNK_SECONDS, CONTEXT_SECONDS, OVERLAP_SECONDS


CHECKPOINT_REPO = "JusperLee/Apollo"
CHECKPOINT_REVISION = "c68bd80fdd9c0d93d2f4a833cb154624f660a561"
CHECKPOINT_SHA256 = "99d9af7f1ff20e63c393035513a655392818d66b4d7fc23d658175c1f15e8d76"


def verified_checkpoint() -> Path:
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        kwargs = dict(repo_id=CHECKPOINT_REPO, filename="pytorch_model.bin",
                      revision=CHECKPOINT_REVISION, token=False)
        try:
            path = Path(hf_hub_download(**kwargs, local_files_only=True))
        except LocalEntryNotFoundError:
            path = Path(hf_hub_download(**kwargs))
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != CHECKPOINT_SHA256:
            raise RepairError(f"Official Apollo checkpoint checksum mismatch: {path}")
        return path
    except RepairError:
        raise
    except Exception as exc:
        raise RepairError(f"Could not obtain or verify the official Apollo checkpoint: {exc}") from exc


def repair(
    input_wav: Path, output_wav: Path, device: str, *,
    chunk_progress: Callable[[int, int], None] | None = None,
    verbose: bool = False,
) -> str:
    checkpoint = verified_checkpoint()
    try:
        output = sys.stderr if verbose else io.StringIO()
        with contextlib.redirect_stdout(output), warnings.catch_warnings():
            if not verbose:
                # PyTorch's internal MPS FFT output buffer warning from issue #174003.
                warnings.filterwarnings(
                    "ignore", message=r"An output with one or more elements was resized since it had shape \[\]",
                    category=UserWarning, module=r"^(torch\.functional|look2hear\.models\.apollo)$",
                )
            from inference import run_inference
            return str(run_inference(
                input_wav, output_wav, checkpoint, requested_device=device,
                chunk_seconds=CHUNK_SECONDS, overlap_seconds=OVERLAP_SECONDS,
                chunk_pad_seconds=CONTEXT_SECONDS, chunk_batch_size=CHUNK_BATCH_SIZE,
                chunk_progress=chunk_progress,
            ))
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        raise RepairError(f"Apollo inference failed: {exc}") from exc
