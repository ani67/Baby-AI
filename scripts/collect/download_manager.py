"""Stream-process-delete download manager.

Peak disk usage during a 600M-token corpus collection is dominated by
intermediate downloads (Stack Exchange .7z archives, Reddit .zst dumps,
Wikipedia bulks). If you keep them around, ~260 GB. If you delete each
source right after processing it, peak is bounded by the size of the
single file currently in flight.

Two primitives:

  stream_process_delete(url, out_jsonl, processor)
      Download → call processor on the downloaded file → write returned
      records to JSONL → delete the download. Used for Reddit .zst,
      single-file downloads.

  stream_7z_process_delete(url, out_jsonl, processor)
      Download .7z → extract → call processor on the extract dir →
      write records → delete BOTH the archive and the extract dir.
      Used for Stack Exchange dumps from archive.org.

Both wrap the download in try/finally so the temp files get deleted
even on processor exceptions. Peak disk = one source file + one
extract dir (for 7z); ~2 GB for SE per-site, ~5 GB for the larger ones.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterable

import requests

log = logging.getLogger('download_manager')

CHUNK_SIZE = 8 * 1024 * 1024            # 8MB streaming chunks
MAX_DISK_USAGE_GB = 20                  # advisory ceiling; check_disk_space enforces
MIN_FREE_GB_BEFORE_DOWNLOAD = 5.0       # refuse to download if less free
DEFAULT_TEMP_DIR = '/tmp/mind_collect'


def check_disk_space(path: str = '.') -> float:
    """Return available GB at `path`."""
    return shutil.disk_usage(path).free / (1024 ** 3)


def stream_process_delete(
    url: str,
    output_jsonl: str | Path,
    processor: Callable[[Path], Iterable[dict]],
    temp_dir: str = DEFAULT_TEMP_DIR,
    expected_hash: str | None = None,
    user_agent: str = 'baby-mind-curriculum-collector/2.0 (research)',
    request_timeout: int = 60,
) -> int:
    """Download a single file → process → delete.

    Returns the number of records appended to `output_jsonl`. JSONL is
    opened in append mode, so re-running this for the next source
    accumulates into the same file. The downloaded temp file is removed
    in a finally block — even on processor exceptions, disk doesn't leak.
    """
    temp_dir_p = Path(temp_dir)
    temp_dir_p.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir_p / hashlib.md5(url.encode()).hexdigest()

    free_gb = check_disk_space(str(temp_dir_p))
    if free_gb < MIN_FREE_GB_BEFORE_DOWNLOAD:
        raise RuntimeError(
            f"Low disk space: {free_gb:.1f} GB free at {temp_dir_p}; "
            f"need at least {MIN_FREE_GB_BEFORE_DOWNLOAD} GB to start a download"
        )

    try:
        log.info(f"download → {url}")
        log.info(f"  temp file: {temp_file}")
        log.info(f"  disk free before: {free_gb:.1f} GB")

        with requests.get(
            url,
            stream=True,
            timeout=request_timeout,
            headers={'User-Agent': user_agent},
        ) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get('content-length') or 0)
            downloaded = 0
            next_log_at = 50 * 1024 * 1024  # log every 50MB

            with open(temp_file, 'wb') as f:
                for chunk in resp.iter_content(CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_log_at:
                        mb = downloaded // 1024 // 1024
                        if total:
                            pct = downloaded / total * 100
                            log.info(f"  downloaded {mb} MB ({pct:.0f}%)")
                        else:
                            log.info(f"  downloaded {mb} MB")
                        next_log_at += 50 * 1024 * 1024

        if expected_hash:
            actual = hashlib.md5(temp_file.read_bytes()).hexdigest()
            if actual != expected_hash:
                raise ValueError(
                    f"Hash mismatch for {url}: {actual} != {expected_hash}"
                )

        log.info(f"processing {temp_file.name} ({temp_file.stat().st_size // 1024 // 1024} MB)")
        records = list(processor(temp_file))

        out_path = Path(output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'a', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

        log.info(f"appended {len(records)} records → {out_path}")
        return len(records)

    finally:
        if temp_file.exists():
            temp_file.unlink()
            log.info(f"deleted temp: {temp_file}")


def stream_7z_process_delete(
    url: str,
    output_jsonl: str | Path,
    processor: Callable[[Path], Iterable[dict]],
    temp_dir: str = DEFAULT_TEMP_DIR,
    user_agent: str = 'baby-mind-curriculum-collector/2.0 (research)',
) -> int:
    """Download .7z → extract → process → delete archive + extract dir.

    Requires `7z` or `7za` on PATH. Install via `brew install p7zip` on
    macOS. Raises RuntimeError if neither binary is available.

    The archive is deleted as soon as extraction completes, BEFORE
    processing — that frees the largest single file early. Peak disk is
    then bounded by the extracted size.
    """
    seven_zip = shutil.which('7z') or shutil.which('7za')
    if not seven_zip:
        raise RuntimeError(
            "Neither `7z` nor `7za` found on PATH. "
            "Install with: brew install p7zip"
        )

    temp_dir_p = Path(temp_dir)
    temp_dir_p.mkdir(parents=True, exist_ok=True)
    archive_name = Path(url).name
    archive_path = temp_dir_p / archive_name
    extract_dir = temp_dir_p / archive_path.stem

    free_gb = check_disk_space(str(temp_dir_p))
    if free_gb < MIN_FREE_GB_BEFORE_DOWNLOAD:
        raise RuntimeError(
            f"Low disk space: {free_gb:.1f} GB free at {temp_dir_p}"
        )

    try:
        log.info(f"download .7z → {url}")
        log.info(f"  disk free before: {free_gb:.1f} GB")

        with requests.get(
            url,
            stream=True,
            timeout=120,
            headers={'User-Agent': user_agent},
        ) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get('content-length') or 0)
            downloaded = 0
            next_log_at = 50 * 1024 * 1024
            with open(archive_path, 'wb') as f:
                for chunk in resp.iter_content(CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_log_at:
                        mb = downloaded // 1024 // 1024
                        if total:
                            log.info(f"  downloaded {mb} MB ({downloaded/total*100:.0f}%)")
                        else:
                            log.info(f"  downloaded {mb} MB")
                        next_log_at += 50 * 1024 * 1024

        archive_mb = archive_path.stat().st_size // 1024 // 1024
        log.info(f"  archive on disk: {archive_mb} MB")

        free_after_dl = check_disk_space(str(temp_dir_p))
        log.info(f"  disk free after download: {free_after_dl:.1f} GB")

        extract_dir.mkdir(exist_ok=True)
        log.info(f"extracting → {extract_dir}")
        result = subprocess.run(
            [seven_zip, 'x', str(archive_path), f'-o{extract_dir}', '-y'],
            check=True,
            capture_output=True,
        )
        log.info(f"  extraction OK")

        # delete archive immediately — keep extract dir for processing
        archive_path.unlink()
        log.info(f"  deleted archive: {archive_path.name}")

        free_after_extract = check_disk_space(str(temp_dir_p))
        log.info(f"  disk free after extract+archive-delete: {free_after_extract:.1f} GB")

        log.info(f"processing {extract_dir.name}")
        records = list(processor(extract_dir))

        out_path = Path(output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'a', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

        log.info(f"appended {len(records)} records → {out_path}")
        return len(records)

    finally:
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)
            log.info(f"deleted extract dir: {extract_dir.name}")
        if archive_path.exists():
            archive_path.unlink()
            log.info(f"deleted archive: {archive_path.name}")


def monitor_disk_during_collection(
    threshold_gb: float = 30.0,
    path: str = 'data/',
) -> bool:
    """Return True if disk is healthy at `path`. Log + return False below threshold.

    Call this periodically inside long collection loops. Cheap.
    """
    available = check_disk_space(path)
    log.info(f"disk available at {path}: {available:.1f} GB")
    if available < threshold_gb:
        log.warning(
            f"disk low: {available:.1f} GB < {threshold_gb} GB threshold; "
            "pause and free up space or reduce batch size"
        )
        return False
    return True
