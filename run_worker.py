"""
Worker module and CLI wrapper for running media organization and unsort/restore tasks
in a background thread or standalone process.
"""

import os
import sys
import csv
import json
import time
import queue
import argparse
import shutil
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from organize_media import (
    DEFAULT_CATEGORIES,
    EXCLUDE_FOLDERS,
    SYSTEM_EXCLUDES,
    IMAGE_EXTS,
    VIDEO_EXTS,
    check_ffmpeg_available,
    load_clip_model,
    get_torch_device,
    load_checkpoint,
    save_checkpoint,
    init_log,
    log_row,
    scan_drive,
    preprocess_single_file,
    classify_image_batch,
    transfer_and_log_file,
    safe_restore_file,
    remove_empty_folders,
)


def log_message(msg_queue: queue.Queue | None, level: str, text: str) -> None:
    """Helper to emit log messages to queue or print to stdout."""
    if msg_queue:
        msg_queue.put(("LOG", (level, text)))
    else:
        print(f"[{level}] {text}")


def run_unsort_worker(
    dest_dir: Path | str,
    dry_run: bool = False,
    msg_queue: queue.Queue | None = None,
    cancel_flag: threading.Event | None = None,
) -> dict:
    """
    Restores organized media files back to their exact original source directories
    using the CSV log (_organizer_log.csv).
    """
    dest_path = Path(dest_dir).resolve()
    log_file = dest_path / "_organizer_log.csv"
    checkpoint_file = dest_path / "_organizer_checkpoint.json"

    log_message(msg_queue, "INFO", "=== UNSORT & RESTORE STARTED ===")
    log_message(msg_queue, "INFO", f"Organized Root: {dest_path}")
    log_message(msg_queue, "INFO", f"Log File:       {log_file}")
    log_message(
        msg_queue,
        "INFO",
        f"Execution Mode: {'DRY RUN' if dry_run else 'REAL RESTORE'}",
    )

    stats = {
        "restored_count": 0,
        "already_original_count": 0,
        "missing_count": 0,
        "error_count": 0,
    }

    if not log_file.exists():
        err_msg = f"Log file not found at {log_file}. Cannot unsort without a valid _organizer_log.csv."
        log_message(msg_queue, "ERROR", err_msg)
        if msg_queue:
            msg_queue.put(("DONE", (stats, False, log_file, checkpoint_file)))
        return stats

    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        log_message(msg_queue, "ERROR", f"Failed to read log file: {e}")
        if msg_queue:
            msg_queue.put(("DONE", (stats, False, log_file, checkpoint_file)))
        return stats

    valid_rows = [
        r
        for r in rows
        if r.get("status") == "OK" and r.get("source_path") and r.get("dest_path")
    ]
    total = len(valid_rows)
    log_message(msg_queue, "INFO", f"Discovered {total} valid transfer log entries.")

    for idx, row in enumerate(valid_rows, start=1):
        if cancel_flag and cancel_flag.is_set():
            log_message(msg_queue, "WARN", "Unsort process cancelled by user.")
            break

        src = Path(row["source_path"])
        dst = Path(row["dest_path"])

        if not dst.exists():
            if src.exists():
                stats["already_original_count"] += 1
            else:
                stats["missing_count"] += 1
                log_message(
                    msg_queue, "WARN", f"[MISSING] Organized file not found: {dst}"
                )
            continue

        if dry_run:
            stats["restored_count"] += 1
            log_message(msg_queue, "OK", f"[PREVIEW RESTORE] {dst.name} ➔ {src}")
        else:
            try:
                safe_restore_file(dst, src)
                stats["restored_count"] += 1
                log_message(msg_queue, "OK", f"[RESTORED] {dst.name} ➔ {src}")
            except Exception as e:
                stats["error_count"] += 1
                log_message(
                    msg_queue, "ERROR", f"[ERROR] Failed to restore {dst.name}: {e}"
                )

        if msg_queue:
            msg_queue.put(
                (
                    "PROGRESS",
                    (idx, total, f"Restoring {idx}/{total}..."),
                )
            )

    if not dry_run and not (cancel_flag and cancel_flag.is_set()):
        log_message(msg_queue, "INFO", "Cleaning up empty organized subdirectories...")
        remove_empty_folders(dest_path)
        if checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                log_message(
                    msg_queue, "INFO", f"Deleted checkpoint file: {checkpoint_file}"
                )
            except Exception:
                pass

    log_message(
        msg_queue,
        "PURPLE",
        f"Unsort finished. Restored: {stats['restored_count']} files, Already Original: {stats['already_original_count']}, Missing: {stats['missing_count']}, Errors: {stats['error_count']}.",
    )

    if msg_queue:
        msg_queue.put(("DONE", (stats, True, log_file, checkpoint_file)))

    return stats


def run_worker(
    config: dict,
    msg_queue: queue.Queue | None = None,
    cancel_flag: threading.Event | None = None,
) -> dict:
    """
    Executes the CLIP media organization or unsort workflow based on configuration dict.

    Config keys:
        src (Path/str): Source directory to scan (required for organization).
        dest (Path/str): Destination root folder (required).
        action (str): "copy", "move", or "unsort".
        dry_run (bool): Perform dry run without copying/moving/deleting files.
        ...
    """
    action = str(config.get("action", "copy")).lower()
    if action == "unsort":
        return run_unsort_worker(
            dest_dir=config["dest"],
            dry_run=bool(config.get("dry_run", False)),
            msg_queue=msg_queue,
            cancel_flag=cancel_flag,
        )

    src_dir = Path(config["src"]).resolve()
    dest_dir = Path(config["dest"]).resolve()
    model_name = config.get("model_name", "openai/clip-vit-base-patch32")
    batch_size = int(config.get("batch_size", 32))
    confidence_thresh = float(config.get("confidence", 0.35))
    dry_run = bool(config.get("dry_run", False))
    classify_videos = bool(config.get("classify_videos", True))
    check_dedupe = bool(config.get("dedupe", False))
    preserve_folders = bool(config.get("preserve_folders", False))
    exclude_str = config.get("exclude_folders", "")
    num_threads = int(config.get("threads", 4))
    categories = config.get("categories", DEFAULT_CATEGORIES)

    log_message(msg_queue, "INFO", f"Scanning source: {src_dir}")
    log_message(msg_queue, "INFO", f"Destination:   {dest_dir}")
    log_message(
        msg_queue,
        "INFO",
        f"Execution Mode: {'DRY RUN' if dry_run else f'REAL EXECUTE ({action.upper()})'}",
    )
    if exclude_str:
        log_message(msg_queue, "INFO", f"Excluded Folders: {exclude_str}")

    ffmpeg_ok = check_ffmpeg_available()
    if classify_videos and not ffmpeg_ok:
        log_message(
            msg_queue,
            "WARN",
            "ffmpeg/ffprobe not found on PATH. Video sorting will fall back to date-only.",
        )
        classify_videos = False

    device = get_torch_device()
    log_message(
        msg_queue,
        "PURPLE",
        f"Loading CLIP model '{model_name}' on device '{device}'...",
    )
    model, processor = load_clip_model(model_name, device)

    checkpoint_file = dest_dir / "_organizer_checkpoint.json"
    log_file = dest_dir / "_organizer_log.csv"

    done_set = load_checkpoint(checkpoint_file)
    log_f, log_writer = init_log(log_file, dry_run)

    known_hashes = set()
    tmp_frame_dir = dest_dir / ".tmp_organizer_frames"
    tmp_frame_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "action": action,
        "total_discovered": 0,
        "photos_count": 0,
        "videos_count": 0,
        "skipped_count": 0,
        "duplicates_count": 0,
        "error_count": 0,
        "categories": {},
    }

    user_excludes = {x.strip() for x in exclude_str.split(",") if x.strip()}
    discovered = list(scan_drive(src_dir, dest_dir, user_excludes))
    stats["total_discovered"] = len(discovered)
    log_message(
        msg_queue,
        "INFO",
        f"Discovered {len(discovered)} candidate media files.",
    )

    pending_items = [
        item for item in discovered if str(item[0]) not in done_set
    ]
    stats["skipped_count"] = len(discovered) - len(pending_items)

    if msg_queue:
        msg_queue.put(("STATS", stats.copy()))

    if not pending_items:
        log_message(
            msg_queue,
            "SUCCESS",
            "All discovered files have already been processed in previous runs.",
        )
        if msg_queue:
            msg_queue.put(("DONE", (stats, True, log_file, checkpoint_file)))
        log_f.close()
        shutil.rmtree(tmp_frame_dir, ignore_errors=True)
        return stats

    log_message(
        msg_queue,
        "INFO",
        f"Processing {len(pending_items)} files ({stats['skipped_count']} already checkpointed)...",
    )

    processed_count = 0
    total_pending = len(pending_items)

    try:
        for i in range(0, total_pending, batch_size):
            if cancel_flag and cancel_flag.is_set():
                log_message(
                    msg_queue,
                    "WARN",
                    "Organization process cancelled by user request.",
                )
                break

            chunk = pending_items[i : i + batch_size]

            # Parallel preprocessing (image decoding, EXIF date extraction, video frame extraction)
            preprocessed_results = []
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(
                        preprocess_single_file,
                        item,
                        classify_videos,
                        tmp_frame_dir,
                        check_dedupe,
                    )
                    for item in chunk
                ]
                for future in as_completed(futures):
                    try:
                        preprocessed_results.append(future.result())
                    except Exception as exc:
                        log_message(
                            msg_queue,
                            "ERROR",
                            f"Preprocessing failed for item: {exc}",
                        )

            # Separate items needing CLIP inference vs fallback items
            clip_batch_imgs = []
            clip_batch_indices = []
            final_classifications = [None] * len(preprocessed_results)

            for idx, res in enumerate(preprocessed_results):
                (
                    file_path,
                    is_image,
                    pil_img,
                    tmp_frame,
                    media_date,
                    file_hash,
                    err_msg,
                ) = res

                if err_msg or pil_img is None:
                    final_classifications[idx] = "Uncategorized"
                    if err_msg:
                        log_row(
                            log_writer,
                            log_f,
                            file_path,
                            "",
                            "VIDEO" if not is_image else "PHOTO",
                            "Uncategorized",
                            "ERROR",
                            err_msg,
                        )
                        stats["error_count"] += 1
                else:
                    clip_batch_imgs.append(pil_img)
                    clip_batch_indices.append(idx)

            # Batch GPU/CPU CLIP classification
            if clip_batch_imgs:
                cat_predictions = classify_image_batch(
                    clip_batch_imgs,
                    model,
                    processor,
                    categories,
                    confidence_thresh,
                    device,
                )
                for idx, cat_name in zip(clip_batch_indices, cat_predictions):
                    final_classifications[idx] = cat_name

            # Transfer files and record progress
            for res, cat_name in zip(
                preprocessed_results, final_classifications
            ):
                (
                    file_path,
                    is_image,
                    pil_img,
                    tmp_frame,
                    media_date,
                    file_hash,
                    err_msg,
                ) = res

                if cat_name != "Uncategorized" or not err_msg:
                    transfer_and_log_file(
                        file_path=file_path,
                        is_video=not is_image,
                        category=cat_name or "Uncategorized",
                        media_date=media_date,
                        file_hash=file_hash,
                        tmp_frame=tmp_frame,
                        drive_root=src_dir,
                        dest_dir=dest_dir,
                        preserve_folders=preserve_folders,
                        dry_run=dry_run,
                        action=action,
                        check_dedupe=check_dedupe,
                        known_hashes=known_hashes,
                        done_set=done_set,
                        log_writer=log_writer,
                        log_f=log_f,
                        stats=stats,
                    )

                media_type = "VIDEO" if not is_image else "PHOTO"
                log_message(msg_queue, "OK", f"[{media_type}] {file_path.name} ➔ {cat_name or 'Uncategorized'}")

                if tmp_frame and tmp_frame.exists():
                    try:
                        tmp_frame.unlink()
                    except Exception:
                        pass

                processed_count += 1
                if msg_queue:
                    msg_queue.put(
                        (
                            "PROGRESS",
                            (
                                processed_count,
                                total_pending,
                                f"Processing {file_path.name} ➔ {cat_name}",
                                stats.copy(),
                            ),
                        )
                    )

            if not dry_run:
                save_checkpoint(checkpoint_file, done_set)

    except Exception as e:
        log_message(
            msg_queue, "ERROR", f"Worker loop encountered unhandled error: {e}"
        )
    finally:
        log_f.close()
        shutil.rmtree(tmp_frame_dir, ignore_errors=True)

    log_message(
        msg_queue,
        "SUCCESS",
        f"Completed organizing media! Photos: {stats['photos_count']}, Videos: {stats['videos_count']}, Errors: {stats['error_count']}",
    )

    if msg_queue:
        msg_queue.put(("DONE", (stats, True, log_file, checkpoint_file)))

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CLIP media organizer or unsort/restore worker process."
    )
    parser.add_argument(
        "--src", default=None, help="Source folder to scan for photos/videos (required for organization)."
    )
    parser.add_argument(
        "--dest", required=True, help="Destination directory containing organized media / _organizer_log.csv."
    )
    parser.add_argument(
        "--action", choices=["copy", "move", "unsort"], default="copy", help="File action or unsort mode."
    )
    parser.add_argument(
        "--unsort", action="store_true", help="Unsort & restore files to original paths using log."
    )
    parser.add_argument(
        "--model",
        default="openai/clip-vit-base-patch32",
        help="CLIP model name.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="CLIP batch size."
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Minimum confidence threshold.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry-run without writing/deleting files.",
    )
    parser.add_argument(
        "--classify-videos",
        action="store_true",
        default=True,
        help="Classify videos via frame extraction.",
    )
    parser.add_argument(
        "--dedupe", action="store_true", help="Detect duplicate files."
    )
    parser.add_argument(
        "--preserve-folders",
        action="store_true",
        help="Preserve source subfolder structure.",
    )
    parser.add_argument(
        "--exclude", default="", help="Comma-separated excluded folder names."
    )
    parser.add_argument(
        "--threads", type=int, default=4, help="Number of preprocessing threads."
    )

    args = parser.parse_args()

    if args.unsort or args.action == "unsort":
        run_unsort_worker(dest_dir=args.dest, dry_run=args.dry_run)
    else:
        if not args.src:
            parser.error("--src parameter is required when running organization (copy/move).")

        cfg = {
            "src": args.src,
            "dest": args.dest,
            "model_name": args.model,
            "batch_size": args.batch_size,
            "confidence": args.confidence,
            "action": args.action,
            "dry_run": args.dry_run,
            "classify_videos": args.classify_videos,
            "dedupe": args.dedupe,
            "preserve_folders": args.preserve_folders,
            "exclude_folders": args.exclude,
            "threads": args.threads,
        }

        run_worker(cfg)
