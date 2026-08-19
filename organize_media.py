"""
Media Organizer — sorts photos/videos into categorized, dated folders using CLIP.

ENHANCEMENTS & FEATURES:
  - CLI Interface: Full argparse support for paths, dry-run/execute mode, batch size,
    confidence threshold, model selection, threads, and deduplication.
  - Multithreaded Processing: Parallel image loading and video frame extraction
    via ThreadPoolExecutor so GPU/CPU CLIP inference is never bottlenecked by I/O.
  - Confidence Thresholding: Prevents forced misclassifications by routing low-confidence
    predictions (< threshold) to a "Low_Confidence" category folder.
  - Explicit Prompt Mapping: Clean folder names (e.g. Portrait, Landscape, Wildlife)
    mapped explicitly to detailed CLIP visual prompts.
  - Duplicate Detection (--dedupe): Optional byte-level MD5 hashing to detect and route
    duplicate files to a dedicated "Duplicates" folder.
  - HEIC & RAW Decoding: pillow-heif and rawpy integration with 512px downscaling.
  - Video Frame Extraction: ffmpeg extraction at ~10% video duration.
  - Resumable & Safe: Checkpointing via JSON, full CSV logging, and hash-verified copy-delete for moves.
  - Progress Tracking & Summary: tqdm progress bar (with fallback) and detailed category summary table.

REQUIRED PACKAGES:
    pip install torch transformers pillow pillow-heif rawpy imageio tqdm --break-system-packages
  ffmpeg/ffprobe must be installed and on PATH for video processing.
"""

import os
import csv
import json
import shutil
import hashlib
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image, ExifTags

# Disable Pillow decompression bomb limit for ultra-high-res panoramas/gigapixel images
Image.MAX_IMAGE_PIXELS = None

# Optional progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Optional decoders — guarded so script runs even if missing
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    HEIF_SUPPORT = False

try:
    import rawpy
    RAW_SUPPORT = True
except ImportError:
    RAW_SUPPORT = False

import torch
from transformers import CLIPProcessor, CLIPModel

# ==========================================
# DEFAULT CATEGORIES (Folder Name -> CLIP Prompt)
# ==========================================
DEFAULT_CATEGORIES = {
    "Flora & Plants": "flowers, floral petals, plants, leaves, trees, or botanical close-up photo",
    "Vehicles & Aviation": "airplane, aircraft, jet, helicopter, car, or vehicle in the sky or on road",
    "Animals & Wildlife": "animal, pet, dog, cat, mammal, or reptile shown close-up",
    "Birds & Avian": "bird shown close-up as the main subject of the photo",
    "Portraits & People": "a portrait or close-up photo of a person or group of people",
    "Landscapes & Nature": "landscape scenery of mountains, water, sky, lakes, or forest as main subject",
    "Street & Urban": "street and urban city view with roads or traffic",
    "Architecture": "architecture, building exterior, facade, monument, or interior architectural design",
    "Food & Dining": "food, meal, dish, or dining setup",
    "Docs & Receipts": "a document, paper, receipt, invoice, or text screenshot",
    "Events & Parties": "an event, party, wedding, concert, or group celebration",
}

LEGACY_CATEGORY_MAP = {
    "Flora_Plants": "Flora & Plants",
    "Aviation_Vehicles": "Vehicles & Aviation",
    "Animal_Wildlife": "Animals & Wildlife",
    "Mammal_Reptile": "Animals & Wildlife",
    "Bird": "Birds & Avian",
    "Birds": "Birds & Avian",
    "Portrait": "Portraits & People",
    "Portraits": "Portraits & People",
    "Landscape": "Landscapes & Nature",
    "Landscapes": "Landscapes & Nature",
    "Street_Urban": "Street & Urban",
    "Architecture": "Architecture",
    "Food_Dining": "Food & Dining",
    "Document": "Docs & Receipts",
    "Documents": "Docs & Receipts",
    "Event": "Events & Parties",
    "Events": "Events & Parties",
}


def normalize_category_name(cat_name: str) -> str:
    if not cat_name:
        return "Uncategorized"
    return LEGACY_CATEGORY_MAP.get(cat_name.strip(), cat_name.strip())

# File Extension Filters (All Camera RAW Formats Supported)
RAW_EXTS = {
    ".arw", ".srf", ".sr2",         # Sony
    ".cr2", ".cr3", ".crw",         # Canon
    ".nef", ".nrw",                 # Nikon
    ".raf",                         # Fujifilm
    ".rw2",                         # Panasonic / Lumix
    ".orf",                         # Olympus / OM System
    ".dng", ".raw", ".rwl",         # Leica / Universal DNG
    ".pef", ".ptx",                 # Pentax
    ".3fr", ".fff",                 # Hasselblad
    ".iiq", ".cap",                 # Phase One / Mamiya
    ".x3f",                         # Sigma
    ".mrw",                         # Minolta / Konica
    ".srw",                         # Samsung
    ".kdc", ".dcr", ".k25", ".erf"  # Kodak / Epson
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tiff", ".tif", ".bmp", ".gif"} | RAW_EXTS
HEIF_EXTS = {".heic", ".heif"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".wmv", ".mxf", ".mts"}

SYSTEM_EXCLUDES = {
    "$recycle.bin", "system volume information", ".trash-1000",
    ".trashes", ".git", "node_modules", "appdata", "tmp",
}

EXCLUDE_FOLDERS = {
    "Games", "Effects", "M31 Backup", "S25 Backup", "PSDs", "LrCats",
    "Pixels '25", "WindowsApps", "Verilog", "Bestof2k25", "WpSystem", "site",
}


# ==========================================
# GPU ACCELERATION SETUP
# ==========================================
def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[GPU ACCELERATION] Enabled via CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[GPU ACCELERATION] Enabled via Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("[COMPUTE] GPU not detected. Running on CPU (consider GPU for faster processing).")
    return device


# ==========================================
# CHECKPOINTING & LOGGING
# ==========================================
def load_checkpoint(checkpoint_file: Path) -> set:
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r") as f:
                data = json.load(f)
            print(f"[RESUME] Found checkpoint with {len(data.get('done', []))} files already processed.")
            return set(data.get("done", []))
        except Exception as e:
            print(f"[WARN] Could not read checkpoint file, starting fresh: {e}")
    return set()


def save_checkpoint(checkpoint_file: Path, done_set: set):
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_file.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump({"done": sorted(done_set)}, f)
    tmp_path.replace(checkpoint_file)


def init_log(log_file: Path, dry_run: bool):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_file.exists()
    f = open(log_file, "a", newline="")
    writer = csv.writer(f)
    if is_new:
        writer.writerow(["timestamp", "source_path", "dest_path", "media_type", "category", "status", "detail"])
    else:
        writer.writerow([f"--- NEW RUN STARTED {datetime.now().isoformat(timespec='seconds')} (DRY_RUN={dry_run}) ---"])
    f.flush()
    return f, writer


def log_row(writer, log_file_handle, source_path, dest_path, media_type, category, status, detail=""):
    writer.writerow([datetime.now().isoformat(timespec="seconds"), str(source_path),
                      str(dest_path) if dest_path else "", media_type, category, status, detail])
    log_file_handle.flush()


# ==========================================
# IMAGE LOADING & DATE EXTRACTION
# ==========================================
def load_image_as_rgb(file_path: Path) -> Image.Image:
    ext = file_path.suffix.lower()

    if ext in RAW_EXTS:
        # Layer 1: Attempt rawpy decoding (fast thumbnail or postprocess)
        if RAW_SUPPORT:
            try:
                with rawpy.imread(str(file_path)) as raw:
                    try:
                        thumb = raw.extract_thumb()
                        if thumb.format == rawpy.ThumbFormat.JPEG:
                            import io
                            img = Image.open(io.BytesIO(thumb.data)).convert("RGB")
                            img.thumbnail((512, 512))
                            return img
                        elif thumb.format == rawpy.ThumbFormat.BITMAP:
                            img = Image.fromarray(thumb.data).convert("RGB")
                            img.thumbnail((512, 512))
                            return img
                    except Exception:
                        pass

                    rgb_array = raw.postprocess(use_camera_wb=True, half_size=True)
                    img = Image.fromarray(rgb_array)
                    img.thumbnail((512, 512))
                    return img
            except Exception:
                pass

        # Layer 2: Pillow direct open fallback (decodes many DNG / TIFF formats)
        try:
            with Image.open(file_path) as img:
                rgb_img = img.convert("RGB")
                rgb_img.thumbnail((512, 512))
                return rgb_img
        except Exception as e:
            raise RuntimeError(f"Could not decode RAW/DNG file: {e}")

    if ext in HEIF_EXTS and not HEIF_SUPPORT:
        raise RuntimeError("pillow-heif not installed — cannot decode HEIC files.")

    with Image.open(file_path) as img:
        rgb_img = img.convert("RGB")
        rgb_img.thumbnail((512, 512))
        return rgb_img


def get_image_date(file_path: Path) -> str:
    try:
        with Image.open(file_path) as img:
            exif = img.getexif()
            if exif:
                date_val = exif.get(36867) or exif.get(306)
                if not date_val:
                    for tag, value in exif.items():
                        tag_name = ExifTags.TAGS.get(tag, tag)
                        if tag_name in ("DateTimeOriginal", "DateTime"):
                            date_val = value
                            break
                if date_val:
                    s = str(date_val).strip()
                    dt = datetime.strptime(s[:19], "%Y:%m:%d %H:%M:%S")
                    return dt.strftime("%Y/%Y-%m")
    except Exception:
        pass

    return _mtime_fallback(file_path)


def get_video_date(file_path: Path) -> str:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(file_path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            meta = json.loads(result.stdout)
            tags = meta.get("format", {}).get("tags", {})
            creation_time = tags.get("creation_time") or tags.get("com.apple.quicktime.creationdate")
            if creation_time:
                dt = datetime.strptime(creation_time[:19], "%Y-%m-%dT%H:%M:%S")
                return dt.strftime("%Y/%Y-%m")
    except Exception:
        pass

    return _mtime_fallback(file_path)


def _mtime_fallback(file_path: Path) -> str:
    try:
        mtime = os.path.getmtime(file_path)
        return datetime.fromtimestamp(mtime).strftime("%Y/%Y-%m")
    except Exception:
        return "Unknown_Date"


def check_ffmpeg_available() -> bool:
    for exe in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([exe, "-version"], capture_output=True, timeout=10)
        except Exception:
            return False
    return True


def extract_video_frame(file_path: Path, out_path: Path) -> bool:
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(file_path)],
            capture_output=True, text=True, timeout=30,
        )
        duration = 1.0
        if probe.returncode == 0:
            meta = json.loads(probe.stdout)
            duration = float(meta.get("format", {}).get("duration", 1.0))
        seek_time = max(0.5, duration * 0.1)

        result = subprocess.run(
            ["ffmpeg", "-y", "-ss", str(seek_time), "-i", str(file_path),
             "-frames:v", "1", "-q:v", "2", str(out_path)],
            capture_output=True, timeout=30,
        )
        return result.returncode == 0 and out_path.exists()
    except Exception:
        return False


# ==========================================
# CLIP MODEL & BATCH CLASSIFICATION
# ==========================================
def load_clip_model(model_name: str, device: torch.device):
    print(f"Loading CLIP model '{model_name}' into memory...")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def classify_image_batch(images: list, model, processor, categories_dict: dict,
                         confidence_threshold: float, device: torch.device) -> list:
    if not images:
        return []
    folder_names = list(categories_dict.keys())
    text_prompts = [categories_dict[k] for k in folder_names]
    try:
        inputs = processor(text=text_prompts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
            max_probs, best_idxs = probs.max(dim=1)

        results = []
        for max_p, idx in zip(max_probs.tolist(), best_idxs.tolist()):
            if max_p < confidence_threshold:
                results.append("Low_Confidence")
            else:
                results.append(folder_names[idx])
        return results
    except Exception as e:
        print(f"\n[WARN] Batch classification failed, falling back to Uncategorized: {e}")
        return ["Uncategorized"] * len(images)


# ==========================================
# FILE SYSTEM & TRANSFER HELPERS
# ==========================================
def should_skip_directory(dir_name: str, exclude_names: set) -> bool:
    if dir_name.startswith("."):
        return True
    return dir_name.lower() in exclude_names


def scan_drive(drive_root: Path, dest_dir: Path, user_exclude_folders: set = None):
    dest_dir_resolved = dest_dir.resolve()
    exclude_names = {d.lower() for d in SYSTEM_EXCLUDES}
    if user_exclude_folders:
        exclude_names.update({d.lower().strip() for d in user_exclude_folders if d.strip()})
    else:
        exclude_names.update({d.lower() for d in EXCLUDE_FOLDERS})

    for root, dirs, files in os.walk(drive_root, topdown=True, followlinks=False):
        current_path = Path(root).resolve()

        if current_path == dest_dir_resolved or dest_dir_resolved in current_path.parents:
            dirs[:] = []
            continue

        dirs[:] = [d for d in dirs if not should_skip_directory(d, exclude_names)]

        for file in files:
            file_path = current_path / file
            ext = file_path.suffix.lower()
            if ext in IMAGE_EXTS or ext in VIDEO_EXTS:
                yield file_path, (ext in IMAGE_EXTS)



def get_base_target_dir(file_path: Path, drive_root: Path, dest_dir: Path, preserve_folders: bool) -> Path:
    if not preserve_folders:
        return dest_dir
    try:
        rel_parent = file_path.parent.resolve().relative_to(drive_root.resolve())
    except ValueError:
        return dest_dir
    if str(rel_parent) == ".":
        return dest_dir
    return dest_dir / rel_parent


def get_unique_target_path(target_dir: Path, filename: str) -> Path:
    target_path = target_dir / filename
    if not target_path.exists():
        return target_path
    stem, suffix = target_path.stem, target_path.suffix
    counter = 1
    while target_path.exists():
        target_path = target_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    return target_path


def _file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def safe_transfer(src: Path, dst: Path, action: str) -> None:
    shutil.copy2(src, dst)
    if action == "move":
        if _file_hash(src) != _file_hash(dst):
            raise IOError(f"Content hash mismatch after copy — refusing to delete original: {src}")
        src.unlink()


def safe_restore_file(dest_path: Path, source_path: Path) -> None:
    source_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.resolve() == source_path.resolve():
        return
    shutil.copy2(dest_path, source_path)
    if _file_hash(dest_path) != _file_hash(source_path):
        raise IOError(f"Hash mismatch after restore — refusing to delete organized file: {dest_path}")
    dest_path.unlink()


def remove_empty_folders(root_dir: Path):
    if not root_dir.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        p = Path(dirpath)
        if p == root_dir.resolve():
            continue
        try:
            if not os.listdir(p):
                p.rmdir()
        except Exception:
            pass


# ==========================================
# PARALLEL PREPROCESSING WORKER
# ==========================================
def preprocess_single_file(item, classify_videos: bool, tmp_frame_dir: Path, check_dedupe: bool):
    file_path, is_image = item
    pil_img = None
    tmp_frame = None
    media_date = "Unknown_Date"
    err_msg = None
    file_hash = None

    try:
        if check_dedupe:
            file_hash = _file_hash(file_path)

        if is_image:
            pil_img = load_image_as_rgb(file_path)
            media_date = get_image_date(file_path)
        else:
            media_date = get_video_date(file_path)
            if classify_videos:
                path_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
                tmp_frame = tmp_frame_dir / f"{file_path.stem}_{path_hash}_frame.jpg"
                if extract_video_frame(file_path, tmp_frame):
                    pil_img = Image.open(tmp_frame).convert("RGB")
                    pil_img.thumbnail((512, 512))
                else:
                    tmp_frame = None
    except Exception as e:
        err_msg = str(e)

    return (file_path, is_image, pil_img, tmp_frame, media_date, file_hash, err_msg)


# ==========================================
# UNIFIED FILE TRANSFER & LOGGING HELPER
# ==========================================
def transfer_and_log_file(file_path: Path, is_video: bool, category: str, media_date: str,
                          file_hash: str, tmp_frame: Path, drive_root: Path, dest_dir: Path,
                          preserve_folders: bool, dry_run: bool, action: str, check_dedupe: bool,
                          known_hashes: set, done_set: set, log_writer, log_f, stats: dict):
    try:
        final_cat = normalize_category_name(category)
        if check_dedupe and file_hash:
            if file_hash in known_hashes:
                final_cat = "Duplicates"
                stats["duplicates_count"] += 1
            else:
                known_hashes.add(file_hash)

        base_target = get_base_target_dir(file_path, drive_root, dest_dir, preserve_folders)
        target_dir = base_target / ("Videos" if is_video else "Photos") / final_cat
        target_path = get_unique_target_path(target_dir, file_path.name)
        media_type = "VIDEO" if is_video else "PHOTO"

        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
            safe_transfer(file_path, target_path, action)
            done_set.add(str(file_path))

        log_row(log_writer, log_f, file_path, target_path, media_type, final_cat, "OK")
        stats["categories"][final_cat] = stats["categories"].get(final_cat, 0) + 1
        if is_video:
            stats["videos_count"] += 1
        else:
            stats["photos_count"] += 1
    except Exception as e:
        stats["error_count"] += 1
        log_row(log_writer, log_f, file_path, None, "VIDEO" if is_video else "PHOTO", category, "ERROR", str(e))
    finally:
        if tmp_frame and tmp_frame.exists():
            try:
                tmp_frame.unlink()
            except Exception:
                pass


# ==========================================
# SUMMARY REPORTING
# ==========================================
def print_summary_report(stats: dict, dry_run: bool, log_file: Path, checkpoint_file: Path):
    print("\n" + "=" * 60)
    print("                 MEDIA ORGANIZER SUMMARY REPORT             ")
    print("=" * 60)
    print(f"Mode:              {'DRY RUN (No files modified)' if dry_run else 'EXECUTED (' + stats['action'].upper() + ')'}")
    print(f"Total Discovered:  {stats['total_discovered']}")
    print(f"Processed Photos:  {stats['photos_count']}")
    print(f"Processed Videos:  {stats['videos_count']}")
    print(f"Skipped (Done):    {stats['skipped_count']}")
    print(f"Duplicates Found:  {stats['duplicates_count']}")
    print(f"Errors Encountered:{stats['error_count']}")
    print("-" * 60)
    print("Category Breakdown:")
    for cat, count in sorted(stats['categories'].items()):
        print(f"  - {cat:<20}: {count} files")
    print("-" * 60)
    print(f"Full CSV Log:      {log_file}")
    if not dry_run:
        print(f"Checkpoint File:   {checkpoint_file}")
    print("=" * 60 + "\n")


# ==========================================
# CLI ARGUMENT PARSER
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Media Organizer powered by CLIP AI")
    parser.add_argument("--src", type=str, default="E:/", help="Source drive or directory to scan")
    parser.add_argument("--dest", type=str, default="E:/Organized_Media", help="Destination root directory")
    parser.add_argument("--action", choices=["copy", "move"], default="copy", help="File action: copy or move")
    parser.add_argument("--execute", action="store_true", help="Perform real transfers (default is dry run)")
    parser.add_argument("--confidence", type=float, default=0.35, help="Minimum confidence threshold (0.0 - 1.0)")
    parser.add_argument("--model", type=str, default="openai/clip-vit-large-patch14-336", help="HuggingFace CLIP model name (e.g. openai/clip-vit-large-patch14-336)")
    parser.add_argument("--batch-size", type=int, default=16, help="CLIP batch size")
    parser.add_argument("--threads", type=int, default=4, help="Worker threads for image decoding & frame extraction")
    parser.add_argument("--no-video-classify", action="store_true", help="Disable video frame classification")
    parser.add_argument("--flat-folders", action="store_true", help="Do not preserve original relative folder structure")
    parser.add_argument("--dedupe", action="store_true", help="Route byte-for-byte duplicate files to Duplicates folder")
    parser.add_argument("--exclude", type=str, default="Games,Effects,M31 Backup,S25 Backup,PSDs,LrCats,Pixels '25,WindowsApps,Verilog,Bestof2k25,WpSystem,site", help="Comma-separated folder names to exclude")
    return parser.parse_args()


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    args = parse_args()
    drive_root = Path(args.src)
    dest_dir = Path(args.dest)
    dry_run = not args.execute
    classify_videos = not args.no_video_classify
    preserve_folders = not args.flat_folders
    user_excludes = {x.strip() for x in args.exclude.split(",") if x.strip()}

    checkpoint_file = dest_dir / "_organizer_checkpoint.json"
    log_file = dest_dir / "_organizer_log.csv"

    print("=" * 60)
    print(" MEDIA ORGANIZER SETUP ")
    print("=" * 60)
    print(f"Source:           {drive_root}")
    print(f"Destination:      {dest_dir}")
    print(f"Action:           {args.action.upper()}")
    print(f"Dry Run:          {dry_run}")
    print(f"Confidence Thresh:{args.confidence}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Worker Threads:   {args.threads}")
    print(f"Deduplication:    {args.dedupe}")
    print(f"Excluded Folders: {args.exclude}")
    print("=" * 60)

    if not HEIF_SUPPORT:
        print("[WARN] pillow-heif not installed — HEIC files will fail.")
    if not RAW_SUPPORT:
        print("[WARN] rawpy not installed — RAW files will fail.")

    ffmpeg_ok = check_ffmpeg_available()
    if classify_videos and not ffmpeg_ok:
        print("[WARN] ffmpeg/ffprobe not found on PATH. Falling back to date-only video sorting.")
        classify_videos = False

    device = get_torch_device()
    model, processor = load_clip_model(args.model, device)
    done_set = load_checkpoint(checkpoint_file)
    log_f, log_writer = init_log(log_file, dry_run)

    known_hashes = set()

    tmp_frame_dir = dest_dir / ".tmp_organizer_frames"
    tmp_frame_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "action": args.action,
        "total_discovered": 0,
        "photos_count": 0,
        "videos_count": 0,
        "skipped_count": 0,
        "duplicates_count": 0,
        "error_count": 0,
        "categories": {}
    }

    # Discover candidate files
    discovered_files = list(scan_drive(drive_root, dest_dir, user_excludes))

    stats["total_discovered"] = len(discovered_files)
    print(f"\nDiscovered {len(discovered_files)} candidate media files.")

    image_batch, image_batch_meta = [], []

    def flush_batch():
        nonlocal image_batch, image_batch_meta
        if not image_batch:
            return
        categories = classify_image_batch(image_batch, model, processor, DEFAULT_CATEGORIES, args.confidence, device)
        for (file_path, is_video, tmp_frame, media_date, file_hash), category in zip(image_batch_meta, categories):
            transfer_and_log_file(file_path, is_video, category, media_date, file_hash, tmp_frame,
                                 drive_root, dest_dir, preserve_folders, dry_run, args.action,
                                 args.dedupe, known_hashes, done_set, log_writer, log_f, stats)
        image_batch.clear()
        image_batch_meta.clear()

    # Process files using ThreadPoolExecutor for multithreaded decoding / frame extraction
    pbar = tqdm(total=len(discovered_files), desc="Processing media") if HAS_TQDM else None

    processed_since_checkpoint = 0
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {}
        for item in discovered_files:
            file_path, _ = item
            if str(file_path) in done_set:
                stats["skipped_count"] += 1
                if pbar:
                    pbar.update(1)
                continue
            future = executor.submit(preprocess_single_file, item, classify_videos, tmp_frame_dir, args.dedupe)
            futures[future] = item

        for future in as_completed(futures):
            if pbar:
                pbar.update(1)
            file_path, is_image, pil_img, tmp_frame, media_date, file_hash, err_msg = future.result()
            is_video = not is_image

            if err_msg:
                stats["error_count"] += 1
                log_row(log_writer, log_f, file_path, None, "VIDEO" if is_video else "PHOTO", "", "ERROR", err_msg)
                continue

            if pil_img:
                image_batch.append(pil_img)
                image_batch_meta.append((file_path, is_video, tmp_frame, media_date, file_hash))
            else:
                # Video with failed frame extraction or video classification disabled
                transfer_and_log_file(file_path, is_video, "Uncategorized", media_date, file_hash, tmp_frame,
                                     drive_root, dest_dir, preserve_folders, dry_run, args.action,
                                     args.dedupe, known_hashes, done_set, log_writer, log_f, stats)

            if len(image_batch) >= args.batch_size:
                flush_batch()

            processed_since_checkpoint += 1
            if processed_since_checkpoint >= 100:
                flush_batch()
                if not dry_run:
                    save_checkpoint(checkpoint_file, done_set)
                processed_since_checkpoint = 0

    flush_batch()
    if pbar:
        pbar.close()

    if not dry_run:
        save_checkpoint(checkpoint_file, done_set)

    log_f.close()
    shutil.rmtree(tmp_frame_dir, ignore_errors=True)

    print_summary_report(stats, dry_run, log_file, checkpoint_file)


if __name__ == "__main__":
    main()
