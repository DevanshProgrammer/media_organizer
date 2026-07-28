"""
AI Media Organizer — Authentic Glassmorphism Web-Desktop Backend Server.

Provides a lightweight local HTTP server with Server-Sent Events (SSE) streaming,
native OS folder picker dialog integration, and background CLIP AI media classification.
"""

import os
import csv
import json
import queue
import shutil
import time
import hashlib
import threading
import subprocess
import webbrowser
from pathlib import Path
from datetime import datetime
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor, as_completed

import tkinter as tk
from tkinter import filedialog
from run_worker import run_worker as execute_run_worker, run_unsort_worker as execute_run_unsort_worker

from PIL import Image, ExifTags

# Disable Pillow decompression bomb limit for ultra-high-res panoramas/gigapixel images
Image.MAX_IMAGE_PIXELS = None

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

# Default Categories
DEFAULT_CATEGORIES = {
    "Flora_Plants": "flowers, floral petals, plants, leaves, trees, or botanical close-up photo",
    "Aviation_Vehicles": "airplane, aircraft, jet, helicopter, car, or vehicle in the sky or on road",
    "Animal_Wildlife": "animal, pet, dog, cat, mammal, or reptile shown close-up",
    "Bird": "bird shown close-up as the main subject of the photo",
    "Portrait": "a portrait or close-up photo of a person or group of people",
    "Landscape": "landscape scenery of mountains, water, sky, lakes, or forest as main subject",
    "Street_Urban": "street and urban city view with roads or traffic",
    "Architecture": "architecture, building exterior, facade, monument, or interior architectural design",
    "Food_Dining": "food, meal, dish, or dining setup",
    "Document": "a document, paper, receipt, invoice, or text screenshot",
    "Event": "an event, party, wedding, concert, or group celebration",
}

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
SYSTEM_EXCLUDES = {"$recycle.bin", "system volume information", ".trash-1000", ".trashes", ".git", "node_modules", "appdata", "tmp"}
EXCLUDE_FOLDERS = {"Games", "Effects", "M31 Backup", "S25 Backup", "PSDs", "LrCats", "Pixels '25", "WindowsApps", "Verilog", "Bestof2k25", "WpSystem", "site"}

# Global execution state
active_worker = None
sse_subscribers = set()
subscriber_lock = threading.Lock()


def get_torch_device() -> tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), f"CUDA: {torch.cuda.get_device_name(0)}"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Silicon MPS"
    else:
        return torch.device("cpu"), "CPU (No GPU Detected)"


def load_checkpoint(checkpoint_file: Path) -> set:
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r") as f:
                data = json.load(f)
            return set(data.get("done", []))
        except Exception:
            pass
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


def load_clip_model(model_name: str, device: torch.device):
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
    except Exception:
        return ["Uncategorized"] * len(images)


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


# Broadcast message to SSE clients
def broadcast_sse(event_type: str, data: dict):
    payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    with subscriber_lock:
        to_remove = set()
        for q in sse_subscribers:
            try:
                q.put_nowait(payload)
            except Exception:
                to_remove.add(q)
        sse_subscribers.difference_update(to_remove)


class OrganizerWorkerThread(threading.Thread):
    def __init__(self, config: dict):
        super().__init__(daemon=True)
        self.config = config
        self.cancel_event = threading.Event()

    def stop(self):
        self.cancel_event.set()

    def run(self):
        msg_queue = queue.Queue()
        cancel_event = self.cancel_event

        def _worker_entry():
            action = str(self.config.get("action", "copy")).lower()
            if action == "unsort":
                execute_run_unsort_worker(
                    dest_dir=self.config["dest"],
                    dry_run=bool(self.config.get("dry_run", False)),
                    msg_queue=msg_queue,
                    cancel_flag=cancel_event,
                )
            else:
                execute_run_worker(
                    self.config,
                    msg_queue=msg_queue,
                    cancel_flag=cancel_event,
                )

        worker_thread = threading.Thread(target=_worker_entry, daemon=True)
        worker_thread.start()

        while worker_thread.is_alive() or not msg_queue.empty():
            try:
                msg_type, payload = msg_queue.get(timeout=0.1)
                if msg_type == "LOG":
                    level, text = payload
                    broadcast_sse("log", {"level": level.lower(), "message": text})
                elif msg_type == "PROGRESS":
                    current, total, status = payload
                    broadcast_sse("progress", {"current": current, "total": total, "status": status})
                elif msg_type == "DONE":
                    stats, success, log_file, checkpoint_file = payload
                    broadcast_sse("done", {
                        "stats": stats or {},
                        "dry_run": self.config.get("dry_run", False),
                        "log_file": str(log_file),
                    })
            except queue.Empty:
                pass


# Native Windows Directory Picker Helper
def open_native_directory_picker(title: str = "Select Folder") -> str:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder_selected = filedialog.askdirectory(title=title)
    root.destroy()
    return folder_selected or ""


# Custom HTTP Request Handler
class GlassOrganizerHTTPHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        web_dir = Path(__file__).parent / "web"
        super().__init__(*args, directory=str(web_dir), **kwargs)

    def do_GET(self):
        if self.path == "/api/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            q = queue.Queue()
            with subscriber_lock:
                sse_subscribers.add(q)

            try:
                # Send initial compute device info
                _, dev_str = get_torch_device()
                initial_msg = f"event: hardware\ndata: {json.dumps({'device': dev_str})}\n\n"
                self.wfile.write(initial_msg.encode("utf-8"))
                self.wfile.flush()

                while True:
                    data = q.get()
                    self.wfile.write(data.encode("utf-8"))
                    self.wfile.flush()
            except Exception:
                pass
            finally:
                with subscriber_lock:
                    sse_subscribers.discard(q)
            return
        return super().do_GET()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else ""
        body = json.loads(post_data) if post_data else {}

        if self.path == "/api/browse-src":
            selected = open_native_directory_picker("Select Source Directory")
            self.send_json_response({"path": selected})
            return

        elif self.path == "/api/browse-dest":
            selected = open_native_directory_picker("Select Destination Directory")
            self.send_json_response({"path": selected})
            return

        elif self.path == "/api/start":
            global active_worker
            if active_worker and active_worker.is_alive():
                self.send_json_response({"error": "An organization task is already running"}, status=400)
                return
            active_worker = OrganizerWorkerThread(body)
            active_worker.start()
            self.send_json_response({"status": "started"})
            return

        elif self.path == "/api/stop":
            if active_worker and active_worker.is_alive():
                active_worker.cancel_requested = True
                broadcast_sse("log", {"level": "warn", "message": "[WARN] Stop requested. Finishing in-flight tasks..."})
            self.send_json_response({"status": "stopping"})
            return

        self.send_error(404, "Endpoint not found")

    def send_json_response(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))


def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    port = 8000
    server_address = ("127.0.0.1", port)
    httpd = ThreadingHTTPServer(server_address, GlassOrganizerHTTPHandler)

    url = f"http://localhost:{port}"
    print("=" * 60)
    print(f"[SERVER] GLASSMORPHISM MEDIA ORGANIZER STARTED")
    print(f"[SERVER] Server running at: {url}")
    print("=" * 60)

    # Open local browser window automatically
    threading.Thread(target=lambda: (time.sleep(0.8), webbrowser.open(url)), daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()



if __name__ == "__main__":
    main()
