"""
Media Organizer Desktop GUI — AI-powered photo & video classifier with Glassmorphism UI.

Provides a modern, high-contrast Glassmorphism desktop user interface built with Python's native
tkinter/ttk library. Features translucent dark card layouts, electric cyan/purple accents,
background multithreading, real-time logging, and interactive controls.
"""

import os
import csv
import json
import queue
import shutil
import hashlib
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from PIL import Image, ExifTags
from run_worker import run_worker as execute_run_worker, run_unsort_worker as execute_run_unsort_worker

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

# ==========================================
# DEFAULT CATEGORIES (Folder Name -> CLIP Prompt)
# ==========================================
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
def get_torch_device() -> tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), f"CUDA: {torch.cuda.get_device_name(0)}"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Silicon MPS"
    else:
        return torch.device("cpu"), "CPU (No GPU Detected)"


# ==========================================
# CHECKPOINTING & LOGGING
# ==========================================
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
# CUSTOM PASTEL ROUNDED BUTTON WIDGET
# ==========================================
class PastelRoundedButton(tk.Canvas):
    def __init__(self, parent, text: str, command=None, bg_color="#e9d5ff", fg_color="#0f172a",
                 hover_bg="#fbcfe8", active_bg="#fef08a", radius=16, border_color="#1e293b",
                 border_width=2, font=("Segoe UI", 9, "bold"), width=190, height=38, state="normal",
                 parent_bg=None):
        if parent_bg is None:
            try:
                parent_bg = parent["bg"]
            except Exception:
                parent_bg = "#ffffff"
        super().__init__(parent, width=width, height=height, bg=parent_bg, highlightthickness=0, bd=0)
        self.command = command
        self.text_str = text
        self.bg_color = bg_color
        self.hover_bg = hover_bg
        self.active_bg = active_bg
        self.fg_color = fg_color
        self.border_color = border_color
        self.border_width = border_width
        self.radius = radius
        self.font = font
        self.btn_state = state
        self.w = width
        self.h = height

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

        self.draw(self.bg_color if state == "normal" else "#e2e8f0")

    def draw(self, fill_color):
        self.delete("all")
        r = self.radius
        w, h = self.w, self.h
        bw = self.border_width

        if self.btn_state == "disabled":
            fill_c = "#e2e8f0"
            text_c = "#94a3b8"
            border_c = "#cbd5e1"
        else:
            fill_c = fill_color
            text_c = self.fg_color
            border_c = self.border_color

        # Arcs for 4 rounded corners
        self.create_arc(bw, bw, 2*r, 2*r, start=90, extent=90, fill=fill_c, outline=border_c, width=bw)
        self.create_arc(w-2*r, bw, w-bw, 2*r, start=0, extent=90, fill=fill_c, outline=border_c, width=bw)
        self.create_arc(w-2*r, h-2*r, w-bw, h-bw, start=270, extent=90, fill=fill_c, outline=border_c, width=bw)
        self.create_arc(bw, h-2*r, 2*r, h-bw, start=180, extent=90, fill=fill_c, outline=border_c, width=bw)

        # Central rectangles
        self.create_rectangle(r, bw, w-r, h-bw, fill=fill_c, outline="")
        self.create_rectangle(bw, r, w-bw, h-r, fill=fill_c, outline="")

        # Connect outer border lines
        self.create_line(r, bw, w-r, bw, fill=border_c, width=bw)
        self.create_line(r, h-bw, w-r, h-bw, fill=border_c, width=bw)
        self.create_line(bw, r, bw, h-r, fill=border_c, width=bw)
        self.create_line(w-bw, r, w-bw, h-r, fill=border_c, width=bw)

        # Centered label
        self.create_text(w/2, h/2, text=self.text_str, fill=text_c, font=self.font)

    def _on_enter(self, event):
        if self.btn_state == "normal":
            self.draw(self.hover_bg)

    def _on_leave(self, event):
        if self.btn_state == "normal":
            self.draw(self.bg_color)

    def _on_press(self, event):
        if self.btn_state == "normal":
            self.draw(self.active_bg)

    def _on_release(self, event):
        if self.btn_state == "normal":
            self.draw(self.hover_bg)
            if self.command:
                self.command()

    def config_state(self, state):
        self.btn_state = state
        self.draw(self.bg_color if state == "normal" else "#e2e8f0")


# ==========================================
# PASTEL DESKTOP GUI CLASS
# ==========================================
class PastelMediaOrganizerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI Media Organizer — Soft Pastel Edition")
        self.root.geometry("1020x780")
        self.root.minsize(860, 640)

        # Soft Pastel Color Palette
        self.bg_main = "#fcf9f2"         # Warm creamy pastel background
        self.glass_card = "#ffffff"      # Solid crisp white card
        self.border_color = "#1e293b"    # Dark slate border
        self.fg_bright = "#0f172a"       # Deep slate text
        self.fg_muted = "#64748b"        # Muted slate text

        self.pastel_yellow = "#fef08a"   # Soft Butter Yellow
        self.pastel_purple = "#e9d5ff"   # Soft Lavender
        self.pastel_pink = "#fbcfe8"     # Soft Blossom Pink
        self.pastel_blue = "#bae6fd"     # Soft Sky Blue
        self.pastel_mint = "#bbf7d0"     # Soft Mint Green
        self.pastel_red = "#fca5a5"      # Soft Pastel Red

        self.cyan_accent = self.pastel_blue
        self.green_accent = self.pastel_mint
        self.red_accent = self.pastel_red
        self.yellow_accent = self.pastel_yellow
        self.btn_bg = self.pastel_purple
        self.input_bg = "#ffffff"

        # Configure TTK Theme Engine
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.configure_pastel_styles()

        self.msg_queue = queue.Queue()
        self.is_running = False
        self.cancel_requested = False
        self.cancel_event = threading.Event()

        self.create_widgets()
        self.root.after(100, self.process_queue)

    def configure_pastel_styles(self):
        self.root.configure(bg=self.bg_main)
        self.style.configure(".", background=self.bg_main, foreground=self.fg_bright, font=("Segoe UI", 9, "bold"))

        # Soft Pastel Cards & Frames
        self.style.configure("TLabelframe", background=self.glass_card, foreground=self.fg_bright, relief="solid", borderwidth=2)
        self.style.configure("TLabelframe.Label", background=self.glass_card, foreground=self.fg_bright, font=("Segoe UI", 11, "bold"))
        self.style.configure("TFrame", background=self.bg_main)
        self.style.configure("Glass.TFrame", background=self.glass_card)
        self.style.configure("TLabel", background=self.glass_card, foreground=self.fg_bright, font=("Segoe UI", 9, "bold"))
        self.style.configure("TCheckbutton", background=self.glass_card, foreground=self.fg_bright, font=("Segoe UI", 9, "bold"))
        self.style.configure("TRadiobutton", background=self.glass_card, foreground=self.fg_bright, font=("Segoe UI", 9, "bold"))

        # Header Titles
        self.style.configure("Header.TLabel", background=self.bg_main, foreground=self.fg_bright, font=("Segoe UI", 17, "bold"))
        self.style.configure("SubHeader.TLabel", background=self.bg_main, foreground=self.fg_muted, font=("Segoe UI", 9, "bold"))

        # Inputs
        self.style.configure("TEntry", fieldbackground=self.input_bg, foreground=self.fg_bright, insertcolor=self.fg_bright, bordercolor=self.border_color, borderwidth=2, relief="solid")
        self.style.configure("TSpinbox", fieldbackground=self.input_bg, foreground=self.fg_bright, insertcolor=self.fg_bright, arrowcolor=self.fg_bright, borderwidth=2, relief="solid")
        self.style.configure("TCombobox", fieldbackground=self.input_bg, foreground=self.fg_bright, selectbackground=self.pastel_yellow, selectforeground=self.fg_bright, arrowcolor=self.fg_bright, borderwidth=2, relief="solid")

        # Progressbar
        self.style.configure("TProgressbar", thickness=16, troughcolor="#ffffff", background=self.pastel_pink, bordercolor=self.border_color, borderwidth=2)

    def create_widgets(self):
        # Header Banner
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill="x", padx=18, pady=(12, 6))

        title_sub_frame = ttk.Frame(header_frame)
        title_sub_frame.pack(side="left")
        ttk.Label(title_sub_frame, text="⚡ AI MEDIA ORGANIZER", style="Header.TLabel").pack(anchor="w")
        ttk.Label(title_sub_frame, text="Zero-Shot Vision Classifier // OpenAI CLIP Engine", style="SubHeader.TLabel").pack(anchor="w")

        # Hardware Badge
        device_obj, device_str = get_torch_device()
        ttk.Label(header_frame, text=f" ⚡ {device_str.upper()} ", font=("Segoe UI", 9, "bold"), background=self.green_accent, foreground="#000000", relief="solid", borderwidth=2).pack(side="right", anchor="e")

        # Card 1: Directory Selection & Exclusions
        paths_frame = ttk.LabelFrame(self.root, text=" 📂 DIRECTORY SELECTION & EXCLUSIONS ", padding=12)
        paths_frame.pack(fill="x", padx=18, pady=6)

        ttk.Label(paths_frame, text="Source Folder:").grid(row=0, column=0, sticky="w", pady=4)
        self.src_var = tk.StringVar(value="E:/")
        ttk.Entry(paths_frame, textvariable=self.src_var, width=65).grid(row=0, column=1, padx=8, pady=4, sticky="ew")
        PastelRoundedButton(paths_frame, text="Browse...", command=self.browse_src, bg_color=self.pastel_purple, hover_bg=self.pastel_pink, radius=12, width=95, height=30).grid(row=0, column=2, padx=4, pady=4)

        ttk.Label(paths_frame, text="Destination:").grid(row=1, column=0, sticky="w", pady=4)
        self.dest_var = tk.StringVar(value="E:/Organized_Media")
        ttk.Entry(paths_frame, textvariable=self.dest_var, width=65).grid(row=1, column=1, padx=8, pady=4, sticky="ew")
        PastelRoundedButton(paths_frame, text="Browse...", command=self.browse_dest, bg_color=self.pastel_purple, hover_bg=self.pastel_pink, radius=12, width=95, height=30).grid(row=1, column=2, padx=4, pady=4)

        ttk.Label(paths_frame, text="Exclude Folders:").grid(row=2, column=0, sticky="w", pady=4)
        default_excludes_str = ", ".join(sorted(EXCLUDE_FOLDERS))
        self.exclude_var = tk.StringVar(value=default_excludes_str)
        ttk.Entry(paths_frame, textvariable=self.exclude_var, width=65).grid(row=2, column=1, padx=8, pady=4, sticky="ew")
        ttk.Label(paths_frame, text="(comma-separated)", font=("Segoe UI", 8, "italic"), foreground=self.fg_muted).grid(row=2, column=2, sticky="w", pady=4)

        paths_frame.columnconfigure(1, weight=1)

        # Card 2: Organizer Settings
        settings_frame = ttk.LabelFrame(self.root, text=" ⚙️ ORGANIZER & AI SETTINGS ", padding=12)
        settings_frame.pack(fill="x", padx=18, pady=6)

        # Row 0: Action Mode & Confidence
        ttk.Label(settings_frame, text="Action Mode:").grid(row=0, column=0, sticky="w", pady=4)
        self.action_var = tk.StringVar(value="copy")
        ttk.Radiobutton(settings_frame, text="Copy (Safe)", variable=self.action_var, value="copy").grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(settings_frame, text="Move (Verified Hash Delete)", variable=self.action_var, value="move").grid(row=0, column=2, sticky="w")

        ttk.Label(settings_frame, text="Confidence Thresh:").grid(row=0, column=3, sticky="w", padx=(20, 5))
        self.confidence_var = tk.DoubleVar(value=0.35)
        conf_spin = ttk.Spinbox(settings_frame, from_=0.0, to=1.0, increment=0.05, textvariable=self.confidence_var, width=6)
        conf_spin.grid(row=0, column=4, sticky="w")

        # Row 1: Execution Mode & Threads
        self.execute_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings_frame, text="Execute Transfers (Uncheck for Dry Run)", variable=self.execute_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=4)

        ttk.Label(settings_frame, text="Batch Size:").grid(row=1, column=2, sticky="w")
        self.batch_var = tk.IntVar(value=16)
        ttk.Spinbox(settings_frame, from_=1, to=128, increment=4, textvariable=self.batch_var, width=6).grid(row=1, column=3, sticky="w")

        ttk.Label(settings_frame, text="Worker Threads:").grid(row=1, column=4, sticky="w", padx=(10, 5))
        self.threads_var = tk.IntVar(value=4)
        ttk.Spinbox(settings_frame, from_=1, to=16, increment=1, textvariable=self.threads_var, width=6).grid(row=1, column=5, sticky="w")

        # Row 2: Options Checkboxes
        self.classify_videos_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Classify Videos via Frame Extraction", variable=self.classify_videos_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=4)

        self.dedupe_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings_frame, text="Deduplicate (Route exact MD5 matches to Duplicates/)", variable=self.dedupe_var).grid(row=2, column=2, columnspan=2, sticky="w", pady=4)

        self.preserve_folders_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Preserve Subfolders", variable=self.preserve_folders_var).grid(row=2, column=4, columnspan=2, sticky="w", pady=4)

        # Row 3: Model Selector
        ttk.Label(settings_frame, text="CLIP Model:").grid(row=3, column=0, sticky="w", pady=4)
        self.model_var = tk.StringVar(value="openai/clip-vit-large-patch14-336")
        model_combo = ttk.Combobox(settings_frame, textvariable=self.model_var, values=["openai/clip-vit-large-patch14-336", "openai/clip-vit-large-patch14", "openai/clip-vit-base-patch32"], width=35, state="readonly")
        model_combo.grid(row=3, column=1, columnspan=3, sticky="w", pady=4)

        # Control Bar with Custom Rounded Pastel Buttons
        ctrl_frame = tk.Frame(self.root, bg=self.bg_main)
        ctrl_frame.pack(fill="x", padx=18, pady=10)

        self.dry_run_btn = PastelRoundedButton(ctrl_frame, text="🧪 RUN DRY RUN (PREVIEW)", command=lambda: self.start_organization(dry_run_override=True), bg_color=self.pastel_purple, hover_bg=self.pastel_pink, active_bg=self.pastel_yellow, radius=18, width=200, height=42, parent_bg=self.bg_main)
        self.dry_run_btn.pack(side="left", padx=4)

        self.execute_btn = PastelRoundedButton(ctrl_frame, text="⚡ EXECUTE FILE TRANSFERS", command=lambda: self.start_organization(dry_run_override=False), bg_color=self.pastel_mint, hover_bg=self.pastel_yellow, active_bg=self.pastel_blue, radius=18, width=200, height=42, parent_bg=self.bg_main)
        self.execute_btn.pack(side="left", padx=4)

        self.unsort_btn = PastelRoundedButton(ctrl_frame, text="⏮ UNSORT & RESTORE", command=self.start_unsort, bg_color=self.pastel_yellow, hover_bg=self.pastel_pink, active_bg=self.pastel_blue, radius=18, width=170, height=42, parent_bg=self.bg_main)
        self.unsort_btn.pack(side="left", padx=4)

        self.stop_btn = PastelRoundedButton(ctrl_frame, text="⏹ STOP", command=self.stop_organization, bg_color=self.pastel_red, hover_bg="#f87171", active_bg="#ef4444", radius=18, width=95, height=42, state="disabled", parent_bg=self.bg_main)
        self.stop_btn.pack(side="left", padx=4)

        # Progress & Console Log Panel
        progress_frame = ttk.LabelFrame(self.root, text=" 📊 PROGRESS & ACTIVITY CONSOLE ", padding=12)
        progress_frame.pack(fill="both", expand=True, padx=18, pady=(4, 12))

        self.pbar = ttk.Progressbar(progress_frame, mode="determinate")
        self.pbar.pack(fill="x", pady=4)

        self.status_lbl = ttk.Label(progress_frame, text="SYSTEM READY. SELECT SOURCE AND DESTINATION FOLDERS TO BEGIN.", font=("Segoe UI", 9, "bold"), foreground=self.fg_muted)
        self.status_lbl.pack(anchor="w", pady=2)

        # High-Tech Soft Slate Terminal Console Log
        self.log_text = scrolledtext.ScrolledText(progress_frame, height=12, bg="#0f172a", fg="#86efac", font=("Consolas", 9, "bold"), insertbackground=self.cyan_accent, relief="solid", borderwidth=2)
        self.log_text.pack(fill="both", expand=True, pady=5)
        self.log_text.tag_config("OK", foreground=self.green_accent)
        self.log_text.tag_config("ERROR", foreground=self.red_accent)
        self.log_text.tag_config("WARN", foreground=self.yellow_accent)
        self.log_text.tag_config("INFO", foreground=self.cyan_accent)
        self.log_text.tag_config("PURPLE", foreground=self.pastel_purple)

    def browse_src(self):
        d = filedialog.askdirectory(initialdir=self.src_var.get())
        if d:
            self.src_var.set(d)

    def browse_dest(self):
        d = filedialog.askdirectory(initialdir=self.dest_var.get())
        if d:
            self.dest_var.set(d)

    def append_log(self, text: str, tag: str = "INFO"):
        self.log_text.insert(tk.END, text + "\n", tag)
        self.log_text.see(tk.END)

    def process_queue(self):
        while not self.msg_queue.empty():
            msg_type, content = self.msg_queue.get()
            if msg_type == "LOG":
                tag, text = content
                self.append_log(text, tag)
            elif msg_type == "PROGRESS":
                current, total, status = content
                if total > 0:
                    self.pbar["maximum"] = total
                    self.pbar["value"] = current
                self.status_lbl.config(text=status)
            elif msg_type == "DONE":
                stats, dry_run, log_file, checkpoint_file = content
                self.is_running = False
                self.dry_run_btn.config_state("normal")
                self.execute_btn.config_state("normal")
                self.unsort_btn.config_state("normal")
                self.stop_btn.config_state("disabled")
                self.append_log("\n=== PROCESS COMPLETED ===", "PURPLE")
                if stats:
                    self.show_summary_dialog(stats, dry_run, log_file, checkpoint_file)
        self.root.after(100, self.process_queue)

    def start_organization(self, dry_run_override: bool = None):
        src = Path(self.src_var.get())
        dest = Path(self.dest_var.get())

        if not src.exists():
            messagebox.showerror("Error", f"Source folder does not exist:\n{src}")
            return
        if not dest.exists():
            try:
                dest.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                messagebox.showerror("Error", f"Could not create destination folder:\n{e}")
                return

        self.is_running = True
        self.cancel_requested = False
        self.cancel_event.clear()
        self.dry_run_btn.config_state("disabled")
        self.execute_btn.config_state("disabled")
        self.unsort_btn.config_state("disabled")
        self.stop_btn.config_state("normal")
        self.log_text.delete("1.0", tk.END)

        is_dry_run = dry_run_override if dry_run_override is not None else (not self.execute_var.get())

        config = {
            "src": src,
            "dest": dest,
            "action": self.action_var.get(),
            "dry_run": is_dry_run,
            "confidence": self.confidence_var.get(),
            "batch_size": self.batch_var.get(),
            "threads": self.threads_var.get(),
            "classify_videos": self.classify_videos_var.get(),
            "dedupe": self.dedupe_var.get(),
            "preserve_folders": self.preserve_folders_var.get(),
            "model_name": self.model_var.get(),
            "exclude_folders": self.exclude_var.get(),
        }

        threading.Thread(target=self.run_worker, args=(config,), daemon=True).start()

    def start_unsort(self):
        dest = Path(self.dest_var.get())
        log_file = dest / "_organizer_log.csv"

        if not log_file.exists():
            messagebox.showerror("Error", f"No transfer log (_organizer_log.csv) found in destination folder:\n{dest}\n\nCannot unsort without log file.")
            return

        confirm = messagebox.askyesno("Confirm Unsort & Restore", f"Are you sure you want to restore all organized files in:\n{dest}\n\nback to their original source folders?")
        if not confirm:
            return

        self.is_running = True
        self.cancel_requested = False
        self.dry_run_btn.config_state("disabled")
        self.execute_btn.config_state("disabled")
        self.unsort_btn.config_state("disabled")
        self.stop_btn.config_state("normal")
        self.log_text.delete("1.0", tk.END)

        threading.Thread(target=self.run_unsort_worker, args=(dest,), daemon=True).start()

    def run_unsort_worker(self, dest: Path):
        execute_run_unsort_worker(dest_dir=dest, dry_run=False, msg_queue=self.msg_queue, cancel_flag=self.cancel_event)

    def stop_organization(self):
        if self.is_running:
            self.cancel_requested = True
            self.cancel_event.set()
            self.append_log("[WARN] Stop requested. Finishing in-flight tasks...", "WARN")

    def run_worker(self, config: dict):
        execute_run_worker(config, msg_queue=self.msg_queue, cancel_flag=self.cancel_event)

    def show_summary_dialog(self, stats: dict, dry_run: bool, log_file: Path, checkpoint_file: Path):
        summary_text = (
            f"Mode: {'DRY RUN (Preview)' if dry_run else 'EXECUTED (' + stats['action'].upper() + ')'}\n\n"
            f"Total Candidate Files: {stats['total_discovered']}\n"
            f"Processed Photos:     {stats['photos_count']}\n"
            f"Processed Videos:     {stats['videos_count']}\n"
            f"Skipped (Done):       {stats['skipped_count']}\n"
            f"Duplicates Identified:{stats['duplicates_count']}\n"
            f"Errors:               {stats['error_count']}\n\n"
            f"Category Breakdown:\n"
        )
        for cat, cnt in sorted(stats["categories"].items()):
            summary_text += f"  • {cat}: {cnt} files\n"

        summary_text += f"\nFull Log Saved To:\n{log_file}"
        messagebox.showinfo("Organization Complete", summary_text)


# ==========================================
# ENTRY POINT
# ==========================================
def main():
    root = tk.Tk()
    app = PastelMediaOrganizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
