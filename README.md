# ⚡ AI Media Organizer

An AI-powered photo and video classifier that automatically sorts media into categorized subfolders using **OpenAI's CLIP Zero-Shot Vision Model**, EXIF date extraction, multithreaded decoding, and optional MD5 duplicate detection.

Supports **Desktop GUI**, **Web Application**, and **CLI Interface**.

---

## 🚀 Quick Start

### 1. Install Prerequisites

#### Python Packages
Install the required dependencies via `pip`:

```bash
pip install torch transformers pillow pillow-heif rawpy tqdm
```

#### FFmpeg Setup (Required for Video Classification)
Ensure `ffmpeg` and `ffprobe` are installed and available on your system `PATH`:
- **Windows**: Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) or install via winget:
  ```powershell
  winget install Gyan.FFmpeg
  ```
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

---

## 🖥️ Running the Application

### Option A: Desktop GUI (Tkinter)
Launches the modern desktop interface:

```powershell
python organize_media_gui.py
```

### Option B: Web Application (Browser Interface)
Launches the local Glassmorphism Web App server and opens it in your default web browser:

```powershell
python organize_media_server.py
```
> Access in browser at: `http://localhost:8000`

### Option C: CLI Interface (Terminal)
Run media organization directly from your terminal:

```powershell
# Copy Mode (Safe preview dry-run)
python run_worker.py --src "E:\Photos" --dest "E:\Organized_Media" --dry-run

# Real Execute Move Mode
python run_worker.py --src "E:\Photos" --dest "E:\Organized_Media" --action move

# Unsort / Restore Files back to Original Paths
python run_worker.py --dest "E:\Organized_Media" --unsort
```

---

## ⚙️ CLI Options & Parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--src` | *Required (for sort)* | Path to source directory containing photos/videos. |
| `--dest` | *Required* | Path to destination directory for organized folders. |
| `--action` | `copy` | File operation mode: `copy`, `move`, or `unsort`. |
| `--unsort` | `False` | Restore organized files back to their original source folders. |
| `--dry-run` | `False` | Perform preview run without copying or deleting files. |
| `--model` | `openai/clip-vit-base-patch32` | HuggingFace CLIP vision model name. |
| `--batch-size` | `32` | Number of image frames processed in parallel per GPU/CPU batch. |
| `--confidence` | `0.35` | Minimum confidence score threshold (lower confidence goes to `Low_Confidence`). |
| `--threads` | `4` | Number of worker threads for parallel image and frame decoding. |
| `--classify-videos` | `True` | Extract video frame at 10% duration for AI classification. |
| `--dedupe` | `False` | Calculate MD5 hashes to route duplicates to `Duplicates/`. |
| `--preserve-folders` | `False` | Retain original subfolder directory hierarchy inside destination. |
| `--exclude` | `""` | Comma-separated list of folder names to skip during scanning. |

---

## 🛠️ Debugging & Diagnostic Commands

If you encounter issues, run these diagnostic commands in your terminal:

### 1. Verify Python Code Compilation
Check all core Python scripts for syntax errors:
```powershell
python -m py_compile organize_media.py run_worker.py organize_media_gui.py organize_media_server.py
```

### 2. Verify GPU Acceleration (CUDA / PyTorch)
Test if PyTorch recognizes your NVIDIA GPU:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 3. Verify FFmpeg & FFprobe Installation
Check if `ffmpeg` and `ffprobe` are correctly installed on your system `PATH`:
```powershell
ffmpeg -version
ffprobe -version
```

### 4. Check Installed Python Library Support
Verify image decoders (HEIC/RAW) and CLIP dependencies:
```powershell
python -c "import PIL, transformers, torch; print('Pillow OK'); print('Transformers OK'); import pillow_heif; print('HEIC Support OK')"
```

### 5. Log & Checkpoint Recovery
- **Transfer Log**: Every run appends entries to `_organizer_log.csv` inside your destination folder.
- **Resumable Checkpoints**: Progress is tracked in `_organizer_checkpoint.json`. If a job is interrupted, running the tool again automatically resumes where it left off.
- **Resetting Checkpoints**: To force re-processing of a destination folder, delete `_organizer_checkpoint.json`.

---

## 📂 Project File Architecture

- `organize_media.py`: Core processing engine (CLIP inference, image decoders, EXIF dates, transfers, restore logic).
- `run_worker.py`: Unified background worker and CLI entry point for sorting and unsorting.
- `organize_media_gui.py`: Desktop GUI built with Python Tkinter.
- `organize_media_server.py`: Web server backend providing SSE streaming and API endpoints.
- `web/`: Frontend interface for the Web App version (HTML, CSS, JS).
