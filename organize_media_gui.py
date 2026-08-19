"""
AI Media Organizer — Neural Vision Command Center (Smooth Maximalist Edition)

A state-of-the-art, high-density, smooth maximalist desktop GUI built with CustomTkinter.
Features antialiased geometry, Windows High-DPI awareness, live hardware telemetry HUD,
real-time KPI metric cards, interactive exclusion tag cloud, neural category matrix with
live file counters, multi-theme engine, and high-performance async processing pipeline.
"""

import os
import sys
import csv
import json
import time
import queue
import shutil
import hashlib
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Enable Windows Per-Monitor High-DPI Scaling for ultra-crisp rendering
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import customtkinter as ctk
from PIL import Image, ExifTags

from run_worker import run_worker as execute_run_worker, run_unsort_worker as execute_run_unsort_worker
from organize_media import (
    DEFAULT_CATEGORIES,
    LEGACY_CATEGORY_MAP,
    normalize_category_name,
    EXCLUDE_FOLDERS,
    SYSTEM_EXCLUDES,
    IMAGE_EXTS,
    VIDEO_EXTS,
    RAW_EXTS,
    HEIF_EXTS,
    get_torch_device as organize_get_torch_device,
)

# Pillow decompression bomb protection override for gigapixel panoramas
Image.MAX_IMAGE_PIXELS = None

# Hardware Acceleration Detector
def get_hardware_info() -> tuple[str, str, bool]:
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return "CUDA ACTIVE", f"{device_name} ({vram_gb:.1f} GB VRAM)", True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "APPLE MPS", "Apple Silicon Neural Engine", True
        else:
            return "CPU MODE", "Host CPU (No GPU Acceleration)", False
    except Exception:
        return "CPU MODE", "Standard Host CPU", False


# Category Icon & Color Metadata for Maximalist Category Matrix (Exact 1:1 match with Excel Log & Folders)
CATEGORY_METADATA = {
    "Flora & Plants": {"icon": "🌸", "label": "Flora & Plants", "color": "#10B981"},
    "Vehicles & Aviation": {"icon": "✈️", "label": "Vehicles & Aviation", "color": "#3B82F6"},
    "Animals & Wildlife": {"icon": "🐾", "label": "Animals & Wildlife", "color": "#F59E0B"},
    "Birds & Avian": {"icon": "🐦", "label": "Birds & Avian", "color": "#06B6D4"},
    "Portraits & People": {"icon": "👤", "label": "Portraits & People", "color": "#EC4899"},
    "Landscapes & Nature": {"icon": "🏔️", "label": "Landscapes & Nature", "color": "#14B8A6"},
    "Street & Urban": {"icon": "🏙️", "label": "Street & Urban", "color": "#8B5CF6"},
    "Architecture": {"icon": "🏛️", "label": "Architecture", "color": "#6366F1"},
    "Food & Dining": {"icon": "🍲", "label": "Food & Dining", "color": "#F97316"},
    "Docs & Receipts": {"icon": "📄", "label": "Docs & Receipts", "color": "#64748B"},
    "Events & Parties": {"icon": "🎉", "label": "Events & Parties", "color": "#D946EF"},
}

# Theme Definitions for Dynamic Theme Engine
THEMES = {
    "Obsidian Glow (Dark)": {
        "mode": "dark",
        "bg_main": "#080B11",
        "card_bg": "#0F172A",
        "card_sub_bg": "#1E293B",
        "border_color": "#1E293B",
        "border_sub": "#334155",
        "text_bright": "#F8FAFC",
        "text_muted": "#94A3B8",
        "accent_cyan": "#06B6D4",
        "accent_purple": "#8B5CF6",
        "accent_green": "#10B981",
        "accent_amber": "#F59E0B",
        "accent_red": "#EF4444",
        "btn_dryrun": ("#7C3AED", "#6D28D9"),
        "btn_execute": ("#059669", "#047857"),
        "btn_unsort": ("#D97706", "#B45309"),
        "btn_stop": ("#DC2626", "#B91C1C"),
        "console_bg": "#020617",
        "console_fg": "#38BDF8",
    },
    "Cyberpunk Neon (Dark)": {
        "mode": "dark",
        "bg_main": "#050508",
        "card_bg": "#121217",
        "card_sub_bg": "#1E1E24",
        "border_color": "#27272A",
        "border_sub": "#3F3F46",
        "text_bright": "#FEF08A",
        "text_muted": "#A1A1AA",
        "accent_cyan": "#22D3EE",
        "accent_purple": "#EC4899",
        "accent_green": "#4ADE80",
        "accent_amber": "#FACC15",
        "accent_red": "#F43F5E",
        "btn_dryrun": ("#EC4899", "#BE185D"),
        "btn_execute": ("#22C55E", "#15803D"),
        "btn_unsort": ("#EAB308", "#CA8A04"),
        "btn_stop": ("#F43F5E", "#E11D48"),
        "console_bg": "#09090B",
        "console_fg": "#4ADE80",
    },
    "Synthwave Sunset (Dark)": {
        "mode": "dark",
        "bg_main": "#0D091A",
        "card_bg": "#17122E",
        "card_sub_bg": "#251D4A",
        "border_color": "#332663",
        "border_sub": "#4A398A",
        "text_bright": "#FDF4FF",
        "text_muted": "#C084FC",
        "accent_cyan": "#38BDF8",
        "accent_purple": "#D946EF",
        "accent_green": "#34D399",
        "accent_amber": "#FB923C",
        "accent_red": "#F43F5E",
        "btn_dryrun": ("#C026D3", "#A21CAF"),
        "btn_execute": ("#2563EB", "#1D4ED8"),
        "btn_unsort": ("#EA580C", "#C2410C"),
        "btn_stop": ("#E11D48", "#BE123C"),
        "console_bg": "#090614",
        "console_fg": "#F472B6",
    },
    "Pastel Maximalist (Light)": {
        "mode": "light",
        "bg_main": "#F8FAFC",
        "card_bg": "#FFFFFF",
        "card_sub_bg": "#F1F5F9",
        "border_color": "#E2E8F0",
        "border_sub": "#CBD5E1",
        "text_bright": "#0F172A",
        "text_muted": "#64748B",
        "accent_cyan": "#0284C7",
        "accent_purple": "#7C3AED",
        "accent_green": "#059669",
        "accent_amber": "#D97706",
        "accent_red": "#DC2626",
        "btn_dryrun": ("#8B5CF6", "#7C3AED"),
        "btn_execute": ("#10B981", "#059669"),
        "btn_unsort": ("#F59E0B", "#D97706"),
        "btn_stop": ("#EF4444", "#DC2626"),
        "console_bg": "#0F172A",
        "console_fg": "#86EFAC",
    },
}


class MaximalistMediaOrganizerApp:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("AI Media Organizer — Neural Vision Command Center")
        self.root.geometry("1200x880")
        self.root.minsize(1020, 720)

        # Active Theme
        self.current_theme_name = "Cyberpunk Neon (Dark)"
        self.theme = THEMES[self.current_theme_name]
        ctk.set_appearance_mode(self.theme["mode"])
        ctk.set_default_color_theme("blue")

        # Runtime State
        self.msg_queue = queue.Queue()
        self.is_running = False
        self.cancel_event = threading.Event()
        self.start_time = 0.0
        self.session_timer_id = None
        self.current_excludes = list(sorted(EXCLUDE_FOLDERS))
        self.stats = {
            "discovered": 0,
            "photos": 0,
            "videos": 0,
            "duplicates": 0,
            "errors": 0,
            "categories": {cat: 0 for cat in DEFAULT_CATEGORIES},
        }

        # UI Element References for Live Updates
        self.category_count_labels = {}
        self.category_chip_frames = {}

        self.create_maximalist_ui()
        self.root.after(100, self.process_queue)

    def create_maximalist_ui(self):
        self.root.configure(fg_color=self.theme["bg_main"])

        # Main Scrollable Container
        self.main_scroll = ctk.CTkScrollableFrame(
            self.root,
            fg_color=self.theme["bg_main"],
            corner_radius=0,
            scrollbar_button_color=self.theme["border_sub"],
            scrollbar_button_hover_color=self.theme["accent_purple"],
        )
        self.main_scroll.pack(fill="both", expand=True, padx=0, pady=0)

        # 1. Header Command Deck
        self.build_header_deck()

        # 2. Live Telemetry Metric KPI Cards
        self.build_kpi_dashboard()

        # 3. Settings Deck (Directories & Neural Parameters)
        self.build_settings_deck()

        # 4. Neural Classification Category Matrix
        self.build_category_matrix()

        # 5. Glowing Command Action Bar
        self.build_action_bar()

        # 6. Neural Activity Console & Live Stream
        self.build_console_deck()

    # ==========================================
    # 1. HEADER COMMAND DECK
    # ==========================================
    def build_header_deck(self):
        header_card = ctk.CTkFrame(
            self.main_scroll,
            fg_color=self.theme["card_bg"],
            corner_radius=14,
            border_width=1,
            border_color=self.theme["border_color"],
        )
        header_card.pack(fill="x", padx=14, pady=(8, 4))

        header_layout = ctk.CTkFrame(header_card, fg_color="transparent")
        header_layout.pack(fill="x", padx=14, pady=8)

        # Left: Branding & Subtitle
        title_box = ctk.CTkFrame(header_layout, fg_color="transparent")
        title_box.pack(side="left", anchor="w")

        title_row = ctk.CTkFrame(title_box, fg_color="transparent")
        title_row.pack(anchor="w")

        ctk.CTkLabel(
            title_row,
            text="⚡",
            font=ctk.CTkFont(size=20),
            text_color=self.theme["accent_cyan"],
        ).pack(side="left", padx=(0, 5))

        ctk.CTkLabel(
            title_row,
            text="AI MEDIA ORGANIZER",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color=self.theme["text_bright"],
        ).pack(side="left")

        ctk.CTkLabel(
            title_row,
            text=" v2.5 MAXIMALIST",
            font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
            text_color=self.theme["accent_purple"],
            fg_color=self.theme["card_sub_bg"],
            corner_radius=6,
            padx=5,
            pady=1,
        ).pack(side="left", padx=(6, 0))

        ctk.CTkLabel(
            title_box,
            text="Zero-Shot Vision Classifier // OpenAI CLIP Multi-Modal Neural Engine",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=self.theme["text_muted"],
        ).pack(anchor="w", pady=(1, 0))

        # Right: Hardware Telemetry HUD & Theme Controls
        hud_box = ctk.CTkFrame(header_layout, fg_color="transparent")
        hud_box.pack(side="right", anchor="e")

        # Hardware Badge Pill
        hw_tag, hw_desc, is_gpu = get_hardware_info()
        hw_pill = ctk.CTkFrame(
            hud_box,
            fg_color=self.theme["card_sub_bg"],
            corner_radius=10,
            border_width=1,
            border_color=self.theme["accent_green"] if is_gpu else self.theme["border_sub"],
        )
        hw_pill.pack(side="left", padx=(0, 8))

        hw_dot = "🟢" if is_gpu else "⚪"
        ctk.CTkLabel(
            hw_pill,
            text=f"{hw_dot} {hw_tag}",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color=self.theme["accent_green"] if is_gpu else self.theme["text_muted"],
        ).pack(side="left", padx=(8, 4), pady=4)

        ctk.CTkLabel(
            hw_pill,
            text=f"| {hw_desc}",
            font=ctk.CTkFont(family="Segoe UI", size=9),
            text_color=self.theme["text_muted"],
        ).pack(side="left", padx=(0, 8), pady=4)

        # Live Session Stopwatch
        self.timer_pill = ctk.CTkFrame(
            hud_box,
            fg_color=self.theme["card_sub_bg"],
            corner_radius=10,
            border_width=1,
            border_color=self.theme["border_color"],
        )
        self.timer_pill.pack(side="left", padx=(0, 8))

        self.timer_lbl = ctk.CTkLabel(
            self.timer_pill,
            text="⏱️ 00:00:00",
            font=ctk.CTkFont(family="Consolas", size=10, weight="bold"),
            text_color=self.theme["accent_cyan"],
        )
        self.timer_lbl.pack(padx=8, pady=4)

        # Theme Selector
        theme_box = ctk.CTkFrame(hud_box, fg_color="transparent")
        theme_box.pack(side="left")

        ctk.CTkLabel(
            theme_box,
            text="🎨 Theme:",
            font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
            text_color=self.theme["text_muted"],
        ).pack(side="left", padx=(0, 3))

        self.theme_menu = ctk.CTkOptionMenu(
            theme_box,
            values=list(THEMES.keys()),
            command=self.change_theme,
            font=ctk.CTkFont(family="Segoe UI", size=10),
            width=165,
            height=26,
            corner_radius=8,
            fg_color=self.theme["card_sub_bg"],
            button_color=self.theme["border_sub"],
            button_hover_color=self.theme["accent_purple"],
            text_color=self.theme["text_bright"],
        )
        self.theme_menu.set(self.current_theme_name)
        self.theme_menu.pack(side="left")

    # ==========================================
    # 2. LIVE TELEMETRY KPI CARDS BAR
    # ==========================================
    def build_kpi_dashboard(self):
        kpi_container = ctk.CTkFrame(self.main_scroll, fg_color="transparent")
        kpi_container.pack(fill="x", padx=14, pady=3)

        kpis = [
            ("📁 DISCOVERED", "0", "kpi_discovered", self.theme["accent_cyan"], "Total media queued"),
            ("🖼️ PHOTOS", "0", "kpi_photos", self.theme["accent_green"], "Classified & copied"),
            ("🎬 VIDEOS", "0", "kpi_videos", self.theme["accent_purple"], "Keyframe analyzed"),
            ("⚡ SPEED", "0.0/s", "kpi_speed", self.theme["accent_amber"], "Active throughput"),
            ("🧬 DUPLICATES", "0", "kpi_duplicates", "#EC4899", "Exact hash matches"),
            ("🎯 ACCURACY", "--", "kpi_confidence", "#38BDF8", "Match confidence"),
        ]

        self.kpi_labels = {}

        for i, (title, val, key, accent, subtitle) in enumerate(kpis):
            card = ctk.CTkFrame(
                kpi_container,
                fg_color=self.theme["card_bg"],
                corner_radius=12,
                border_width=1,
                border_color=self.theme["border_color"],
            )
            card.pack(side="left", fill="both", expand=True, padx=(0 if i == 0 else 5, 0))

            card_inner = ctk.CTkFrame(card, fg_color="transparent")
            card_inner.pack(fill="both", expand=True, padx=10, pady=7)

            ctk.CTkLabel(
                card_inner,
                text=title,
                font=ctk.CTkFont(family="Segoe UI", size=8, weight="bold"),
                text_color=self.theme["text_muted"],
            ).pack(anchor="w")

            val_lbl = ctk.CTkLabel(
                card_inner,
                text=val,
                font=ctk.CTkFont(family="Segoe UI", size=17, weight="bold"),
                text_color=accent,
            )
            val_lbl.pack(anchor="w", pady=(0, 0))
            self.kpi_labels[key] = val_lbl

            ctk.CTkLabel(
                card_inner,
                text=subtitle,
                font=ctk.CTkFont(family="Segoe UI", size=7),
                text_color=self.theme["text_muted"],
            ).pack(anchor="w")

    # ==========================================
    # 3. SETTINGS & PATHS DECK
    # ==========================================
    def build_settings_deck(self):
        deck_container = ctk.CTkFrame(self.main_scroll, fg_color="transparent")
        deck_container.pack(fill="x", padx=14, pady=2)

        # Left: Directory & Exclusion Explorer
        dir_card = ctk.CTkFrame(
            deck_container,
            fg_color=self.theme["card_bg"],
            corner_radius=14,
            border_width=1,
            border_color=self.theme["border_color"],
        )
        dir_card.pack(side="left", fill="both", expand=True, padx=(0, 4), pady=0)

        dir_inner = ctk.CTkFrame(dir_card, fg_color="transparent")
        dir_inner.pack(fill="x", padx=12, pady=8)

        # Header
        dir_header = ctk.CTkFrame(dir_inner, fg_color="transparent")
        dir_header.pack(fill="x", pady=(0, 3))

        ctk.CTkLabel(
            dir_header,
            text="📂 DIRECTORY SELECTION & EXCLUSIONS",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=self.theme["text_bright"],
        ).pack(side="left")

        # Source Path
        src_row = ctk.CTkFrame(dir_inner, fg_color="transparent")
        src_row.pack(fill="x", pady=(0, 3))

        ctk.CTkLabel(
            src_row,
            text="Source:",
            font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
            text_color=self.theme["text_muted"],
            width=50,
            anchor="w",
        ).pack(side="left")

        self.src_var = tk.StringVar(value="E:/")
        self.src_entry = ctk.CTkEntry(
            src_row,
            textvariable=self.src_var,
            font=ctk.CTkFont(family="Segoe UI", size=10),
            corner_radius=8,
            fg_color=self.theme["card_sub_bg"],
            border_color=self.theme["border_color"],
            text_color=self.theme["text_bright"],
            height=26,
        )
        self.src_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))

        ctk.CTkButton(
            src_row,
            text="Browse...",
            command=self.browse_src,
            width=70,
            height=26,
            corner_radius=8,
            fg_color=self.theme["card_sub_bg"],
            hover_color=self.theme["border_sub"],
            border_width=1,
            border_color=self.theme["border_color"],
            text_color=self.theme["text_bright"],
            font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
        ).pack(side="left")

        # Destination Path
        dest_row = ctk.CTkFrame(dir_inner, fg_color="transparent")
        dest_row.pack(fill="x", pady=(0, 3))

        ctk.CTkLabel(
            dest_row,
            text="Target:",
            font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
            text_color=self.theme["text_muted"],
            width=50,
            anchor="w",
        ).pack(side="left")

        self.dest_var = tk.StringVar(value="E:/Organized_Media")
        self.dest_entry = ctk.CTkEntry(
            dest_row,
            textvariable=self.dest_var,
            font=ctk.CTkFont(family="Segoe UI", size=10),
            corner_radius=8,
            fg_color=self.theme["card_sub_bg"],
            border_color=self.theme["border_color"],
            text_color=self.theme["text_bright"],
            height=26,
        )
        self.dest_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))

        ctk.CTkButton(
            dest_row,
            text="Browse...",
            command=self.browse_dest,
            width=70,
            height=26,
            corner_radius=8,
            fg_color=self.theme["card_sub_bg"],
            hover_color=self.theme["border_sub"],
            border_width=1,
            border_color=self.theme["border_color"],
            text_color=self.theme["text_bright"],
            font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
        ).pack(side="left", padx=(0, 3))

        ctk.CTkButton(
            dest_row,
            text="📂",
            command=self.open_dest_in_explorer,
            width=28,
            height=26,
            corner_radius=8,
            fg_color=self.theme["card_sub_bg"],
            hover_color=self.theme["border_sub"],
            border_width=1,
            border_color=self.theme["border_color"],
            text_color=self.theme["accent_cyan"],
            font=ctk.CTkFont(size=11),
        ).pack(side="left")

        # Excluded Folders Header Line with Add & Reset
        excl_header = ctk.CTkFrame(dir_inner, fg_color="transparent")
        excl_header.pack(fill="x", pady=(2, 2))

        ctk.CTkLabel(
            excl_header,
            text="Excluded Folders:",
            font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
            text_color=self.theme["text_muted"],
        ).pack(side="left", padx=(0, 4))

        self.new_tag_var = tk.StringVar()
        self.new_tag_entry = ctk.CTkEntry(
            excl_header,
            textvariable=self.new_tag_var,
            placeholder_text="Add folder...",
            font=ctk.CTkFont(family="Segoe UI", size=9),
            corner_radius=6,
            fg_color=self.theme["card_sub_bg"],
            border_color=self.theme["border_color"],
            text_color=self.theme["text_bright"],
            height=24,
            width=110,
        )
        self.new_tag_entry.pack(side="left", padx=(0, 3))
        self.new_tag_entry.bind("<Return>", lambda e: self.add_exclusion_tag())

        ctk.CTkButton(
            excl_header,
            text="+ Add",
            command=self.add_exclusion_tag,
            width=48,
            height=24,
            corner_radius=6,
            fg_color=self.theme["accent_purple"],
            hover_color="#7C3AED",
            text_color="#FFFFFF",
            font=ctk.CTkFont(family="Segoe UI", size=8, weight="bold"),
        ).pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            excl_header,
            text="Reset",
            command=self.reset_exclusions,
            font=ctk.CTkFont(family="Segoe UI", size=8),
            height=24,
            width=44,
            corner_radius=6,
            fg_color=self.theme["card_sub_bg"],
            text_color=self.theme["accent_cyan"],
            hover_color=self.theme["border_sub"],
        ).pack(side="right")

        # Single Row Horizontal Scrollable Tag Strip
        self.tags_frame = ctk.CTkScrollableFrame(
            dir_inner,
            orientation="horizontal",
            height=28,
            fg_color=self.theme["card_sub_bg"],
            corner_radius=6,
            border_width=1,
            border_color=self.theme["border_color"],
            scrollbar_button_color=self.theme["border_sub"],
            scrollbar_button_hover_color=self.theme["accent_purple"],
        )
        self.tags_frame.pack(fill="x", pady=(0, 0))

        self.render_exclusion_tags()

        # Right: Vision AI & Neural Engine Parameters
        ai_card = ctk.CTkFrame(
            deck_container,
            fg_color=self.theme["card_bg"],
            corner_radius=14,
            border_width=1,
            border_color=self.theme["border_color"],
        )
        ai_card.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=0)

        ai_inner = ctk.CTkFrame(ai_card, fg_color="transparent")
        ai_inner.pack(fill="x", padx=12, pady=8)

        # Header
        ctk.CTkLabel(
            ai_inner,
            text="⚙️ VISION AI & ENGINE PARAMETERS",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=self.theme["text_bright"],
        ).pack(anchor="w", pady=(0, 3))

        # Row 1: Mode + Model (Side by Side)
        mode_model_row = ctk.CTkFrame(ai_inner, fg_color="transparent")
        mode_model_row.pack(fill="x", pady=(0, 3))

        self.action_segmented = ctk.CTkSegmentedButton(
            mode_model_row,
            values=["🛡️ Copy (Safe)", "⚡ Move (Hash Delete)"],
            font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
            corner_radius=8,
            selected_color=self.theme["accent_purple"],
            selected_hover_color="#7C3AED",
            unselected_color=self.theme["card_sub_bg"],
            unselected_hover_color=self.theme["border_sub"],
            text_color=self.theme["text_bright"],
            height=26,
            width=210,
        )
        self.action_segmented.set("🛡️ Copy (Safe)")
        self.action_segmented.pack(side="left", padx=(0, 6))

        self.model_var = tk.StringVar(value="openai/clip-vit-large-patch14-336")
        self.model_combo = ctk.CTkOptionMenu(
            mode_model_row,
            values=[
                "openai/clip-vit-large-patch14-336",
                "openai/clip-vit-large-patch14",
                "openai/clip-vit-base-patch32",
            ],
            variable=self.model_var,
            font=ctk.CTkFont(family="Segoe UI", size=9),
            corner_radius=8,
            fg_color=self.theme["card_sub_bg"],
            button_color=self.theme["border_sub"],
            button_hover_color=self.theme["accent_purple"],
            text_color=self.theme["text_bright"],
            height=26,
        )
        self.model_combo.pack(side="left", fill="x", expand=True)

        # Row 2: Confidence Threshold
        conf_row = ctk.CTkFrame(ai_inner, fg_color="transparent")
        conf_row.pack(fill="x", pady=(0, 2))

        ctk.CTkLabel(
            conf_row,
            text="Confidence Threshold:",
            font=ctk.CTkFont(family="Segoe UI", size=8, weight="bold"),
            text_color=self.theme["text_muted"],
        ).pack(side="left")

        self.conf_val_lbl = ctk.CTkLabel(
            conf_row,
            text="35%",
            font=ctk.CTkFont(family="Segoe UI", size=8, weight="bold"),
            text_color=self.theme["accent_cyan"],
            fg_color=self.theme["card_sub_bg"],
            corner_radius=4,
            padx=4,
            pady=0,
        )
        self.conf_val_lbl.pack(side="right")

        self.confidence_slider = ctk.CTkSlider(
            ai_inner,
            from_=0.05,
            to=0.95,
            number_of_steps=18,
            command=self._on_conf_change,
            height=11,
            button_color=self.theme["accent_cyan"],
            button_hover_color="#0891B2",
            progress_color=self.theme["accent_cyan"],
            fg_color=self.theme["card_sub_bg"],
        )
        self.confidence_slider.set(0.35)
        self.confidence_slider.pack(fill="x", pady=(0, 3))

        # Row 3: Dual Steppers: Batch & Threads
        sliders_grid = ctk.CTkFrame(ai_inner, fg_color="transparent")
        sliders_grid.pack(fill="x", pady=(0, 3))

        # Batch Size
        b_box = ctk.CTkFrame(sliders_grid, fg_color="transparent")
        b_box.pack(side="left", fill="x", expand=True, padx=(0, 4))

        b_hdr = ctk.CTkFrame(b_box, fg_color="transparent")
        b_hdr.pack(fill="x")
        ctk.CTkLabel(
            b_hdr,
            text="Batch Size:",
            font=ctk.CTkFont(family="Segoe UI", size=8, weight="bold"),
            text_color=self.theme["text_muted"],
        ).pack(side="left")

        self.batch_lbl = ctk.CTkLabel(
            b_hdr,
            text="16",
            font=ctk.CTkFont(family="Segoe UI", size=8, weight="bold"),
            text_color=self.theme["accent_purple"],
        )
        self.batch_lbl.pack(side="right")

        self.batch_slider = ctk.CTkSlider(
            b_box,
            from_=1,
            to=64,
            number_of_steps=63,
            command=lambda v: self.batch_lbl.configure(text=str(int(v))),
            height=11,
            button_color=self.theme["accent_purple"],
            progress_color=self.theme["accent_purple"],
            fg_color=self.theme["card_sub_bg"],
        )
        self.batch_slider.set(16)
        self.batch_slider.pack(fill="x", pady=(1, 0))

        # Threads
        t_box = ctk.CTkFrame(sliders_grid, fg_color="transparent")
        t_box.pack(side="left", fill="x", expand=True, padx=(4, 0))

        t_hdr = ctk.CTkFrame(t_box, fg_color="transparent")
        t_hdr.pack(fill="x")
        ctk.CTkLabel(
            t_hdr,
            text="Worker Threads:",
            font=ctk.CTkFont(family="Segoe UI", size=8, weight="bold"),
            text_color=self.theme["text_muted"],
        ).pack(side="left")

        self.threads_lbl = ctk.CTkLabel(
            t_hdr,
            text="4",
            font=ctk.CTkFont(family="Segoe UI", size=8, weight="bold"),
            text_color=self.theme["accent_green"],
        )
        self.threads_lbl.pack(side="right")

        self.threads_slider = ctk.CTkSlider(
            t_box,
            from_=1,
            to=16,
            number_of_steps=15,
            command=lambda v: self.threads_lbl.configure(text=str(int(v))),
            height=11,
            button_color=self.theme["accent_green"],
            progress_color=self.theme["accent_green"],
            fg_color=self.theme["card_sub_bg"],
        )
        self.threads_slider.set(4)
        self.threads_slider.pack(fill="x", pady=(1, 0))

        # Row 4: Switches
        switches_box = ctk.CTkFrame(ai_inner, fg_color="transparent")
        switches_box.pack(fill="x", pady=(2, 0))

        self.classify_videos_var = tk.BooleanVar(value=True)
        self.sw_video = ctk.CTkSwitch(
            switches_box,
            text="🎥 Videos",
            variable=self.classify_videos_var,
            font=ctk.CTkFont(family="Segoe UI", size=8, weight="bold"),
            text_color=self.theme["text_bright"],
            progress_color=self.theme["accent_cyan"],
        )
        self.sw_video.pack(side="left", padx=(0, 6))

        self.dedupe_var = tk.BooleanVar(value=False)
        self.sw_dedupe = ctk.CTkSwitch(
            switches_box,
            text="🧬 Dedupe",
            variable=self.dedupe_var,
            font=ctk.CTkFont(family="Segoe UI", size=8, weight="bold"),
            text_color=self.theme["text_bright"],
            progress_color=self.theme["accent_purple"],
        )
        self.sw_dedupe.pack(side="left", padx=(0, 6))

        self.preserve_folders_var = tk.BooleanVar(value=True)
        self.sw_preserve = ctk.CTkSwitch(
            switches_box,
            text="📁 Subfolders",
            variable=self.preserve_folders_var,
            font=ctk.CTkFont(family="Segoe UI", size=8, weight="bold"),
            text_color=self.theme["text_bright"],
            progress_color=self.theme["accent_green"],
        )
        self.sw_preserve.pack(side="left")

    def _on_conf_change(self, val):
        self.conf_val_lbl.configure(text=f"{int(val * 100)}%")

    # ==========================================
    # 4. NEURAL CATEGORY MATRIX CHIPS
    # ==========================================
    def build_category_matrix(self):
        cat_card = ctk.CTkFrame(
            self.main_scroll,
            fg_color=self.theme["card_bg"],
            corner_radius=14,
            border_width=1,
            border_color=self.theme["border_color"],
        )
        cat_card.pack(fill="x", padx=14, pady=2)

        cat_inner = ctk.CTkFrame(cat_card, fg_color="transparent")
        cat_inner.pack(fill="x", padx=12, pady=6)

        cat_hdr = ctk.CTkFrame(cat_inner, fg_color="transparent")
        cat_hdr.pack(fill="x", pady=(0, 5))

        ctk.CTkLabel(
            cat_hdr,
            text="🏷️ NEURAL CLASSIFICATION MATRIX & LIVE DISPATCH COUNTERS",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color=self.theme["text_bright"],
        ).pack(side="left")

        ctk.CTkLabel(
            cat_hdr,
            text="11 Active Classes",
            font=ctk.CTkFont(family="Segoe UI", size=8, weight="bold"),
            text_color=self.theme["accent_cyan"],
            fg_color=self.theme["card_sub_bg"],
            corner_radius=5,
            padx=5,
            pady=1,
        ).pack(side="right")

        chips_frame = ctk.CTkFrame(cat_inner, fg_color="transparent")
        chips_frame.pack(fill="x")

        cols = 4
        for idx, (cat_key, meta) in enumerate(CATEGORY_METADATA.items()):
            row_idx = idx // cols
            col_idx = idx % cols

            chip = ctk.CTkFrame(
                chips_frame,
                fg_color=self.theme["card_sub_bg"],
                corner_radius=8,
                border_width=1,
                border_color=self.theme["border_color"],
                height=26,
            )
            chip.grid(row=row_idx, column=col_idx, padx=3, pady=2, sticky="ew")
            chips_frame.columnconfigure(col_idx, weight=1)

            chip_layout = ctk.CTkFrame(chip, fg_color="transparent")
            chip_layout.pack(fill="x", padx=6, pady=3)

            ctk.CTkLabel(
                chip_layout,
                text=f"{meta['icon']} {meta['label']}",
                font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
                text_color=self.theme["text_bright"],
            ).pack(side="left")

            count_lbl = ctk.CTkLabel(
                chip_layout,
                text="0",
                font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
                text_color=meta["color"],
                fg_color=self.theme["card_bg"],
                corner_radius=4,
                padx=5,
                pady=1,
            )
            count_lbl.pack(side="right")

            self.category_count_labels[cat_key] = count_lbl
            self.category_chip_frames[cat_key] = chip

    # ==========================================
    # 5. CONTROL ACTION COMMAND DECK
    # ==========================================
    def build_action_bar(self):
        act_container = ctk.CTkFrame(self.main_scroll, fg_color="transparent")
        act_container.pack(fill="x", padx=14, pady=5)

        self.dryrun_btn = ctk.CTkButton(
            act_container,
            text="🧪 RUN DRY RUN (PREVIEW)",
            command=lambda: self.start_organization(dry_run_override=True),
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            height=38,
            corner_radius=10,
            fg_color=self.theme["btn_dryrun"][0],
            hover_color=self.theme["btn_dryrun"][1],
            text_color="#FFFFFF",
        )
        self.dryrun_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.execute_btn = ctk.CTkButton(
            act_container,
            text="⚡ EXECUTE FILE TRANSFERS",
            command=lambda: self.start_organization(dry_run_override=False),
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            height=38,
            corner_radius=10,
            fg_color=self.theme["btn_execute"][0],
            hover_color=self.theme["btn_execute"][1],
            text_color="#FFFFFF",
        )
        self.execute_btn.pack(side="left", fill="x", expand=True, padx=2)

        self.unsort_btn = ctk.CTkButton(
            act_container,
            text="⏮️ UNSORT & RESTORE",
            command=self.start_unsort,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            height=38,
            corner_radius=10,
            fg_color=self.theme["btn_unsort"][0],
            hover_color=self.theme["btn_unsort"][1],
            text_color="#FFFFFF",
        )
        self.unsort_btn.pack(side="left", fill="x", expand=True, padx=2)

        self.stop_btn = ctk.CTkButton(
            act_container,
            text="🛑 CANCEL / STOP",
            command=self.stop_organization,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            height=38,
            corner_radius=10,
            fg_color=self.theme["btn_stop"][0],
            hover_color=self.theme["btn_stop"][1],
            text_color="#FFFFFF",
            state="disabled",
        )
        self.stop_btn.pack(side="left", fill="x", expand=True, padx=(4, 0))

    # ==========================================
    # 6. NEURAL ACTIVITY CONSOLE & LIVE STREAM
    # ==========================================
    def build_console_deck(self):
        console_card = ctk.CTkFrame(
            self.main_scroll,
            fg_color=self.theme["card_bg"],
            corner_radius=14,
            border_width=1,
            border_color=self.theme["border_color"],
        )
        console_card.pack(fill="both", expand=True, padx=14, pady=(2, 10))

        console_inner = ctk.CTkFrame(console_card, fg_color="transparent")
        console_inner.pack(fill="both", expand=True, padx=12, pady=8)

        # Progress Bar
        self.pbar = ctk.CTkProgressBar(
            console_inner,
            height=10,
            corner_radius=5,
            progress_color=self.theme["accent_cyan"],
            fg_color=self.theme["card_sub_bg"],
        )
        self.pbar.set(0)
        self.pbar.pack(fill="x", pady=(0, 6))

        # Status Strip
        status_strip = ctk.CTkFrame(console_inner, fg_color="transparent")
        status_strip.pack(fill="x", pady=(0, 5))

        self.status_lbl = ctk.CTkLabel(
            status_strip,
            text="🟢 SYSTEM READY // SELECT SOURCE AND DESTINATION FOLDERS TO BEGIN.",
            font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
            text_color=self.theme["text_muted"],
        )
        self.status_lbl.pack(side="left")

        # Toolbar
        ctk.CTkButton(
            status_strip,
            text="📂 Open Log CSV",
            command=self.open_log_csv,
            font=ctk.CTkFont(family="Segoe UI", size=8),
            height=20,
            width=75,
            corner_radius=5,
            fg_color=self.theme["card_sub_bg"],
            hover_color=self.theme["border_sub"],
            text_color=self.theme["accent_cyan"],
        ).pack(side="right", padx=(3, 0))

        ctk.CTkButton(
            status_strip,
            text="📋 Copy Log",
            command=self.copy_log_to_clipboard,
            font=ctk.CTkFont(family="Segoe UI", size=8),
            height=20,
            width=65,
            corner_radius=5,
            fg_color=self.theme["card_sub_bg"],
            hover_color=self.theme["border_sub"],
            text_color=self.theme["text_bright"],
        ).pack(side="right", padx=(3, 0))

        ctk.CTkButton(
            status_strip,
            text="🧹 Clear",
            command=self.clear_console,
            font=ctk.CTkFont(family="Segoe UI", size=8),
            height=20,
            width=50,
            corner_radius=5,
            fg_color=self.theme["card_sub_bg"],
            hover_color=self.theme["border_sub"],
            text_color=self.theme["text_bright"],
        ).pack(side="right")

        # Monospace Terminal
        self.log_text = scrolledtext.ScrolledText(
            console_inner,
            height=8,
            bg=self.theme["console_bg"],
            fg=self.theme["console_fg"],
            font=("Consolas", 9),
            insertbackground=self.theme["accent_cyan"],
            relief="flat",
            bd=0,
            padx=8,
            pady=6,
        )
        self.log_text.pack(fill="both", expand=True)

        self.setup_console_tags()

    def setup_console_tags(self):
        self.log_text.tag_config("OK", foreground=self.theme["accent_green"])
        self.log_text.tag_config("ERROR", foreground=self.theme["accent_red"])
        self.log_text.tag_config("WARN", foreground=self.theme["accent_amber"])
        self.log_text.tag_config("INFO", foreground=self.theme["accent_cyan"])
        self.log_text.tag_config("PURPLE", foreground=self.theme["accent_purple"])
        self.log_text.tag_config("SUCCESS", foreground="#4ADE80")

    # ==========================================
    # EXCLUSION TAG CLOUD LOGIC (Single Row Horizontal Strip)
    # ==========================================
    def render_exclusion_tags(self):
        for widget in self.tags_frame.winfo_children():
            widget.destroy()

        for folder_name in self.current_excludes:
            tag_chip = ctk.CTkFrame(
                self.tags_frame,
                fg_color=self.theme["card_bg"],
                corner_radius=5,
                border_width=1,
                border_color=self.theme["border_color"],
            )
            tag_chip.pack(side="left", padx=2, pady=1)

            ctk.CTkLabel(
                tag_chip,
                text=folder_name,
                font=ctk.CTkFont(family="Segoe UI", size=8, weight="bold"),
                text_color=self.theme["text_muted"],
            ).pack(side="left", padx=(4, 1), pady=1)

            del_btn = ctk.CTkButton(
                tag_chip,
                text="×",
                width=14,
                height=14,
                corner_radius=3,
                fg_color="transparent",
                hover_color=self.theme["border_sub"],
                text_color=self.theme["accent_red"],
                font=ctk.CTkFont(size=10, weight="bold"),
                command=lambda f=folder_name: self.remove_exclusion_tag(f),
            )
            del_btn.pack(side="left", padx=(0, 3), pady=1)

    def add_exclusion_tag(self):
        new_tag = self.new_tag_var.get().strip()
        if new_tag and new_tag not in self.current_excludes:
            self.current_excludes.append(new_tag)
            self.new_tag_var.set("")
            self.render_exclusion_tags()

    def remove_exclusion_tag(self, folder_name: str):
        if folder_name in self.current_excludes:
            self.current_excludes.remove(folder_name)
            self.render_exclusion_tags()

    def reset_exclusions(self):
        self.current_excludes = list(sorted(EXCLUDE_FOLDERS))
        self.render_exclusion_tags()

    # ==========================================
    # THEME SWITCHER
    # ==========================================
    def change_theme(self, theme_name: str):
        if theme_name in THEMES:
            self.current_theme_name = theme_name
            self.theme = THEMES[theme_name]
            ctk.set_appearance_mode(self.theme["mode"])

            for widget in self.root.winfo_children():
                widget.destroy()
            self.create_maximalist_ui()
            self.append_log(f"Theme switched to: {theme_name}", "INFO")

    # ==========================================
    # BROWSE & HELPER ACTIONS
    # ==========================================
    def browse_src(self):
        d = filedialog.askdirectory(initialdir=self.src_var.get())
        if d:
            self.src_var.set(d)

    def browse_dest(self):
        d = filedialog.askdirectory(initialdir=self.dest_var.get())
        if d:
            self.dest_var.set(d)

    def open_dest_in_explorer(self):
        dest = Path(self.dest_var.get())
        if dest.exists():
            subprocess.run(["explorer", str(dest.resolve())])
        else:
            messagebox.showinfo("Folder Not Found", f"Destination folder does not exist yet:\n{dest}")

    def open_log_csv(self):
        dest = Path(self.dest_var.get())
        log_file = dest / "_organizer_log.csv"
        if log_file.exists():
            subprocess.run(["explorer", str(log_file.resolve())])
        else:
            messagebox.showinfo("Log File", f"No log file found yet at:\n{log_file}")

    def clear_console(self):
        self.log_text.delete("1.0", tk.END)

    def copy_log_to_clipboard(self):
        content = self.log_text.get("1.0", tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        messagebox.showinfo("Clipboard", "Console log copied to clipboard!")

    def append_log(self, text: str, tag: str = "INFO"):
        self.log_text.insert(tk.END, text + "\n", tag)
        self.log_text.see(tk.END)

    # ==========================================
    # SESSION TIMER & REAL-TIME QUEUE PROCESSING
    # ==========================================
    def update_session_timer(self):
        if self.is_running:
            elapsed = time.time() - self.start_time
            hrs = int(elapsed // 3600)
            mins = int((elapsed % 3600) // 60)
            secs = int(elapsed % 60)
            self.timer_lbl.configure(text=f"⏱️ {hrs:02d}:{mins:02d}:{secs:02d}")
            self.session_timer_id = self.root.after(1000, self.update_session_timer)

    def process_queue(self):
        while not self.msg_queue.empty():
            msg_type, content = self.msg_queue.get()

            # 1. Direct STATS Message from Worker
            if msg_type == "STATS":
                stats_dict = content
                if isinstance(stats_dict, dict):
                    if "total_discovered" in stats_dict:
                        self.stats["discovered"] = stats_dict["total_discovered"]
                        self.kpi_labels["kpi_discovered"].configure(text=str(stats_dict["total_discovered"]))
                    if "photos_count" in stats_dict:
                        self.stats["photos"] = stats_dict["photos_count"]
                        self.kpi_labels["kpi_photos"].configure(text=str(stats_dict["photos_count"]))
                    if "videos_count" in stats_dict:
                        self.stats["videos"] = stats_dict["videos_count"]
                        self.kpi_labels["kpi_videos"].configure(text=str(stats_dict["videos_count"]))
                    if "duplicates_count" in stats_dict:
                        self.stats["duplicates"] = stats_dict["duplicates_count"]
                        self.kpi_labels["kpi_duplicates"].configure(text=str(stats_dict["duplicates_count"]))

                    for cat, cnt in stats_dict.get("categories", {}).items():
                        norm_cat = normalize_category_name(cat)
                        self.stats["categories"][norm_cat] = cnt
                        if norm_cat in self.category_count_labels:
                            self.category_count_labels[norm_cat].configure(text=str(cnt))
                            if cnt > 0 and norm_cat in self.category_chip_frames:
                                self.category_chip_frames[norm_cat].configure(border_color=self.theme["accent_cyan"])

            # 2. LOG Message
            elif msg_type == "LOG":
                tag, text = content
                self.append_log(text, tag)

                # Fallback parser for discovery count
                if "Discovered " in text and "candidate media files" in text:
                    try:
                        num = int(text.split("Discovered ")[1].split(" candidate")[0].strip())
                        self.stats["discovered"] = num
                        self.kpi_labels["kpi_discovered"].configure(text=str(num))
                    except Exception:
                        pass

                # Fallback parser for single file transfer
                if " ➔ " in text and ("PHOTO" in text or "VIDEO" in text):
                    dest_cat = normalize_category_name(text.split(" ➔ ")[-1].strip())
                    if "PHOTO" in text:
                        self.stats["photos"] += 1
                        self.kpi_labels["kpi_photos"].configure(text=str(self.stats["photos"]))
                    elif "VIDEO" in text:
                        self.stats["videos"] += 1
                        self.kpi_labels["kpi_videos"].configure(text=str(self.stats["videos"]))

                    if dest_cat in self.category_count_labels:
                        self.stats["categories"][dest_cat] = self.stats["categories"].get(dest_cat, 0) + 1
                        self.category_count_labels[dest_cat].configure(text=str(self.stats["categories"][dest_cat]))
                        if dest_cat in self.category_chip_frames:
                            self.category_chip_frames[dest_cat].configure(border_color=self.theme["accent_cyan"])

            # 3. Real-Time PROGRESS Message
            elif msg_type == "PROGRESS":
                live_stats = None
                if len(content) == 4:
                    current, total, status, live_stats = content
                elif len(content) == 3:
                    current, total, status = content
                else:
                    current, total = content[:2]
                    status = "Processing..."

                if total > 0:
                    prog_val = min(1.0, current / total)
                    self.pbar.set(prog_val)
                    self.status_lbl.configure(text=f"⚡ {status} ({current}/{total} - {int(prog_val*100)}%)")

                    # Live Speed & ETA Calculation
                    elapsed = max(0.1, time.time() - self.start_time)
                    speed = current / elapsed
                    eta_secs = int((total - current) / max(0.1, speed))
                    eta_str = f"{eta_secs//60:02d}:{eta_secs%60:02d}" if speed > 0 else "--:--"
                    self.kpi_labels["kpi_speed"].configure(text=f"{speed:.1f}/s (ETA {eta_str})")
                else:
                    self.status_lbl.configure(text=f"● {status}")

                # Update live KPI cards & neural matrix immediately
                if live_stats and isinstance(live_stats, dict):
                    disc = live_stats.get("total_discovered", total)
                    photos = live_stats.get("photos_count", 0)
                    videos = live_stats.get("videos_count", 0)
                    dupes = live_stats.get("duplicates_count", 0)
                    errs = live_stats.get("error_count", 0)

                    self.stats["discovered"] = disc
                    self.stats["photos"] = photos
                    self.stats["videos"] = videos
                    self.stats["duplicates"] = dupes
                    self.stats["errors"] = errs

                    self.kpi_labels["kpi_discovered"].configure(text=str(disc))
                    self.kpi_labels["kpi_photos"].configure(text=str(photos))
                    self.kpi_labels["kpi_videos"].configure(text=str(videos))
                    self.kpi_labels["kpi_duplicates"].configure(text=str(dupes))

                    tot_done = photos + videos
                    if tot_done > 0:
                        acc = max(0.0, (tot_done - errs) / tot_done * 100)
                        self.kpi_labels["kpi_confidence"].configure(text=f"{acc:.1f}%")

                    cats = live_stats.get("categories", {})
                    for cat, cnt in cats.items():
                        norm_cat = normalize_category_name(cat)
                        self.stats["categories"][norm_cat] = cnt
                        if norm_cat in self.category_count_labels:
                            self.category_count_labels[norm_cat].configure(text=str(cnt))
                            if cnt > 0 and norm_cat in self.category_chip_frames:
                                self.category_chip_frames[norm_cat].configure(border_color=self.theme["accent_cyan"])

            # 4. DONE Message
            elif msg_type == "DONE":
                stats, success_or_dryrun, log_file, checkpoint_file = content
                self.is_running = False
                self.dryrun_btn.configure(state="normal")
                self.execute_btn.configure(state="normal")
                self.unsort_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")
                self.pbar.set(1.0)
                self.status_lbl.configure(text="✅ PROCESS COMPLETED.")
                self.append_log("\n=== PROCESS EXECUTION FINISHED ===", "PURPLE")

                if stats and isinstance(stats, dict):
                    if "total_discovered" in stats:
                        self.kpi_labels["kpi_discovered"].configure(text=str(stats["total_discovered"]))
                    if "photos_count" in stats:
                        self.kpi_labels["kpi_photos"].configure(text=str(stats["photos_count"]))
                    if "videos_count" in stats:
                        self.kpi_labels["kpi_videos"].configure(text=str(stats["videos_count"]))
                    if "duplicates_count" in stats:
                        self.kpi_labels["kpi_duplicates"].configure(text=str(stats["duplicates_count"]))

                    tot_done = stats.get("photos_count", 0) + stats.get("videos_count", 0)
                    if tot_done > 0:
                        errs = stats.get("error_count", 0)
                        acc = max(0.0, (tot_done - errs) / tot_done * 100)
                        self.kpi_labels["kpi_confidence"].configure(text=f"{acc:.1f}%")

                    for cat, cnt in stats.get("categories", {}).items():
                        norm_cat = normalize_category_name(cat)
                        if norm_cat in self.category_count_labels:
                            self.category_count_labels[norm_cat].configure(text=str(cnt))

                    self.show_summary_dialog(stats, success_or_dryrun, log_file, checkpoint_file)

        self.root.after(100, self.process_queue)

    # ==========================================
    # TASK EXECUTION DISPATCHERS
    # ==========================================
    def start_organization(self, dry_run_override: bool = None):
        src = Path(self.src_var.get().strip())
        dest = Path(self.dest_var.get().strip())

        if not src.exists():
            messagebox.showerror("Invalid Source", f"Source folder does not exist:\n{src}")
            return
        if not dest.exists():
            try:
                dest.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                messagebox.showerror("Destination Error", f"Could not create destination folder:\n{e}")
                return

        # Reset counters
        self.stats = {
            "discovered": 0,
            "photos": 0,
            "videos": 0,
            "duplicates": 0,
            "errors": 0,
            "categories": {cat: 0 for cat in DEFAULT_CATEGORIES},
        }
        for key in ("kpi_discovered", "kpi_photos", "kpi_videos", "kpi_duplicates"):
            self.kpi_labels[key].configure(text="0")
        self.kpi_labels["kpi_speed"].configure(text="0.0/s")
        self.kpi_labels["kpi_confidence"].configure(text="--")

        for cat, lbl in self.category_count_labels.items():
            lbl.configure(text="0")
            if cat in self.category_chip_frames:
                self.category_chip_frames[cat].configure(border_color=self.theme["border_color"])

        self.is_running = True
        self.cancel_event.clear()
        self.start_time = time.time()
        self.update_session_timer()

        self.dryrun_btn.configure(state="disabled")
        self.execute_btn.configure(state="disabled")
        self.unsort_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.log_text.delete("1.0", tk.END)

        is_dry_run = dry_run_override if dry_run_override is not None else False
        action_mode = "move" if "Move" in self.action_segmented.get() else "copy"

        config = {
            "src": src,
            "dest": dest,
            "action": action_mode,
            "dry_run": is_dry_run,
            "confidence": float(self.confidence_slider.get()),
            "batch_size": int(self.batch_slider.get()),
            "threads": int(self.threads_slider.get()),
            "classify_videos": self.classify_videos_var.get(),
            "dedupe": self.dedupe_var.get(),
            "preserve_folders": self.preserve_folders_var.get(),
            "model_name": self.model_var.get(),
            "exclude_folders": ", ".join(self.current_excludes),
        }

        self.append_log(f"⚡ Launching Neural Organizer in {'DRY-RUN (Preview)' if is_dry_run else action_mode.upper()} mode...", "PURPLE")
        threading.Thread(target=self.run_worker, args=(config,), daemon=True).start()

    def start_unsort(self):
        dest = Path(self.dest_var.get().strip())
        log_file = dest / "_organizer_log.csv"

        if not log_file.exists():
            messagebox.showerror(
                "Missing Log File",
                f"No transfer log (_organizer_log.csv) found in destination:\n{dest}\n\nCannot restore files without log history."
            )
            return

        confirm = messagebox.askyesno(
            "Confirm Unsort & Restore",
            f"Are you sure you want to restore all organized files in:\n{dest}\n\nback to their original source locations?"
        )
        if not confirm:
            return

        self.is_running = True
        self.cancel_event.clear()
        self.start_time = time.time()
        self.update_session_timer()

        self.dryrun_btn.configure(state="disabled")
        self.execute_btn.configure(state="disabled")
        self.unsort_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.log_text.delete("1.0", tk.END)

        self.append_log(f"⏪ Initializing Unsort & Restore pipeline for {dest}...", "PURPLE")
        threading.Thread(target=self.run_unsort_worker, args=(dest,), daemon=True).start()

    def run_worker(self, config: dict):
        execute_run_worker(config, msg_queue=self.msg_queue, cancel_flag=self.cancel_event)

    def run_unsort_worker(self, dest: Path):
        execute_run_unsort_worker(dest_dir=dest, dry_run=False, msg_queue=self.msg_queue, cancel_flag=self.cancel_event)

    def stop_organization(self):
        if self.is_running:
            self.cancel_event.set()
            self.append_log("🛑 Stop requested. Finishing in-flight tasks and saving checkpoint...", "WARN")

    def show_summary_dialog(self, stats: dict, dry_run: bool, log_file: Path, checkpoint_file: Path):
        summary_text = (
            f"Mode: {'DRY RUN (Preview)' if dry_run else 'EXECUTED (' + stats.get('action', 'N/A').upper() + ')'}\n\n"
            f"Total Discovered Files: {stats.get('total_discovered', 0)}\n"
            f"Processed Photos:       {stats.get('photos_count', 0)}\n"
            f"Processed Videos:       {stats.get('videos_count', 0)}\n"
            f"Skipped (Checkpointed): {stats.get('skipped_count', 0)}\n"
            f"Duplicates Routed:      {stats.get('duplicates_count', 0)}\n"
            f"Errors / Fallbacks:     {stats.get('error_count', 0)}\n\n"
            f"Neural Category Breakdown:\n"
        )
        for cat, cnt in sorted(stats.get("categories", {}).items()):
            summary_text += f"  • {cat}: {cnt} items\n"

        summary_text += f"\nFull Log Saved To:\n{log_file}"
        messagebox.showinfo("Organization Finished", summary_text)


def main():
    root = ctk.CTk()
    app = MaximalistMediaOrganizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
