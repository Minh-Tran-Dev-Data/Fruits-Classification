import os
import csv
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

import processing as proc

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
MAX_IMAGES_PER_CLASS = 20   # so anh mau toi da lay moi loai khi xay ngan hang dac trung

APP_TITLE = "Phan Loai Trai Cay  |  HSV - Color Histogram - Thong Ke"
CANVAS_W, CANVAS_H = 400, 400

# Bang mau giao dien (fruit-themed)
COLOR_BG = "#f4f6f2"
COLOR_PANEL = "#ffffff"
COLOR_PRIMARY = "#2e7d32"       # xanh la (nut chinh)
COLOR_PRIMARY_DARK = "#1b5e20"
COLOR_ACCENT = "#ef6c00"        # cam (nhan manh)
COLOR_TEXT = "#212121"
COLOR_MUTED = "#6b6b6b"
COLOR_BORDER = "#dcdfd9"
FONT_FAMILY = "Segoe UI"


def resize_keep_aspect(image, max_w, max_h):
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


class FruitClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1320x860")
        self.root.minsize(1150, 740)
        self.root.configure(bg=COLOR_BG)

        self.original_rgb = None
        self.current_mask = None
        self.current_segmented = None
        self.image_path = None
        self.folder_images = []
        self.current_folder = None
        self.video_cap = None
        self.video_playing = False
        self.video_after_id = None
        self._last_hist_fig = None
        self._last_detail_fig = None

        self.db_norm = None
        self.norm_params = None

        self._setup_style()
        self._build_menu()
        self._build_layout()
        self._build_status_bar()

        # Tu dong xay ngan hang dac trung tu DATASET_DIR khi khoi dong
        self.root.after(200, self.load_dataset_auto)

    # ------------------------------------------------------------- Style
    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", font=(FONT_FAMILY, 10), background=COLOR_BG, foreground=COLOR_TEXT)
        style.configure("TFrame", background=COLOR_BG)
        style.configure("Panel.TFrame", background=COLOR_PANEL)
        style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT)
        style.configure("Panel.TLabel", background=COLOR_PANEL, foreground=COLOR_TEXT)
        style.configure("Header.TLabel", background=COLOR_BG, foreground=COLOR_PRIMARY_DARK,
                         font=(FONT_FAMILY, 11, "bold"))
        style.configure("Muted.TLabel", background=COLOR_BG, foreground=COLOR_MUTED,
                         font=(FONT_FAMILY, 9))
        style.configure("PanelMuted.TLabel", background=COLOR_PANEL, foreground=COLOR_MUTED,
                         font=(FONT_FAMILY, 9))

        style.configure("TLabelframe", background=COLOR_BG, bordercolor=COLOR_BORDER)
        style.configure("TLabelframe.Label", background=COLOR_BG, foreground=COLOR_PRIMARY_DARK,
                         font=(FONT_FAMILY, 10, "bold"))

        style.configure("TButton", font=(FONT_FAMILY, 10), padding=(10, 7))
        style.configure("Accent.TButton", font=(FONT_FAMILY, 10, "bold"), padding=(10, 8),
                         background=COLOR_PRIMARY, foreground="white")
        style.map("Accent.TButton",
                  background=[("active", COLOR_PRIMARY_DARK), ("disabled", "#a5c9a8")])
        style.configure("Secondary.TButton", font=(FONT_FAMILY, 9), padding=(8, 5))

        style.configure("TNotebook", background=COLOR_BG, borderwidth=0)
        style.configure("TNotebook.Tab", font=(FONT_FAMILY, 9, "bold"), padding=(10, 6))
        style.map("TNotebook.Tab", background=[("selected", COLOR_PRIMARY)],
                  foreground=[("selected", "white")])

        style.configure("TScale", background=COLOR_BG)
        style.configure("Horizontal.TProgressbar", background=COLOR_PRIMARY, troughcolor=COLOR_BORDER)

    # ------------------------------------------------------------------ UI
    def _build_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Mo anh...", command=self.open_image)
        file_menu.add_command(label="Mo thu muc anh (batch)...", command=self.open_folder)
        file_menu.add_command(label="Mo video...", command=self.open_video)
        file_menu.add_separator()
        file_menu.add_command(label="Luu anh da xu ly...", command=self.save_processed_image)
        file_menu.add_command(label="Luu bieu do histogram...", command=self.save_histogram)
        file_menu.add_command(label="Xuat bao cao thong ke (CSV)...", command=self.export_stats_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Thoat", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Gioi thieu", command=self._show_about)
        menubar.add_cascade(label="Tro giup", menu=help_menu)
        self.root.config(menu=menubar)

    def _build_layout(self):
        header = tk.Frame(self.root, bg=COLOR_PRIMARY, height=54)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)
        tk.Label(header, text="Phan Loai Trai Cay", bg=COLOR_PRIMARY, fg="white",
                 font=(FONT_FAMILY, 15, "bold")).pack(side=tk.LEFT, padx=18)
        tk.Label(header, text="HSV Segmentation  -  Color Histogram  -  Thong ke",
                 bg=COLOR_PRIMARY, fg="#dff0da", font=(FONT_FAMILY, 10)).pack(side=tk.LEFT)

        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ---- trai: danh sach anh (batch) ----
        left = ttk.Labelframe(main, text=" Danh sach anh (batch) ", width=210)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        self.listbox = tk.Listbox(left, width=26, height=32, bd=0, highlightthickness=0,
                                   selectbackground=COLOR_PRIMARY, activestyle="none",
                                   font=(FONT_FAMILY, 9))
        self.listbox.pack(fill=tk.Y, expand=True, padx=6, pady=6)
        self.listbox.bind("<<ListboxSelect>>", self._on_select_from_list)

        # ---- giua: anh truoc/sau + ket qua ----
        center = ttk.Frame(main)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        img_frame = ttk.Frame(center)
        img_frame.pack(fill=tk.BOTH, expand=True)

        before_box = ttk.Labelframe(img_frame, text=" Anh truoc xu ly ")
        before_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.canvas_before = tk.Canvas(before_box, bg="#eceeea", width=CANVAS_W, height=CANVAS_H,
                                        highlightthickness=0)
        self.canvas_before.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        after_box = ttk.Labelframe(img_frame, text=" Anh sau xu ly / Ket qua ")
        after_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        self.canvas_after = tk.Canvas(after_box, bg="#eceeea", width=CANVAS_W, height=CANVAS_H,
                                       highlightthickness=0)
        self.canvas_after.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.video_ctrl = ttk.Frame(center)
        ttk.Button(self.video_ctrl, text="Play", style="Secondary.TButton",
                   command=self.video_play).pack(side=tk.LEFT, padx=3, pady=4)
        ttk.Button(self.video_ctrl, text="Pause", style="Secondary.TButton",
                   command=self.video_pause).pack(side=tk.LEFT, padx=3, pady=4)
        ttk.Button(self.video_ctrl, text="Stop", style="Secondary.TButton",
                   command=self.video_stop).pack(side=tk.LEFT, padx=3, pady=4)

        result_box = ttk.Labelframe(center, text=" Ket qua ")
        result_box.pack(fill=tk.BOTH, expand=False, pady=(8, 0))
        self.txt_result = tk.Text(result_box, height=11, font=("Consolas", 9), bd=0,
                                   bg=COLOR_PANEL, fg=COLOR_TEXT, padx=8, pady=6,
                                   highlightthickness=0)
        self.txt_result.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.txt_result.tag_configure("h1", font=(FONT_FAMILY, 10, "bold"), foreground=COLOR_PRIMARY_DARK)
        self.txt_result.tag_configure("accent", foreground=COLOR_ACCENT, font=("Consolas", 9, "bold"))
        self.txt_result.tag_configure("muted", foreground=COLOR_MUTED)

        # ---- phai: dieu khien (tabs) ----
        right = ttk.Frame(main, width=360)
        right.pack(side=tk.LEFT, fill=tk.Y)

        notebook = ttk.Notebook(right)
        notebook.pack(fill=tk.BOTH, expand=True)

        self._build_tab_hsv(notebook)
        self._build_tab_histogram(notebook)
        self._build_tab_stats(notebook)
        self._build_tab_classify(notebook)
        self._build_tab_batch(notebook)

    # ---- Tab 1: HSV Segmentation ----
    def _build_tab_hsv(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="HSV")

        ttk.Label(tab, text="Nguong HSV", style="Header.TLabel").pack(anchor="w", padx=10, pady=(12, 4))

        self.h_low = self._make_slider(tab, "H thap (0-179)", 0, 179, 0)
        self.h_high = self._make_slider(tab, "H cao (0-179)", 0, 179, 10)
        self.s_low = self._make_slider(tab, "S thap (0-255)", 0, 255, 70)
        self.s_high = self._make_slider(tab, "S cao (0-255)", 0, 255, 255)
        self.v_low = self._make_slider(tab, "V thap (0-255)", 0, 255, 50)
        self.v_high = self._make_slider(tab, "V cao (0-255)", 0, 255, 255)

        ttk.Button(tab, text="Ap dung phan doan HSV", style="Accent.TButton",
                   command=self.apply_hsv_segmentation).pack(fill=tk.X, padx=10, pady=(12, 8))

        ttk.Separator(tab).pack(fill=tk.X, pady=8, padx=10)
        ttk.Label(tab, text="Preset mau tham khao", style="Header.TLabel").pack(anchor="w", padx=10)
        presets = {
            "Do (Red)": (0, 10, 70, 255, 50, 255),
            "Vang (Yellow)": (20, 35, 70, 255, 70, 255),
            "Cam (Orange)": (5, 20, 100, 255, 100, 255),
            "Xanh la (Green)": (36, 85, 50, 255, 50, 255),
            "Tim (Purple)": (130, 160, 40, 255, 30, 255),
        }
        preset_var = tk.StringVar(value="Do (Red)")
        combo = ttk.Combobox(tab, textvariable=preset_var, values=list(presets.keys()), state="readonly")
        combo.pack(padx=10, pady=(4, 0), fill=tk.X)

        def apply_preset(event=None):
            h_lo, h_hi, s_lo, s_hi, v_lo, v_hi = presets[preset_var.get()]
            self.h_low.set(h_lo); self.h_high.set(h_hi)
            self.s_low.set(s_lo); self.s_high.set(s_hi)
            self.v_low.set(v_lo); self.v_high.set(v_hi)

        combo.bind("<<ComboboxSelected>>", apply_preset)

    def _make_slider(self, parent, label, mn, mx, default):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=3)
        var = tk.IntVar(value=default)
        lbl = ttk.Label(frame, text=f"{label}: {default}", width=20, style="Muted.TLabel")
        lbl.pack(side=tk.LEFT)

        def on_change(v):
            lbl.config(text=f"{label}: {int(float(v))}")

        scale = ttk.Scale(frame, from_=mn, to=mx, orient=tk.HORIZONTAL, variable=var, command=on_change)
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return var

    # ---- Tab 2: Color Histogram ----
    def _build_tab_histogram(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Histogram")

        ttk.Label(tab, text="Color Histogram", style="Header.TLabel").pack(anchor="w", padx=10, pady=(12, 6))

        ttk.Label(tab, text="So bins", style="Muted.TLabel").pack(anchor="w", padx=10)
        self.hist_bins = tk.IntVar(value=32)
        ttk.Spinbox(tab, from_=8, to=256, increment=8, textvariable=self.hist_bins, width=10).pack(
            padx=10, pady=(2, 8), anchor="w")

        self.hist_use_mask = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab, text="Chi tinh tren vung da phan doan (mask)",
                         variable=self.hist_use_mask).pack(anchor="w", padx=10, pady=(0, 8))

        ttk.Button(tab, text="Ve Color Histogram", style="Accent.TButton",
                   command=self.show_histogram).pack(fill=tk.X, padx=10, pady=4)

        self.hist_canvas_frame = ttk.Frame(tab)
        self.hist_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

    # ---- Tab 3: Thong ke ----
    def _build_tab_stats(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Thong ke")

        ttk.Label(tab, text="Thong ke anh", style="Header.TLabel").pack(anchor="w", padx=10, pady=(12, 6))
        ttk.Label(tab, text="Tinh tren vung da phan doan (mask tu\ntab HSV), hoac toan anh neu chua co mask.",
                  style="Muted.TLabel", justify=tk.LEFT).pack(anchor="w", padx=10, pady=(0, 10))

        ttk.Button(tab, text="Tinh thong ke chi tiet", style="Accent.TButton",
                   command=self.run_statistics).pack(fill=tk.X, padx=10, pady=4)

    # ---- Tab 4: Phan loai ----
    def _build_tab_classify(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Phan loai")

        ttk.Label(tab, text="Ngan hang dac trung", style="Header.TLabel").pack(anchor="w", padx=10, pady=(12, 4))
        self.classify_db_status = ttk.Label(tab, text="Dang tai...", style="Muted.TLabel",
                                             wraplength=310, justify=tk.LEFT)
        self.classify_db_status.pack(anchor="w", padx=10, pady=(0, 6))

        ttk.Button(tab, text="Tai lai du lieu mau", style="Secondary.TButton",
                   command=self.load_dataset_auto).pack(anchor="w", padx=10, pady=(0, 10))

        ttk.Separator(tab).pack(fill=tk.X, padx=10, pady=6)

        ttk.Label(tab, text="So lang gieng gan nhat (k)", style="Muted.TLabel").pack(anchor="w", padx=10, pady=(4, 2))
        self.classify_k = tk.IntVar(value=3)
        ttk.Spinbox(tab, from_=1, to=9, textvariable=self.classify_k, width=10).pack(padx=10, anchor="w")

        ttk.Button(tab, text="Phan loai anh hien tai", style="Accent.TButton",
                   command=self.classify_current_image).pack(fill=tk.X, padx=10, pady=(14, 10))

        ttk.Separator(tab).pack(fill=tk.X, padx=10, pady=6)
        ttk.Label(tab, text="Giai thich du doan", style="Header.TLabel").pack(anchor="w", padx=10, pady=(4, 4))

        explain_row = ttk.Frame(tab)
        explain_row.pack(fill=tk.X, padx=10)
        self.nearest_thumb = tk.Canvas(explain_row, width=110, height=110, bg="#eceeea",
                                        highlightthickness=1, highlightbackground=COLOR_BORDER)
        self.nearest_thumb.pack(side=tk.LEFT, pady=4)
        self.nearest_label_var = tk.StringVar(value="Chua co ket qua")
        ttk.Label(explain_row, textvariable=self.nearest_label_var, style="Muted.TLabel",
                  wraplength=190, justify=tk.LEFT).pack(side=tk.LEFT, padx=8)

        self.detail_canvas_frame = ttk.Frame(tab)
        self.detail_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

    # ---- Tab 5: Batch ----
    def _build_tab_batch(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Hang loat")

        ttk.Label(tab, text="Xu ly hang loat", style="Header.TLabel").pack(anchor="w", padx=10, pady=(12, 6))
        ttk.Label(tab, text="Xu ly toan bo anh trong thu muc da mo\n(File > Mo thu muc anh), dung nguong HSV\ndang chinh o tab HSV, xuat CSV.",
                  style="Muted.TLabel", justify=tk.LEFT).pack(anchor="w", padx=10, pady=(0, 10))
        ttk.Button(tab, text="Chay xu ly hang loat...", style="Accent.TButton",
                   command=self.run_batch).pack(fill=tk.X, padx=10, pady=4)

        self.batch_progress = ttk.Progressbar(tab, mode="determinate")
        self.batch_progress.pack(fill=tk.X, padx=10, pady=(14, 4))
        self.batch_progress_label = ttk.Label(tab, text="", style="Muted.TLabel")
        self.batch_progress_label.pack(anchor="w", padx=10)

    def _build_status_bar(self):
        bar = tk.Frame(self.root, bg=COLOR_PRIMARY_DARK, height=26)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        bar.pack_propagate(False)
        self.status_var = tk.StringVar(value="San sang.")
        tk.Label(bar, textvariable=self.status_var, bg=COLOR_PRIMARY_DARK, fg="white",
                 font=(FONT_FAMILY, 9), anchor="w").pack(fill=tk.BOTH, expand=True, padx=10)

    def _set_status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()

    def _show_about(self):
        messagebox.showinfo(
            "Gioi thieu",
            "Chu de 6 - Project 1: Phan loai trai cay\n"
            "Ky thuat: HSV Segmentation, Color Histogram, Thong ke\n"
            "Mon hoc: Xu ly anh"
        )

    # -------------------------------------------------------------- Dataset (tu dong)
    def load_dataset_auto(self):
        if not os.path.isdir(DATASET_DIR):
            self.classify_db_status.config(
                text=f"Khong tim thay thu muc dataset:\n{DATASET_DIR}\n\n"
                     f"Tao thu muc nay va them anh mau theo cau truc:\n"
                     f"dataset/TenLoai/anh.jpg",
                foreground="#b91c1c")
            return

        self.classify_db_status.config(text="Dang xay ngan hang dac trung...", foreground="#0066cc")
        self.root.update_idletasks()

        def worker():
            try:
                db_norm, norm_params = proc.build_pipeline_from_folder(
                    DATASET_DIR, max_images_per_class=MAX_IMAGES_PER_CLASS)
            except Exception as e:
                self.root.after(0, lambda: self.classify_db_status.config(
                    text=f"Loi khi doc dataset: {e}", foreground="#b91c1c"))
                return

            if not db_norm:
                self.root.after(0, lambda: self.classify_db_status.config(
                    text=f"Thu muc dataset khong co du lieu hop le:\n{DATASET_DIR}",
                    foreground="#b91c1c"))
                return

            self.db_norm = db_norm
            self.norm_params = norm_params
            counts = {lb: len(items) for lb, items in db_norm.items()}
            summary = "\n".join(f"  - {lb}: {n} anh" for lb, n in counts.items())

            self.root.after(0, lambda: self.classify_db_status.config(
                text=f"San sang ({len(counts)} loai):\n{summary}", foreground="#0a7a1e"))
            self.root.after(0, lambda: self._set_status(
                f"Da tai ngan hang dac trung: {len(counts)} loai."))

        threading.Thread(target=worker, daemon=True).start()

    # -------------------------------------------------------------- Open
    def _load_image_rgb(self, path):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            return None
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def open_image(self):
        self.video_stop()
        path = filedialog.askopenfilename(
            title="Chon anh",
            filetypes=[("Anh", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("Tat ca", "*.*")])
        if not path:
            return
        img = self._load_image_rgb(path)
        if img is None:
            messagebox.showerror("Loi", "Khong doc duoc anh nay.")
            return
        self.image_path = path
        self.original_rgb = img
        self.current_mask = None
        self.current_segmented = None
        self.folder_images = []
        self.listbox.delete(0, tk.END)
        self._show_image(self.canvas_before, img)
        self._show_image(self.canvas_after, img)
        self.txt_result.delete("1.0", tk.END)
        self._set_status(f"Da mo anh: {os.path.basename(path)} ({img.shape[1]}x{img.shape[0]})")

    def open_folder(self):
        self.video_stop()
        folder = filedialog.askdirectory(title="Chon thu muc chua anh")
        if not folder:
            return
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(valid_ext)])
        if not files:
            messagebox.showwarning("Canh bao", "Thu muc khong co anh hop le.")
            return
        self.current_folder = folder
        self.folder_images = files
        self.listbox.delete(0, tk.END)
        for f in files:
            self.listbox.insert(tk.END, f)
        self._set_status(f"Da mo thu muc: {folder} ({len(files)} anh)")
        self._load_from_folder(0)

    def _on_select_from_list(self, event):
        sel = self.listbox.curselection()
        if sel:
            self._load_from_folder(sel[0])

    def _load_from_folder(self, idx):
        fname = self.folder_images[idx]
        path = os.path.join(self.current_folder, fname)
        img = self._load_image_rgb(path)
        if img is None:
            return
        self.image_path = path
        self.original_rgb = img
        self.current_mask = None
        self.current_segmented = None
        self._show_image(self.canvas_before, img)
        self._show_image(self.canvas_after, img)
        self._set_status(f"Dang xem: {fname}")

    def open_video(self):
        path = filedialog.askopenfilename(
            title="Chon video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("Tat ca", "*.*")])
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Loi", "Khong mo duoc video nay.")
            return
        self.video_cap = cap
        self.video_ctrl.pack(fill=tk.X, pady=4)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.original_rgb = frame_rgb
            self._show_image(self.canvas_before, frame_rgb)
            self._show_image(self.canvas_after, frame_rgb)
        self._set_status(f"Da mo video: {os.path.basename(path)}. Nhan Play de chay.")

    def video_play(self):
        if self.video_cap is None:
            return
        self.video_playing = True
        self._video_loop()

    def _video_loop(self):
        if not self.video_playing or self.video_cap is None:
            return
        ret, frame = self.video_cap.read()
        if not ret:
            self.video_playing = False
            self._set_status("Video da ket thuc.")
            return
        self.original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._show_image(self.canvas_before, self.original_rgb)
        self.apply_hsv_segmentation(silent=True)
        self.video_after_id = self.root.after(30, self._video_loop)

    def video_pause(self):
        self.video_playing = False

    def video_stop(self):
        self.video_playing = False
        if self.video_after_id:
            self.root.after_cancel(self.video_after_id)
            self.video_after_id = None
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        self.video_ctrl.pack_forget()

    # -------------------------------------------------------------- Display
    def _show_image(self, canvas, image_rgb):
        img_resized = resize_keep_aspect(image_rgb, CANVAS_W, CANVAS_H)
        pil_img = Image.fromarray(img_resized)
        tk_img = ImageTk.PhotoImage(pil_img)
        canvas.delete("all")
        canvas.create_image(canvas.winfo_width() // 2 or CANVAS_W // 2,
                             canvas.winfo_height() // 2 or CANVAS_H // 2,
                             image=tk_img, anchor="center")
        canvas.image = tk_img

    def _check_image_loaded(self):
        if self.original_rgb is None:
            messagebox.showwarning("Canh bao", "Vui long mo anh hoac video truoc.")
            return False
        return True

    # -------------------------------------------------------------- Tab HSV
    def apply_hsv_segmentation(self, silent=False):
        if not self._check_image_loaded():
            return
        h_lo, h_hi = int(self.h_low.get()), int(self.h_high.get())
        s_lo, s_hi = int(self.s_low.get()), int(self.s_high.get())
        v_lo, v_hi = int(self.v_low.get()), int(self.v_high.get())

        mask, segmented = proc.hsv_segment(self.original_rgb, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)
        self.current_mask = mask
        self.current_segmented = segmented
        self._show_image(self.canvas_after, segmented)

        if not silent:
            feat = proc.extract_hsv(self.original_rgb, mask=mask)
            pct = 100.0 * np.count_nonzero(mask) / mask.size
            self.txt_result.delete("1.0", tk.END)
            self.txt_result.insert(tk.END, "DAC TRUNG HSV (mean / std)\n", "h1")
            self.txt_result.insert(tk.END, f"Ty le vung phat hien: {pct:.2f}%\n\n", "accent")
            self.txt_result.insert(tk.END, f"mean_H={feat[0]:.2f}   std_H={feat[1]:.2f}\n")
            self.txt_result.insert(tk.END, f"mean_S={feat[2]:.2f}   std_S={feat[3]:.2f}\n")
            self.txt_result.insert(tk.END, f"mean_V={feat[4]:.2f}   std_V={feat[5]:.2f}\n")
            self._set_status(f"Da phan doan HSV. Dien tich khop: {pct:.2f}%")

    # -------------------------------------------------------------- Tab Histogram
    def show_histogram(self):
        if not self._check_image_loaded():
            return
        mask = self.current_mask if self.hist_use_mask.get() else None
        bins = int(self.hist_bins.get())
        hist_h, hist_s, hist_v = proc.histogram_procession(self.original_rgb, mask=mask, bins=bins)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=100)
        ax.plot(hist_h, color="#ef6c00", label="H", linewidth=1.6)
        ax.plot(hist_s, color="#2e7d32", label="S", linewidth=1.6)
        ax.plot(hist_v, color="#1565c0", label="V", linewidth=1.6)
        ax.set_title(f"Color Histogram (bins={bins})")
        ax.set_xlabel("Bin")
        ax.set_ylabel("Tan suat (chuan hoa)")
        ax.legend()
        fig.tight_layout()
        self._last_hist_fig = fig

        for w in self.hist_canvas_frame.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.hist_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._set_status("Da ve color histogram.")

    # -------------------------------------------------------------- Tab Thong ke
    def run_statistics(self):
        if not self._check_image_loaded():
            return
        mask = self.current_mask
        stats, boxes = proc.compute_full_statistics(self.original_rgb, mask=mask)
        self.txt_result.delete("1.0", tk.END)
        self.txt_result.insert(tk.END, "THONG KE ANH\n", "h1")
        for k, v in stats.items():
            self.txt_result.insert(tk.END, f"{k:28s}: {v}\n")
        if boxes:
            self.txt_result.insert(tk.END, f"\nBounding boxes doi tuong ({len(boxes)}):\n", "accent")
            for i, b in enumerate(boxes, 1):
                self.txt_result.insert(tk.END, f"  #{i}: x={b[0]}, y={b[1]}, w={b[2]}, h={b[3]}\n")
        self._set_status("Da tinh thong ke.")

    # -------------------------------------------------------------- Tab Phan loai
    def classify_current_image(self):
        if not self._check_image_loaded():
            return
        if self.db_norm is None or self.norm_params is None:
            messagebox.showwarning(
                "Canh bao",
                f"Chua co ngan hang dac trung.\n"
                f"Kiem tra thu muc dataset:\n{DATASET_DIR}\n"
                f"roi bam 'Tai lai du lieu mau'.")
            return

        k = int(self.classify_k.get())
        label, confidence, dist_table, detail = proc.classify_fruit_detailed(
            self.original_rgb, self.db_norm, self.norm_params, k=k)

        # ---- Text ket qua ----
        self.txt_result.delete("1.0", tk.END)
        self.txt_result.insert(tk.END, "KET QUA PHAN LOAI\n", "h1")
        self.txt_result.insert(tk.END, f"Du doan: {label}\n", "accent")
        self.txt_result.insert(tk.END, f"Do tin cay (trong {k} lang gieng gan nhat): {confidence*100:.0f}%\n\n")

        self.txt_result.insert(tk.END, "Khoang cach trung binh toi tung loai:\n", "h1")
        for lb, d in sorted(dist_table.items(), key=lambda x: x[1]):
            self.txt_result.insert(tk.END, f"  {lb:<15s}: {d:.3f}\n")

        self.txt_result.insert(tk.END, "\nTop lang gieng gan nhat (dung de vote):\n", "h1")
        for m in detail["top_k"]:
            self.txt_result.insert(tk.END, f"  [{m['label']:<10s}] {os.path.basename(m['path'])}  "
                                             f"(khoang cach {m['dist']:.3f})\n")

        # ---- Anh mau giong nhat (thumbnail) ----
        near_img = self._load_image_rgb(detail["nearest_path"])
        if near_img is not None:
            thumb = resize_keep_aspect(near_img, 108, 108)
            pil_thumb = Image.fromarray(thumb)
            tk_thumb = ImageTk.PhotoImage(pil_thumb)
            self.nearest_thumb.delete("all")
            self.nearest_thumb.create_image(55, 55, image=tk_thumb, anchor="center")
            self.nearest_thumb.image = tk_thumb

        self.nearest_label_var.set(
            f"Anh mau giong nhat:\n{os.path.basename(detail['nearest_path'])}\n"
            f"(loai: {detail['nearest_label']})\n"
            f"Tong khoang cach: {detail['nearest_total_dist']:.3f}")

        # ---- Bieu do dong gop tung ky thuat ----
        self._draw_breakdown_chart(detail["breakdown"])

        self._set_status(f"Phan loai: {label} (tin cay {confidence*100:.0f}%)")

    def _draw_breakdown_chart(self, breakdown):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        names = ["HSV", "Histogram", "Thong ke"]
        values = [breakdown["hsv"], breakdown["hist"], breakdown["stats"]]
        colors = ["#ef6c00", "#2e7d32", "#1565c0"]

        fig, ax = plt.subplots(figsize=(3.2, 2.6), dpi=100)
        bars = ax.bar(names, values, color=colors)
        ax.set_title("Dong gop khoang cach\ntoi anh mau giong nhat", fontsize=9)
        ax.set_ylabel("Khoang cach", fontsize=8)
        ax.tick_params(labelsize=8)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        self._last_detail_fig = fig

        for w in self.detail_canvas_frame.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.detail_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # -------------------------------------------------------------- Tab Hang loat
    def run_batch(self):
        if not self.folder_images:
            messagebox.showwarning("Canh bao", "Vui long mo mot thu muc anh truoc (File > Mo thu muc anh).")
            return
        out_path = filedialog.asksaveasfilename(
            title="Luu bao cao CSV", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not out_path:
            return

        h_lo, h_hi = int(self.h_low.get()), int(self.h_high.get())
        s_lo, s_hi = int(self.s_low.get()), int(self.s_high.get())
        v_lo, v_hi = int(self.v_low.get()), int(self.v_high.get())

        def worker():
            rows = []
            total = len(self.folder_images)
            for i, fname in enumerate(self.folder_images, start=1):
                path = os.path.join(self.current_folder, fname)
                img = self._load_image_rgb(path)
                if img is None:
                    continue
                mask, _ = proc.hsv_segment(img, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)
                stats, _ = proc.compute_full_statistics(img, mask=mask)
                row = {"file": fname}
                row.update(stats)
                rows.append(row)

                self.batch_progress["maximum"] = total
                self.batch_progress["value"] = i
                self.batch_progress_label.config(text=f"{i}/{total}: {fname}")
                self.root.update_idletasks()

            if rows:
                with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)

            self._set_status(f"Da xu ly xong {len(rows)} anh. Bao cao: {out_path}")
            messagebox.showinfo("Hoan tat", f"Da xu ly {len(rows)} anh.\nBao cao luu tai:\n{out_path}")

        threading.Thread(target=worker, daemon=True).start()

    # -------------------------------------------------------------- Save
    def save_processed_image(self):
        if self.current_segmented is None:
            messagebox.showwarning("Canh bao", "Chua co anh da xu ly de luu.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            bgr = cv2.cvtColor(self.current_segmented, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr)
            self._set_status(f"Da luu anh: {path}")

    def save_histogram(self):
        if self._last_hist_fig is None:
            messagebox.showwarning("Canh bao", "Chua ve histogram nao de luu. Vao tab Histogram va bam 've Color Histogram'.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            self._last_hist_fig.savefig(path)
            self._set_status(f"Da luu histogram: {path}")

    def export_stats_csv(self):
        if not self._check_image_loaded():
            return
        mask = self.current_mask
        stats, _ = proc.compute_full_statistics(self.original_rgb, mask=mask)
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["chi_so", "gia_tri"])
            for k, v in stats.items():
                writer.writerow([k, v])
        self._set_status(f"Da xuat thong ke: {path}")
