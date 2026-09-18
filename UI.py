import os
import csv
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

import processing as proc

APP_TITLE = "Ung dung Phan loai Trai cay - HSV / Color Histogram / Thong ke"
CANVAS_W, CANVAS_H = 420, 420


def resize_keep_aspect(image, max_w, max_h):
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


class FruitClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1280x820")
        self.root.minsize(1100, 720)

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

        # State cho phan loai (tab 5)
        self.db_norm = None
        self.norm_params = None
        self.reference_folder = None

        self._build_menu()
        self._build_layout()
        self._build_status_bar()

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
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = ttk.Frame(main, width=200)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        ttk.Label(left, text="Danh sach anh (batch)", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.listbox = tk.Listbox(left, width=26, height=30)
        self.listbox.pack(fill=tk.Y, expand=True, pady=4)
        self.listbox.bind("<<ListboxSelect>>", self._on_select_from_list)

        center = ttk.Frame(main)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

        img_frame = ttk.Frame(center)
        img_frame.pack(fill=tk.BOTH, expand=True)

        before_box = ttk.LabelFrame(img_frame, text="Anh truoc xu ly (Original)")
        before_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.canvas_before = tk.Canvas(before_box, bg="#dddddd", width=CANVAS_W, height=CANVAS_H)
        self.canvas_before.pack(fill=tk.BOTH, expand=True)

        after_box = ttk.LabelFrame(img_frame, text="Anh sau xu ly (Result)")
        after_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        self.canvas_after = tk.Canvas(after_box, bg="#dddddd", width=CANVAS_W, height=CANVAS_H)
        self.canvas_after.pack(fill=tk.BOTH, expand=True)

        self.video_ctrl = ttk.Frame(center)
        ttk.Button(self.video_ctrl, text="Play", command=self.video_play).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(self.video_ctrl, text="Pause", command=self.video_pause).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(self.video_ctrl, text="Stop", command=self.video_stop).pack(side=tk.LEFT, padx=4, pady=4)

        result_box = ttk.LabelFrame(center, text="Ket qua thong ke")
        result_box.pack(fill=tk.BOTH, expand=False, pady=(6, 0))
        self.txt_result = tk.Text(result_box, height=10, font=("Consolas", 9))
        self.txt_result.pack(fill=tk.BOTH, expand=True)

        right = ttk.Frame(main, width=340)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0))

        notebook = ttk.Notebook(right)
        notebook.pack(fill=tk.BOTH, expand=True)

        self._build_tab_hsv(notebook)
        self._build_tab_histogram(notebook)
        self._build_tab_stats(notebook)
        self._build_tab_batch(notebook)
        self._build_tab_classify(notebook)

    def _build_tab_hsv(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="1. HSV Segmentation")

        ttk.Label(tab, text="Tuy chinh nguong HSV:", font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=8, pady=(10, 2))

        self.h_low = self._make_slider(tab, "H thap (0-179)", 0, 179, 0)
        self.h_high = self._make_slider(tab, "H cao (0-179)", 0, 179, 10)
        self.s_low = self._make_slider(tab, "S thap (0-255)", 0, 255, 70)
        self.s_high = self._make_slider(tab, "S cao (0-255)", 0, 255, 255)
        self.v_low = self._make_slider(tab, "V thap (0-255)", 0, 255, 50)
        self.v_high = self._make_slider(tab, "V cao (0-255)", 0, 255, 255)

        ttk.Button(tab, text="Ap dung phan doan HSV",
                   command=self.apply_hsv_segmentation).pack(fill=tk.X, padx=8, pady=10)

        ttk.Separator(tab).pack(fill=tk.X, pady=6, padx=8)
        ttk.Label(tab, text="Preset mau tham khao:", font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=8)
        presets = {
            "Do (Red)": (0, 10, 70, 255, 50, 255),
            "Vang (Yellow)": (20, 35, 70, 255, 70, 255),
            "Cam (Orange)": (5, 20, 100, 255, 100, 255),
            "Xanh la (Green)": (36, 85, 50, 255, 50, 255),
            "Tim (Purple)": (130, 160, 40, 255, 30, 255),
        }
        preset_var = tk.StringVar(value="Do (Red)")
        combo = ttk.Combobox(tab, textvariable=preset_var, values=list(presets.keys()), state="readonly")
        combo.pack(padx=8, fill=tk.X)

        def apply_preset(event=None):
            h_lo, h_hi, s_lo, s_hi, v_lo, v_hi = presets[preset_var.get()]
            self.h_low.set(h_lo); self.h_high.set(h_hi)
            self.s_low.set(s_lo); self.s_high.set(s_hi)
            self.v_low.set(v_lo); self.v_high.set(v_hi)

        combo.bind("<<ComboboxSelected>>", apply_preset)

    def _make_slider(self, parent, label, mn, mx, default):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=8, pady=3)
        var = tk.IntVar(value=default)
        lbl = ttk.Label(frame, text=f"{label}: {default}", width=20)
        lbl.pack(side=tk.LEFT)

        def on_change(v):
            lbl.config(text=f"{label}: {int(float(v))}")

        scale = ttk.Scale(frame, from_=mn, to=mx, orient=tk.HORIZONTAL, variable=var, command=on_change)
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return var

    def _build_tab_histogram(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="2. Color Histogram")

        ttk.Label(tab, text="So bins:").pack(anchor="w", padx=8, pady=(10, 2))
        self.hist_bins = tk.IntVar(value=32)
        ttk.Spinbox(tab, from_=8, to=256, increment=8, textvariable=self.hist_bins, width=10).pack(padx=8, anchor="w")

        self.hist_use_mask = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab, text="Chi tinh tren vung da phan doan (mask)",
                         variable=self.hist_use_mask).pack(anchor="w", padx=8, pady=(10, 0))

        ttk.Button(tab, text="Ve Color Histogram (H, S, V)",
                   command=self.show_histogram).pack(fill=tk.X, padx=8, pady=10)

        self.hist_canvas_frame = ttk.Frame(tab)
        self.hist_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    def _build_tab_stats(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="3. Thong ke")

        ttk.Label(tab, text="Tinh thong ke tren vung da phan doan\n(mask tu tab HSV, hoac toan anh neu\nchua phan doan).",
                  justify=tk.LEFT).pack(anchor="w", padx=8, pady=(10, 6))

        ttk.Button(tab, text="Tinh thong ke chi tiet",
                   command=self.run_statistics).pack(fill=tk.X, padx=8, pady=6)

    def _build_tab_batch(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="4. Xu ly hang loat")

        ttk.Label(tab, text="Xu ly toan bo anh trong thu muc da mo,\ntinh thong ke voi cung nguong HSV\ndang chinh o tab 1, xuat CSV.",
                  justify=tk.LEFT).pack(anchor="w", padx=8, pady=(10, 6))
        ttk.Button(tab, text="Chay xu ly hang loat...",
                   command=self.run_batch).pack(fill=tk.X, padx=8, pady=6)

        self.batch_progress = ttk.Progressbar(tab, mode="determinate")
        self.batch_progress.pack(fill=tk.X, padx=8, pady=6)
        self.batch_progress_label = ttk.Label(tab, text="")
        self.batch_progress_label.pack(anchor="w", padx=8)

    def _build_tab_classify(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="5. Phan loai")

        ttk.Label(
            tab,
            text="Buoc 1: Chon thu muc anh mau, cau truc:\n"
                 "  dataset/\n"
                 "    Apple/anh1.jpg, anh2.jpg...\n"
                 "    Banana/anh1.jpg...\n"
                 "(moi thu muc con la 1 nhan/loai trai cay)",
            justify=tk.LEFT,
        ).pack(anchor="w", padx=8, pady=(10, 6))

        ttk.Button(tab, text="Chon thu muc anh mau...",
                   command=self.build_reference_database).pack(fill=tk.X, padx=8, pady=4)

        self.classify_db_status = ttk.Label(tab, text="Chua co ngan hang dac trung.",
                                             foreground="#a05a00")
        self.classify_db_status.pack(anchor="w", padx=8, pady=(2, 10))

        ttk.Separator(tab).pack(fill=tk.X, pady=6, padx=8)

        ttk.Label(tab, text="Buoc 2: So lang gieng gan nhat (k):").pack(anchor="w", padx=8, pady=(6, 2))
        self.classify_k = tk.IntVar(value=3)
        ttk.Spinbox(tab, from_=1, to=9, textvariable=self.classify_k, width=10).pack(padx=8, anchor="w")

        ttk.Button(tab, text="Phan loai anh hien tai",
                   command=self.classify_current_image).pack(fill=tk.X, padx=8, pady=10)

    def _build_status_bar(self):
        self.status_var = tk.StringVar(value="San sang.")
        bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")
        bar.pack(side=tk.BOTTOM, fill=tk.X)

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
            self.txt_result.insert(tk.END, "=== DAC TRUNG HSV (mean/std) ===\n")
            self.txt_result.insert(tk.END, f"Ty le vung phat hien: {pct:.2f}%\n")
            self.txt_result.insert(tk.END, f"mean_H={feat[0]:.2f}  std_H={feat[1]:.2f}\n")
            self.txt_result.insert(tk.END, f"mean_S={feat[2]:.2f}  std_S={feat[3]:.2f}\n")
            self.txt_result.insert(tk.END, f"mean_V={feat[4]:.2f}  std_V={feat[5]:.2f}\n")
            self._set_status(f"Da phan doan HSV. Dien tich khop: {pct:.2f}%")

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
        ax.plot(hist_h, color="darkorange", label="H", linewidth=1.5)
        ax.plot(hist_s, color="seagreen", label="S", linewidth=1.5)
        ax.plot(hist_v, color="steelblue", label="V", linewidth=1.5)
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

    def run_statistics(self):
        if not self._check_image_loaded():
            return
        mask = self.current_mask
        stats, boxes = proc.compute_full_statistics(self.original_rgb, mask=mask)
        self.txt_result.delete("1.0", tk.END)
        self.txt_result.insert(tk.END, "=== THONG KE ANH ===\n")
        for k, v in stats.items():
            self.txt_result.insert(tk.END, f"{k:28s}: {v}\n")
        if boxes:
            self.txt_result.insert(tk.END, f"\nBounding boxes doi tuong ({len(boxes)}):\n")
            for i, b in enumerate(boxes, 1):
                self.txt_result.insert(tk.END, f"  #{i}: x={b[0]}, y={b[1]}, w={b[2]}, h={b[3]}\n")
        self._set_status("Da tinh thong ke.")

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

    # -------------------------------------------------------------- Tab 5
    def build_reference_database(self):
        folder = filedialog.askdirectory(
            title="Chon thu muc goc chua cac thu muc con theo loai trai cay")
        if not folder:
            return

        subfolders = [f for f in os.listdir(folder)
                      if os.path.isdir(os.path.join(folder, f))]
        if not subfolders:
            messagebox.showwarning(
                "Canh bao",
                "Thu muc nay khong co thu muc con nao.\n"
                "Can cau truc: dataset/Apple/, dataset/Banana/, ...")
            return

        self.classify_db_status.config(text="Dang xu ly, vui long doi...", foreground="#0066cc")
        self.root.update_idletasks()

        def worker():
            try:
                db_norm, norm_params = proc.build_pipeline_from_folder(
                    folder, max_images_per_class=20)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Loi", f"Khong xay dung duoc: {e}"))
                self.root.after(0, lambda: self.classify_db_status.config(
                    text="Loi khi xay dung ngan hang dac trung.", foreground="#cc0000"))
                return

            self.db_norm = db_norm
            self.norm_params = norm_params
            self.reference_folder = folder

            labels = list(db_norm.keys())
            counts = {lb: len(vecs) for lb, vecs in db_norm.items()}
            summary = ", ".join(f"{lb} ({counts[lb]} anh)" for lb in labels)

            self.root.after(0, lambda: self.classify_db_status.config(
                text=f"Da xay xong: {summary}", foreground="#0a7a1e"))
            self.root.after(0, lambda: self._set_status(
                f"Ngan hang dac trung san sang: {len(labels)} loai."))

        threading.Thread(target=worker, daemon=True).start()

    def classify_current_image(self):
        if not self._check_image_loaded():
            return
        if self.db_norm is None or self.norm_params is None:
            messagebox.showwarning(
                "Canh bao",
                "Vui long chon thu muc anh mau truoc (Buoc 1) de xay ngan hang dac trung.")
            return

        k = int(self.classify_k.get())
        label, confidence, dist_table = proc.classify_fruit(
            self.original_rgb, self.db_norm, self.norm_params, k=k)

        self.txt_result.delete("1.0", tk.END)
        self.txt_result.insert(tk.END, "=== KET QUA PHAN LOAI ===\n")
        self.txt_result.insert(tk.END, f"Du doan: {label}\n")
        self.txt_result.insert(tk.END, f"Do tin cay (trong {k} lang gieng gan nhat): {confidence*100:.0f}%\n\n")
        self.txt_result.insert(tk.END, "Khoang cach trung binh toi tung loai (cang nho cang giong):\n")
        for lb, d in sorted(dist_table.items(), key=lambda x: x[1]):
            self.txt_result.insert(tk.END, f"  {lb:<15s}: {d:.3f}\n")

        self._set_status(f"Phan loai: {label} (tin cay {confidence*100:.0f}%)")

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
            messagebox.showwarning("Canh bao", "Chua ve histogram nao de luu. Vao tab 2 va bam 've Color Histogram'.")
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
