#!/usr/bin/env python3
"""Tkinter GUI for Metashape 360 -> COLMAP with optional SAM3 masks."""
from __future__ import annotations
import queue, subprocess, sys, threading, tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk


class ConverterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Metashape 360° → COLMAP + SAM3 masks")
        self.root.geometry("920x820")
        self.root.minsize(820, 720)
        self.process = None
        self.q = queue.Queue()
        self.images = tk.StringVar()
        self.masks = tk.StringVar()
        self.xml = tk.StringVar()
        self.ply = tk.StringVar()
        self.output = tk.StringVar(value="./colmap_dataset")
        self.crop = tk.StringVar(value="1920")
        self.fov = tk.StringVar(value="90")
        self.max_images = tk.StringVar(value="10000")
        self.yaw = tk.StringVar(value="0")
        self.workers = tk.StringVar(value="1")
        self.rotate = tk.BooleanVar()
        self.lichtfeld_axis = tk.BooleanVar(value=True)
        self._build()
        self.root.after(100, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build(self):
        m = ttk.Frame(self.root, padding=14)
        m.pack(fill="both", expand=True)
        m.columnconfigure(1, weight=1)
        m.rowconfigure(11, weight=1)
        ttk.Label(m, text="Metashape 360° → COLMAP", font=("TkDefaultFont", 16, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 12)
        )
        self._path(m, 1, "Input equirectangular images", self.images, "dir")
        self._path(m, 2, "SAM3 masks (optional)", self.masks, "dir")
        self._path(m, 3, "Metashape Camera.xml", self.xml, "file")
        self._path(m, 4, "PLY (optional)", self.ply, "ply")
        self._path(m, 5, "Output folder", self.output, "dir")

        o = ttk.LabelFrame(m, text="Conversion options", padding=10)
        o.grid(row=6, column=0, columnspan=3, sticky="ew", pady=10)
        for i in range(5):
            o.columnconfigure(i, weight=1)
        for i, (label, var) in enumerate((("Crop size", self.crop), ("FoV", self.fov), ("Max images", self.max_images), ("Yaw offset", self.yaw), ("Workers", self.workers))):
            ttk.Label(o, text=label).grid(row=0, column=i, sticky="w", padx=4)
            ttk.Entry(o, textvariable=var, width=14).grid(row=1, column=i, sticky="ew", padx=4)
        ttk.Checkbutton(o, text="Rotate scene 180° around Z", variable=self.rotate).grid(row=2, column=0, columnspan=2, sticky="w", padx=4, pady=(8, 0))
        ttk.Checkbutton(o, text="LichtFeld axis correction (X, -Y, -Z)", variable=self.lichtfeld_axis).grid(row=2, column=2, columnspan=3, sticky="w", padx=4, pady=(8, 0))
        ttk.Label(m, text="SAM3 masks use the same ERP → perspective projection.").grid(row=7, column=0, columnspan=3, sticky="w", pady=(0, 8))
        c = ttk.Frame(m)
        c.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        self.run_button = ttk.Button(c, text="▶ Run conversion", command=self.run)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(c, text="■ Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        ttk.Button(c, text="Clear log", command=self.clear).pack(side="right")
        ttk.Label(m, text="Output log").grid(row=9, column=0, columnspan=3, sticky="w")
        self.log = scrolledtext.ScrolledText(m, wrap="word", height=20, state="disabled")
        self.log.grid(row=11, column=0, columnspan=3, sticky="nsew", pady=(4, 0))

    def _path(self, parent, row, label, var, kind):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=8, pady=4)
        ttk.Button(parent, text="Browse...", command=lambda: self._browse(var, kind)).grid(row=row, column=2, pady=4)

    def _browse(self, var, kind):
        if kind == "dir":
            p = filedialog.askdirectory()
        elif kind == "ply":
            p = filedialog.askopenfilename(filetypes=[("PLY", "*.ply"), ("All", "*.*")])
        else:
            p = filedialog.askopenfilename(filetypes=[("XML", "*.xml"), ("All", "*.*")])
        if p:
            var.set(p)

    def _validate(self):
        if not Path(self.images.get()).expanduser().is_dir():
            messagebox.showerror("Input error", "Input image folder not found.")
            return False
        if not Path(self.xml.get()).expanduser().is_file():
            messagebox.showerror("Input error", "Camera.xml not found.")
            return False
        if self.ply.get().strip() and not Path(self.ply.get()).expanduser().is_file():
            messagebox.showerror("Input error", "PLY file not found.")
            return False
        if self.masks.get().strip() and not Path(self.masks.get()).expanduser().is_dir():
            messagebox.showerror("Input error", "SAM3 mask folder not found.")
            return False
        try:
            if int(self.crop.get()) <= 0 or not 0 < float(self.fov.get()) < 180 or int(self.max_images.get()) <= 0 or int(self.workers.get()) <= 0:
                raise ValueError
            float(self.yaw.get())
        except ValueError:
            messagebox.showerror("Input error", "Check numeric options.")
            return False
        Path(self.output.get()).expanduser().mkdir(parents=True, exist_ok=True)
        return True

    def run(self):
        if self.process is not None or not self._validate():
            return
        base = Path(__file__).parent
        wrapper = base / "convert_with_sam3_masks.py"
        cmd = [sys.executable, str(wrapper), "--images", str(Path(self.images.get()).expanduser()), "--xml", str(Path(self.xml.get()).expanduser()), "--output", str(Path(self.output.get()).expanduser()), "--crop-size", self.crop.get(), "--fov-deg", self.fov.get(), "--max-images", self.max_images.get(), "--yaw-offset", self.yaw.get(), "--num-workers", self.workers.get()]
        if self.masks.get().strip():
            cmd += ["--masks", str(Path(self.masks.get()).expanduser())]
        if self.ply.get().strip():
            cmd += ["--ply", str(Path(self.ply.get()).expanduser())]
        if self.rotate.get():
            cmd.append("--rotate-z180")
        if self.lichtfeld_axis.get():
            cmd.append("--lichtfeld-axis")
        self._write("$ " + " ".join(cmd) + "\n\n")
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        threading.Thread(target=self._worker, args=(cmd,), daemon=True).start()

    def _worker(self, cmd):
        try:
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert self.process.stdout
            for line in self.process.stdout:
                self.q.put(line)
            self.q.put(f"\nProcess finished with exit code {self.process.wait()}.\n")
        except Exception as exc:
            self.q.put(f"\nERROR: {exc}\n")
        finally:
            self.process = None
            self.q.put("__DONE__")

    def _poll(self):
        try:
            while True:
                line = self.q.get_nowait()
                if line == "__DONE__":
                    self.run_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                else:
                    self._write(line)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            self._write("\nStopping...\n")

    def clear(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _write(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _close(self):
        if self.process is not None and not messagebox.askyesno("Confirm exit", "Conversion is running. Stop and exit?"):
            return
        if self.process is not None:
            self.process.terminate()
        self.root.destroy()


def main():
    root = tk.Tk()
    ConverterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
