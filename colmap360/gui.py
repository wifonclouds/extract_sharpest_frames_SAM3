#!/usr/bin/env python3
"""Tkinter GUI for the local Metashape 360 -> COLMAP converter."""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk


class ConverterGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Metashape 360° → COLMAP")
        self.root.geometry("900x720")
        self.root.minsize(780, 620)

        self.process: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[str] = queue.Queue()

        self.images_var = tk.StringVar()
        self.xml_var = tk.StringVar()
        self.ply_var = tk.StringVar()
        self.output_var = tk.StringVar(value="./colmap_dataset")
        self.crop_size_var = tk.StringVar(value="1920")
        self.fov_var = tk.StringVar(value="90")
        self.max_images_var = tk.StringVar(value="10000")
        self.yaw_var = tk.StringVar(value="0")
        self.workers_var = tk.StringVar(value="1")
        self.rotate_var = tk.BooleanVar(value=False)

        self._build_ui()
        self.root.after(100, self._poll_output)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=14)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(8, weight=1)

        ttk.Label(main, text="Metashape 360° → COLMAP", font=("TkDefaultFont", 16, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 14)
        )

        self._path_row(main, 1, "Input images", self.images_var, "dir")
        self._path_row(main, 2, "Metashape Camera.xml", self.xml_var, "file")
        self._path_row(main, 3, "PLY (optional)", self.ply_var, "ply")
        self._path_row(main, 4, "Output folder", self.output_var, "dir")

        options = ttk.LabelFrame(main, text="Conversion options", padding=10)
        options.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(10, 10))
        for col in range(6):
            options.columnconfigure(col, weight=1)

        fields = [
            ("Crop size", self.crop_size_var),
            ("FoV (deg)", self.fov_var),
            ("Max images", self.max_images_var),
            ("Yaw offset (deg)", self.yaw_var),
            ("Workers", self.workers_var),
        ]
        for i, (label, var) in enumerate(fields):
            ttk.Label(options, text=label).grid(row=0, column=i, sticky="w", padx=5)
            ttk.Entry(options, textvariable=var, width=14).grid(row=1, column=i, sticky="ew", padx=5)

        ttk.Checkbutton(
            options,
            text="Rotate scene 180° around Z",
            variable=self.rotate_var,
        ).grid(row=2, column=0, columnspan=3, sticky="w", padx=5, pady=(10, 0))

        ttk.Label(
            options,
            text="Six rectilinear views are generated per spherical camera.",
        ).grid(row=2, column=3, columnspan=3, sticky="e", padx=5, pady=(10, 0))

        controls = ttk.Frame(main)
        controls.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        self.run_button = ttk.Button(controls, text="▶ Run conversion", command=self.run)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(controls, text="■ Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        ttk.Button(controls, text="Clear log", command=self.clear_log).pack(side="right")

        ttk.Label(main, text="Output log").grid(row=7, column=0, columnspan=3, sticky="w")
        self.log = scrolledtext.ScrolledText(main, wrap="word", height=20, state="disabled")
        self.log.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=(4, 0))

    def _path_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        kind: str,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8, pady=5)
        ttk.Button(parent, text="Browse...", command=lambda: self._browse(variable, kind)).grid(
            row=row, column=2, pady=5
        )

    def _browse(self, variable: tk.StringVar, kind: str) -> None:
        if kind == "dir":
            path = filedialog.askdirectory()
        elif kind == "ply":
            path = filedialog.askopenfilename(filetypes=[("PLY files", "*.ply"), ("All files", "*.*")])
        else:
            path = filedialog.askopenfilename(
                filetypes=[("XML files", "*.xml"), ("All files", "*.*")]
            )
        if path:
            variable.set(path)

    def _validate(self) -> bool:
        images = Path(self.images_var.get()).expanduser()
        xml = Path(self.xml_var.get()).expanduser()
        output = Path(self.output_var.get()).expanduser()
        if not images.is_dir():
            messagebox.showerror("Input error", f"Input image folder not found:\n{images}")
            return False
        if not xml.is_file():
            messagebox.showerror("Input error", f"Camera.xml not found:\n{xml}")
            return False
        ply = self.ply_var.get().strip()
        if ply and not Path(ply).expanduser().is_file():
            messagebox.showerror("Input error", f"PLY file not found:\n{ply}")
            return False
        try:
            if int(self.crop_size_var.get()) <= 0:
                raise ValueError
            if not 0 < float(self.fov_var.get()) < 180:
                raise ValueError
            if int(self.max_images_var.get()) <= 0:
                raise ValueError
            int(self.workers_var.get())
            float(self.yaw_var.get())
        except ValueError:
            messagebox.showerror("Input error", "Check the numeric conversion options.")
            return False
        output.mkdir(parents=True, exist_ok=True)
        return True

    def run(self) -> None:
        if self.process is not None:
            return
        if not self._validate():
            return

        script = Path(__file__).with_name("metashape_360_to_colmap.py")
        cmd = [
            sys.executable,
            str(script),
            "--images", str(Path(self.images_var.get()).expanduser()),
            "--xml", str(Path(self.xml_var.get()).expanduser()),
            "--output", str(Path(self.output_var.get()).expanduser()),
            "--crop-size", self.crop_size_var.get(),
            "--fov-deg", self.fov_var.get(),
            "--max-images", self.max_images_var.get(),
            "--yaw-offset", self.yaw_var.get(),
            "--num-workers", self.workers_var.get(),
        ]
        ply = self.ply_var.get().strip()
        if ply:
            cmd += ["--ply", str(Path(ply).expanduser())]
        if self.rotate_var.get():
            cmd.append("--rotate-z180")

        self._write("$ " + " ".join(cmd) + "\n\n")
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        threading.Thread(target=self._worker, args=(cmd,), daemon=True).start()

    def _worker(self, cmd: list[str]) -> None:
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.output_queue.put(line)
            code = self.process.wait()
            self.output_queue.put(f"\nProcess finished with exit code {code}.\n")
        except Exception as exc:
            self.output_queue.put(f"\nERROR: {exc}\n")
        finally:
            self.process = None
            self.output_queue.put("__PROCESS_DONE__")

    def _poll_output(self) -> None:
        try:
            while True:
                line = self.output_queue.get_nowait()
                if line == "__PROCESS_DONE__":
                    self.run_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                else:
                    self._write(line)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_output)

    def stop(self) -> None:
        if self.process is not None:
            self.process.terminate()
            self._write("\nStopping conversion...\n")

    def clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _write(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _on_close(self) -> None:
        if self.process is not None:
            if not messagebox.askyesno("Confirm exit", "A conversion is running. Stop it and exit?"):
                return
            self.process.terminate()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    ConverterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
