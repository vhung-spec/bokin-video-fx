"""A ready-made Tk activation screen for Bokin apps built with Tkinter.

    frame = ActivationFrame(root, "fx", on_activated=build_main_window)
    frame.pack(fill="both", expand=True)

It shows the machine code (with a copy button) and a field for the key; a valid key is stored
(store.activate) and on_activated() runs, so the app can build its real window. Apps with other
toolkits use bokin_license.status/activate and draw their own screen with the same texts.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from . import store
from .machine import LicenseError
from .products import display_name


class ActivationFrame(ttk.Frame):
    def __init__(self, master, product, on_activated, *, folder=None, public=None, **options):
        super().__init__(master, padding=24, **options)
        self.product, self.on_activated, self.folder, self.public = product, on_activated, folder, public
        state = store.status(product, folder, public=public)
        ttk.Label(self, text=f"Kích hoạt {display_name(product)}", font=("Segoe UI", 15, "bold")).pack(anchor="w")
        ttk.Label(self, text="App chỉ dùng được sau khi kích hoạt key cho máy này. Key gắn với từng máy.").pack(
            anchor="w", pady=(4, 16))
        ttk.Label(self, text="1. Gửi Mã máy này cho người bán:").pack(anchor="w")
        row = ttk.Frame(self)
        row.pack(fill="x", pady=(6, 14))
        self.machine = tk.StringVar(value=state["machine"])
        ttk.Entry(row, textvariable=self.machine, state="readonly", font=("Consolas", 13), width=24).pack(side="left", ipady=3)
        ttk.Button(row, text="Sao chép", command=self.copy_machine).pack(side="left", padx=(8, 0))
        ttk.Label(self, text="2. Dán key nhận được (BOKIN-…) rồi bấm Kích hoạt:").pack(anchor="w")
        self.key = tk.Text(self, height=4, width=62, wrap="char", font=("Consolas", 10), relief="solid", borderwidth=1)
        self.key.pack(fill="x", pady=(6, 12))
        row = ttk.Frame(self)
        row.pack(fill="x")
        ttk.Button(row, text="Kích hoạt", command=self.activate).pack(side="left")
        self.message = tk.StringVar(value="" if state["message"] == "Chưa kích hoạt." else state["message"])
        ttk.Label(row, textvariable=self.message, foreground="#2f5bd3", wraplength=380).pack(side="left", padx=(12, 0))

    def copy_machine(self):
        self.clipboard_clear()
        self.clipboard_append(self.machine.get())
        self.message.set("Đã chép Mã máy.")

    def activate(self):
        try:
            store.activate(self.key.get("1.0", "end"), self.product, self.folder, public=self.public)
        except LicenseError as error:
            self.message.set(str(error))
            return False
        self.destroy()
        self.on_activated()
        return True
