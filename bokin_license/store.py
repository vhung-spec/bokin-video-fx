"""Activated keys on the customer's computer, shared by every Bokin app.

%APPDATA%\\BokinLicense\\keys.txt holds one key per line (per Windows user). A bundle key activated
in one app therefore unlocks the other products it lists on that computer too.
"""
from __future__ import annotations

import os
from pathlib import Path

from .keys import check_key, normalize, read_key
from .machine import LicenseError, machine_id
from .products import display_name


def default_folder():
    return Path(os.environ.get("APPDATA") or Path.home()) / "BokinLicense"


def keys_file(folder=None):
    return Path(folder or default_folder()) / "keys.txt"


def stored_keys(folder=None):
    try:
        lines = keys_file(folder).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [line.strip() for line in lines if line.strip()]


def status(product, folder=None, machine=None, today=None, public=None):
    """{"licensed", "message", "machine", "product", "expires", "serial", "products"} for this product:
    the best stored key (never expiring first, then the latest expiry); the reason otherwise."""
    machine = machine or machine_id()
    result = {"licensed": False, "machine": machine, "product": product, "expires": None, "serial": None,
              "products": []}
    best, reasons = None, []
    for key in stored_keys(folder):
        try:
            details = check_key(key, product, machine, today, public)
        except LicenseError as error:
            reasons.append(str(error))
            continue
        rank = (details["expires"] is None, details["expires"] or 0)
        if best is None or rank > best[0]:
            best = (rank, details)
    if best is None:
        # Expired keys explain more than keys for other products or machines.
        reason = next((r for r in reasons if "hết hạn" in r), None)
        return {**result, "message": f"Chưa kích hoạt: {reason}" if reason else "Chưa kích hoạt."}
    details = best[1]
    until = f" đến {details['expires']:%d/%m/%Y}" if details["expires"] else " vĩnh viễn"
    return {**result, **details, "licensed": True, "message": f"Đã kích hoạt{until}."}


def activate(key, product, folder=None, machine=None, today=None, public=None):
    """Verify a key for this product on this machine and keep it; returns its details. A rejected key
    is never saved."""
    details = check_key(key, product, machine, today, public)
    path = keys_file(folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = normalize(key)
    keys = stored_keys(folder)
    if key not in keys:
        path.write_text("\n".join(keys + [key]) + "\n", encoding="utf-8")
    return details


def describe(key, public=None):
    """Readable summary of a key (key center, support): products, expiry, serial; no machine check."""
    details = read_key(key, public)
    until = f"đến {details['expires']:%d/%m/%Y}" if details["expires"] else "vĩnh viễn"
    return f"{', '.join(display_name(p) for p in details['products'])} · {until} · số {details['serial']}"
