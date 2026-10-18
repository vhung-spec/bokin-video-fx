"""Bokin License: one licensing system for every Bokin app (offline, machine-bound, signed keys).

    import bokin_license
    state = bokin_license.status("fx")              # {"licensed", "message", "machine", ...}
    if not state["licensed"]:
        bokin_license.activate(key_text, "fx")      # LicenseError with a message for the customer

Pure Python (standard library only, Python 3.8+), so it can be copied into any app, PyInstaller
builds included. See docs/INTEGRATION.md. The vendor side (signing key, issuing keys, install
codes) is the separate keycenter package, which never ships to customers.
"""
from __future__ import annotations

from .keys import check_key, key_version, read_key
from .machine import LicenseError, clean_machine, machine_id
from .products import PRODUCTS, display_name
from .store import activate, describe, status, stored_keys

__version__ = "1.0.0"
__all__ = ["LicenseError", "PRODUCTS", "activate", "check_key", "clean_machine", "describe", "display_name",
           "key_version", "machine_id", "read_key", "status", "stored_keys"]
