"""The machine code: 16 characters from Windows' MachineGuid (MAC address elsewhere), e.g. ABCD-EFGH-IJKL-MNOP.

The same formula as Bokin Studio Music 0.6-0.9, so a customer's machine code never changes between
the Bokin apps or versions: the customer sends it once and every key for that computer uses it.
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
import uuid


class LicenseError(ValueError):
    """A key or machine code that cannot be used; the message is for the customer (Vietnamese)."""


def machine_id():
    raw = None
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography", 0,
                                winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
                raw = winreg.QueryValueEx(key, "MachineGuid")[0]
        except OSError:
            raw = None
    return machine_code(raw or f"{uuid.getnode():012x}")


def machine_code(raw):
    """The code for a raw machine identifier (exposed for tests and the key center)."""
    code = base64.b32encode(hashlib.sha256(b"bokin-machine:" + raw.strip().lower().encode()).digest()[:10]).decode()
    return "-".join(code[i:i + 4] for i in range(0, 16, 4))


def clean_machine(text):
    """ABCD-EFGH-IJKL-MNOP from whatever was pasted (spaces, lower case, missing dashes), or None."""
    code = re.sub(r"[^A-Z2-7]", "", str(text).upper())
    return "-".join(code[i:i + 4] for i in range(0, 16, 4)) if len(code) == 16 else None


def machine_digest(code):
    clean = re.sub(r"[^A-Z2-7]", "", str(code).upper())
    if len(clean) != 16:
        raise LicenseError("Mã máy phải có 16 ký tự (A–Z, 2–7).")
    return base64.b32decode(clean)
