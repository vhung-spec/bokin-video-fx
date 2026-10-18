"""Bokin FX's licence: the app works on a computer once a key for it (product "fx") is activated.

The keys are Bokin License keys (the vendored bokin_license package, copied unchanged): signed by the owner's key
center, bound to the computer's machine code, maybe with an end date, kept per Windows user in
%APPDATA%\\BokinLicense\\keys.txt (a key for several Bokin apps, activated in one of them, opens the others too).
Until one is there the page shows the activation screen (the machine code to send to the seller, a field for the
key) and the server refuses everything else (server.FREE_GET, FREE_POST). There is no switch that skips the check.
"""
import threading
import time

import bokin_license

PRODUCT = "fx"
RECHECK = 60  # seconds: a key with an end date can run out while the app is open
SOON = 14  # days before its end date the page warns

_lock = threading.Lock()
_last = (0.0, None)


def status(fresh=False):
    """bokin_license.status() for this app, read again at most every RECHECK seconds (or now: `fresh`)."""
    global _last
    with _lock:
        at, state = _last
        if fresh or state is None or time.monotonic() - at > RECHECK:
            state = bokin_license.status(PRODUCT)
            _last = (time.monotonic(), state)
        return state


def licensed(fresh=False):
    return status(fresh)["licensed"]


def activate(text):
    """Keeps a key for this computer and app; LicenseError (its message is for the customer) otherwise."""
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        raise bokin_license.LicenseError("Hãy dán key nhận được từ người bán.")
    bokin_license.activate(text, PRODUCT)
    return view(fresh=True)


def view(fresh=False):
    """What the page shows: licensed or why not, the machine code, the end date, the apps the key opens."""
    s = status(fresh)
    return {"licensed": s["licensed"], "message": s["message"], "machine": s["machine"],
            "expires": s["expires"].isoformat() if s.get("expires") else None, "soon": SOON,
            "products": [bokin_license.display_name(p) for p in s.get("products") or []]}
