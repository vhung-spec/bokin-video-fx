"""Bokin license keys, version 2: one machine, one or more products, an optional expiry date.

A key is "BOKIN-" + base32 of a 22-byte payload and its 64-byte Ed25519 signature, in groups of 5:

    payload = version (2) | product mask (4 bytes) | machine digest (10) | expiry (2) | serial (4) | flags (1)

The expiry is days since 2020-01-01 (0 = never expires). The signature covers MESSAGE_PREFIX +
payload with the vendor's key (keycenter; never shipped); apps hold PUBLIC_KEY only. Version 1
keys (Bokin Studio Music 0.6-0.9: same container, version byte 1, no products) are recognised so
that app can keep accepting them through its own legacy check.
"""
from __future__ import annotations

import base64
import datetime
import re

from .ed25519 import sign, verify_signature
from .machine import LicenseError, machine_digest, machine_id
from .products import display_name, display_names, mask, products_of

KEY_PREFIX = "BOKIN"
VERSION = 2
MESSAGE_PREFIX = b"Bokin license v2\x00"
EPOCH = datetime.date(2020, 1, 1)
PAYLOAD_BYTES = 22
RAW_BYTES = PAYLOAD_BYTES + 64
PUBLIC_KEY = bytes.fromhex("a411bc6a3dd967b17c5c0066b3c7e151389f8f21b78928dd37be15e81a550a21")


def _raw(key):
    """The bytes inside a key's text (any version); LicenseError if it is not a Bokin key."""
    clean = re.sub(r"[^A-Z2-7]", "", str(key).upper())
    if clean.startswith(KEY_PREFIX):
        clean = clean[len(KEY_PREFIX):]
    try:
        raw = base64.b32decode(clean + "=" * (-len(clean) % 8))
    except ValueError:
        raise LicenseError("Key không đúng định dạng.") from None
    if not raw:
        raise LicenseError("Key không đúng định dạng.")
    return raw


def key_version(key):
    """1 (Bokin Studio Music 0.6-0.9), 2, or None when the text is not a Bokin key."""
    try:
        raw = _raw(key)
    except LicenseError:
        return None
    return raw[0] if raw[0] in (1, 2) else None


def normalize(key):
    """The key as stored: no whitespace."""
    return re.sub(r"\s+", "", str(key).strip())


def make_key(secret, machine, products, expires=None, serial=0, flags=0):
    """Vendor side (keycenter): a key for one machine and the given products, optionally until a date."""
    days = 0 if expires is None else (expires - EPOCH).days
    if not 0 <= days < 65536 or (expires is not None and days == 0):
        raise LicenseError("Ngày hết hạn không hợp lệ.")
    if not 0 <= int(serial) < 2 ** 32:
        raise LicenseError("Số khách phải từ 0 đến 4294967295.")
    payload = (bytes([VERSION]) + mask(products).to_bytes(4, "big") + machine_digest(machine)
               + days.to_bytes(2, "big") + int(serial).to_bytes(4, "big") + bytes([flags]))
    text = base64.b32encode(payload + sign(secret, MESSAGE_PREFIX + payload)).decode().rstrip("=")
    return KEY_PREFIX + "-" + "-".join(text[i:i + 5] for i in range(0, len(text), 5))


def read_key(key, public=None):
    """{"products", "machine_digest", "expires", "serial", "flags"} of a genuine version 2 key, without
    checking machine, product or date (the key center describes keys with it)."""
    raw = _raw(key)
    if raw[0] == 1:
        raise LicenseError("Đây là key đời cũ của Bokin Studio Music (0.6–0.9); app này cần key mới.")
    if len(raw) != RAW_BYTES or raw[0] != VERSION:
        raise LicenseError("Key không đúng định dạng.")
    payload, signature = raw[:PAYLOAD_BYTES], raw[PAYLOAD_BYTES:]
    if not verify_signature(public or PUBLIC_KEY, MESSAGE_PREFIX + payload, signature):
        raise LicenseError("Key không hợp lệ.")
    days = int.from_bytes(payload[15:17], "big")
    return {"products": products_of(int.from_bytes(payload[1:5], "big")), "machine_digest": payload[5:15],
            "expires": EPOCH + datetime.timedelta(days=days) if days else None,
            "serial": int.from_bytes(payload[17:21], "big"), "flags": payload[21]}


def check_key(key, product, machine=None, today=None, public=None):
    """Details of a key valid for this product on this machine today: {"products", "expires", "serial"};
    LicenseError (message for the customer) otherwise."""
    details = read_key(key, public)
    if product not in details["products"]:
        raise LicenseError(f"Key này không dành cho {display_name(product)} "
                           f"(key mở: {display_names(details['products']) or 'không có sản phẩm nào'}).")
    if details["machine_digest"] != machine_digest(machine or machine_id()):
        raise LicenseError("Key này dành cho máy khác. Hãy gửi đúng Mã máy hiện trên app.")
    expires = details["expires"]
    if expires is not None and (today or datetime.date.today()) > expires:
        raise LicenseError(f"Key đã hết hạn ngày {expires:%d/%m/%Y}.")
    return {"products": details["products"], "expires": expires, "serial": details["serial"]}
