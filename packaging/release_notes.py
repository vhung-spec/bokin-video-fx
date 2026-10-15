"""Writes the notes of a release: python packaging/release_notes.py <version> <sha256> <out.md>

The SHA-256 line is also read by the app's updater when GitHub gives no digest for the file."""
import sys

version, sha, out = sys.argv[1:4]
with open(out, "w", encoding="utf-8") as f:
    f.write(f"""Bokin FX {version}: thêm hiệu ứng vào video loop và xuất video 30 giây hoặc 1 phút, vẫn loop liền mạch.
Giao diện giống Premiere: thư viện hiệu ứng, timeline, xem nhanh.

Tải **BokinFX-Setup-{version}.exe** bên dưới, bấm đúp để cài (không cần quyền quản trị). App đã cài tự cập nhật.

SHA-256: {sha}
""")
