"""Writes what a release carries besides its installer:

  python packaging/release_notes.py <installer> <version> <folder>

- <folder>/notes.md: the release notes, with a SHA-256 line (the app's updater reads it when GitHub's API gives no
  digest for the file)
- <folder>/bokin-fx-latest.json (fx/updater.py, MANIFEST): the version, the installer's name, size and SHA-256. The
  installed apps read it first, from the newest release, as a plain download: not through GitHub's API and its limit
  of 60 requests an hour for each Internet connection."""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fx import updater  # noqa: E402


def write(installer, version, folder):
    name, size = os.path.basename(installer), os.path.getsize(installer)
    if not updater.INSTALLER.match(name) or not updater.parse_version(version):
        raise SystemExit(f"not an installer of a version: {name} {version}")
    digest = hashlib.sha256()
    with open(installer, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    sha = digest.hexdigest()
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "notes.md"), "w", encoding="utf-8") as f:
        f.write(f"""Bokin FX {version}: thêm hiệu ứng vào video loop và xuất video 30 giây hoặc 1 phút, vẫn loop liền mạch.
Giao diện giống Premiere: thư viện hiệu ứng, timeline, xem nhanh.

Tải **{name}** bên dưới, bấm đúp để cài (không cần quyền quản trị). App đã cài tự cập nhật.

SHA-256: {sha}
""")
    with open(os.path.join(folder, updater.MANIFEST), "w", encoding="utf-8") as f:
        json.dump({"version": version, "name": name, "size": size, "sha256": sha}, f)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    write(*sys.argv[1:4])
