"""Writes the app icon (brand/: the "Fx" tile of the Bokin family) into packaging/icon.ico. Run before PyInstaller."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "brand"))
import make_icons  # noqa: E402

OUT = os.path.join(ROOT, "packaging", "icon.ico")

if __name__ == "__main__":
    print(make_icons.write_ico("fx", OUT))
