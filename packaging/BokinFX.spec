# PyInstaller recipe for Bokin FX: one folder, packed into a single installer by BokinFX.iss.
# Built by .github/workflows/bokin-fx.yml:
#   python packaging/make_icon.py
#   pyinstaller packaging/BokinFX.spec --noconfirm      -> dist/BokinFX/
import os

from PyInstaller.utils.hooks import collect_data_files

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))  # noqa: F821 (SPECPATH is set by PyInstaller)
FX = os.path.join(ROOT, "fx")
datas = [(os.path.join(FX, "index.html"), ".")]
if os.path.exists(os.path.join(FX, "version.txt")):
    datas.append((os.path.join(FX, "version.txt"), "."))

a = Analysis(
    [os.path.join(ROOT, "bokin_fx.py")],
    pathex=[ROOT],
    datas=datas + collect_data_files("imageio_ffmpeg"),  # the bundled ffmpeg.exe
    hiddenimports=["imageio_ffmpeg"],
    excludes=["tkinter", "matplotlib", "scipy", "pytest", "IPython"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="BokinFX",
          icon=os.path.join(SPECPATH, "icon.ico"), console=False, upx=False)  # noqa: F821
coll = COLLECT(exe, a.binaries, a.datas, name="BokinFX", upx=False)
