# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

ROOT = Path.cwd()

block_cipher = None

excluded = [
    "gradio",
    "tensorflow",
    "keras",
    "pandas",
    "pyarrow",
    "matplotlib",
    "IPython",
    "jupyter",
    "notebook",
    "scipy",
    "torchvision",
    "cv2",
    "tkinter",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "win32com",
]

a = Analysis(
    ["package_app.py"],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        (str(ROOT / "checkpoints" / "anime_cyclegan" / "latest.pt"), "checkpoints/anime_cyclegan"),
    ],
    hiddenimports=[
        "torch",
        "numpy",
        "PIL",
        "anime_style.models",
        "anime_style.infer",
        "web_demo.app",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Photo2Anime",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Photo2Anime",
)
