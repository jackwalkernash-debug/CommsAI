from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

packages = [
    "faster_whisper",
    "ctranslate2",
    "tokenizers",
    "huggingface_hub",
    "av",
    "requests",
    "certifi",
    "charset_normalizer",
    "idna",
    "urllib3",
    "nvidia.cublas",
    "nvidia.cudnn",
]

for package in packages:
    try:
        package_datas, package_binaries, package_hidden = collect_all(package)
        datas += package_datas
        binaries += package_binaries
        hiddenimports += package_hidden
        print(f"Collected package: {package}")
    except Exception as exc:
        print(f"Warning: failed to collect {package}: {exc}")

a = Analysis(
    ["backend.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CommsAI.Backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="CommsAI.Backend",
)
