# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


analyzer = Analysis(
    ["analyze.py"],
    pathex=[],
    binaries=[],
    datas=[
        ("eBird_taxonomy_codes_2021E.json", "."),
        ("checkpoints", "checkpoints"),
        ("example/soundscape.wav", "example"),
        ("example/species_list.txt", "example"),
        ("labels", "labels"),
        ("gui", "gui"),
    ],
    hiddenimports=[],
    hookspath=["extra-hooks"],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
analyzer_pyz = PYZ(analyzer.pure, analyzer.zipped_data, cipher=block_cipher)

analyzer_exe = EXE(
    analyzer_pyz,
    analyzer.scripts,
    [],
    exclude_binaries=True,
    name="BirdNET-Analyzer",
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
    icon=["gui\\img\\birdnet-icon.ico"],
)

gui = Analysis(
    ["gui.py"],
    pathex=[],
    binaries=[],
    datas=[
        ("eBird_taxonomy_codes_2021E.json", "."),
        ("checkpoints", "checkpoints"),
        ("example/soundscape.wav", "example"),
        ("example/species_list.txt", "example"),
        ("labels", "labels"),
        ("gui", "gui"),
        ("gui-settings.json", "."),
        ("lang", "lang")
    ],
    hiddenimports=[],
    hookspath=["extra-hooks"],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    module_collection_mode={"gradio": "py", "tensorflow": "py"},  # Collect gradio package as source .py files
)
gui_pyz = PYZ(gui.pure, gui.zipped_data, cipher=block_cipher)

splash = Splash(
    'gui/img/birdnet_logo_no_transparent.png',
    gui.binaries,
    gui.datas,
    text_pos=None,
    text_size=12,
    minify_script=True,
    always_on_top=True,
)

gui_exe = EXE(
    gui_pyz,
    gui.scripts,
    splash,
    [],
    exclude_binaries=True,
    name="BirdNET-Analyzer-GUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=["gui\\img\\birdnet-icon.ico"],
)


coll = COLLECT(
    analyzer_exe,
    analyzer.binaries,
    analyzer.zipfiles,
    analyzer.datas,
    splash.binaries,
    gui_exe,
    gui.binaries,
    gui.zipfiles,
    gui.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="BirdNET-Analyzer",
)
