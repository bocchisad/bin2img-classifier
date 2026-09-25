# Real binary dataset layout

Put real files here, then train:

```bash
python scripts/ingest_dataset.py --root data
python scripts/train.py --real-data data --mix-synthetic --per-class 40 --out artifacts/model.cbm
```

## Recommended structure

```
data/
  benign/                          → labeled Benign_Code
    putty.exe
    7z.exe
    notepadplusplus.exe
  malware/
    Ransomware_Encrypted/          → family label = folder name
      sample_a.bin
    Packed_PE/
      packed1.exe
    Downloader_TextHeavy/
      ...
    MyCustomFamily/                → any folder name becomes a class
      ...
```

## Rules

- Use **isolated VM / lab** for malware samples.
- Prefer diverse sizes and vendors for benign.
- Keep each class with **≥20–30 files** before expecting stable metrics.
- Synthetic-only training is fine for demos; real files need `--real-data`.

## Auto-fetch a safe demo set (recommended)

Does **not** download live malware. Uses official PuTTY builds + UPX + encrypted/URL overlays:

```bash
brew install upx   # once
python scripts/fetch_demo_dataset.py
python scripts/ingest_dataset.py --root data
python scripts/train.py --real-data data --mix-synthetic --per-class 40
```

| Folder | Source |
|--------|--------|
| `benign/` | Official PuTTY `.exe` |
| `malware/Packed_PE/` | Same files packed with **UPX** |
| `malware/Ransomware_Encrypted/` | XOR-encrypted copies (high entropy, not malware) |
| `malware/Downloader_TextHeavy/` | PE + large URL/string overlay |

## Safety

This project is for defensive research. Do not execute unknown binaries on your host OS.
For real malware corpora use an isolated VM / MalwareBazaar lab workflow.
