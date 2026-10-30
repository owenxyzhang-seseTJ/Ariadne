#!/usr/bin/env python3
"""Create a sorted source ZIP and per-file size manifest (no runtime binaries)."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

EXCLUDED_DIRS = {'.git', '__pycache__', '.pytest_cache', '.ruff_cache', '.venv', 'latex-build',
                 'jobs', 'raw_data', 'sessions'}
FONT_SUFFIXES = {'.ttf', '.otf', '.woff', '.woff2', '.ttc'}


def source_files(root: Path):
    for p in sorted(root.rglob('*')):
        relative = p.relative_to(root)
        if any(x in EXCLUDED_DIRS or x.endswith('.egg-info') for x in relative.parts):
            continue
        if relative.parts[:2] == ('python', 'build'):
            continue
        if p.is_symlink():
            raise ValueError(f'symlink in release: {relative}')
        if not p.is_file() or p.name == 'RELEASE_MANIFEST.json' or p.suffix in FONT_SUFFIXES:
            continue
        if p.name == '.env' or p.name.startswith('.env.') or p.name in {'runtime.json', 'core'} or p.name.startswith('core.'):
            continue
        if p.suffix in {'.sqlite', '.sqlite-wal', '.sqlite-shm', '.pem', '.key'}:
            continue
        if p.suffix in {'.pyc', '.pyo', '.aux', '.toc', '.out', '.fls', '.fdb_latexmk'} or p.name == '.DS_Store':
            continue
        yield p, relative.as_posix()


def make(root: Path, destination: Path):
    root, destination = root.resolve(), destination.resolve()
    if destination.is_relative_to(root):
        raise ValueError('ZIP output must be outside the source tree')
    files = list(source_files(root))
    records = []
    for path, name in files:
        records.append({'path': name, 'bytes': path.stat().st_size})
    manifest = {'schema': 'ariadne-release-manifest/2', 'build': '0.7.0-rebuild-20261030',
                'generated_utc': datetime.now(timezone.utc).isoformat(), 'files': records}
    manifest_path = root/'RELEASE_MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path, name in files+[(manifest_path, 'RELEASE_MANIFEST.json')]:
            entry = zipfile.ZipInfo('Ariadne/'+name, (2026, 10, 30, 0, 0, 0))
            entry.external_attr = (0o100755 if path.suffix in {'.sh'} else 0o100644) << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(entry, path.read_bytes())
    with zipfile.ZipFile(destination) as z:
        if z.testzip():
            raise RuntimeError('ZIP CRC check failed')
    result = {'archive': str(destination), 'bytes': destination.stat().st_size,
              'files': len(files)+1}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('destination', type=Path)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    make(a.root, a.destination)
