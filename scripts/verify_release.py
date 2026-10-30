#!/usr/bin/env python3
"""Check the complete release manifest, or verify a ZIP without extracting it."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', type=Path, nargs='?', default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    target = args.target.resolve()
    errors: list[str] = []
    if target.is_dir():
        manifest = json.loads((target/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
        for record in manifest['files']:
            path = (target/record['path']).resolve()
            if not path.is_relative_to(target) or not path.is_file():
                errors.append('missing or unsafe: '+record['path'])
                continue
            data = path.read_bytes()
            if len(data) != record['bytes'] or sha(data) != record['sha256']:
                errors.append('changed: '+record['path'])
    else:
        with zipfile.ZipFile(target) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                errors.append('duplicate member names')
            candidates = [x for x in names if x.endswith('/RELEASE_MANIFEST.json')]
            if len(candidates) != 1:
                raise ValueError('expected one release manifest')
            base = candidates[0].removesuffix('RELEASE_MANIFEST.json')
            manifest = json.loads(archive.read(candidates[0]))
            for record in manifest['files']:
                member = base+record['path']
                if member not in names:
                    errors.append('missing: '+record['path'])
                    continue
                data = archive.read(member)
                if len(data) != record['bytes'] or sha(data) != record['sha256']:
                    errors.append('changed: '+record['path'])
            expected = {base+record['path'] for record in manifest['files']} | {base+'RELEASE_MANIFEST.json'}
            extras = {x for x in names if not x.endswith('/')} - expected
            errors.extend('unlisted: '+x for x in sorted(extras))
            bad = archive.testzip()
            if bad:
                errors.append('CRC failure: '+bad)
    result = {'build': manifest['build'], 'checked_files': len(manifest['files']), 'errors': errors, 'ok': not errors}
    print(json.dumps(result, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
