"""Safely stage multi-file scientific input bundles without flattening sidecars."""
from __future__ import annotations

from pathlib import PurePosixPath
import stat
import zipfile

from .util import artifact, artifact_path


def unpack(ctx, args):
    source = artifact_path(ctx.root, args['archive'])
    max_bytes = args.get('max_bytes', 16*1024**3)
    max_files = args.get('max_files', 20000)
    if type(max_bytes) is not int or not 1 <= max_bytes <= 1024**4 or type(max_files) is not int or not 1 <= max_files <= 100000:
        raise ValueError('invalid archive size/file budget')
    suffixes = args.get('extensions', [])
    if not isinstance(suffixes, list) or any(not isinstance(s, str) or not s.startswith('.') for s in suffixes):
        raise ValueError('extensions must be a list such as [.cbf, .sfrm]')
    destination = ctx.directory/'bundle'
    destination.mkdir()
    with zipfile.ZipFile(source) as archive:
        entries = archive.infolist()
        seen, files, declared_total = set(), [], 0
        for info in entries:
            path = PurePosixPath(info.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in info.filename or not path.parts or ':' in path.parts[0]:
                raise ValueError('unsafe archive member path')
            normalized = path.as_posix().casefold()
            if normalized in seen:
                raise ValueError('duplicate/case-colliding archive member')
            seen.add(normalized)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise ValueError('archive symlinks and special files are forbidden')
            if info.flag_bits & 1:
                raise ValueError('encrypted scientific bundles are unsupported')
            if info.is_dir():
                continue
            declared_total += info.file_size
            files.append(info)
        if len(files) > max_files or declared_total > max_bytes:
            raise ValueError('archive exceeds its declared resource budget')
        used = 0
        for info in files:
            target = destination.joinpath(*PurePosixPath(info.filename).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as incoming, target.open('xb') as outgoing:
                while data := incoming.read(1024*1024):
                    used += len(data)
                    if used > max_bytes:
                        raise ValueError('archive exceeded its actual decompressed byte budget')
                    if ctx.store.get('cancel_requested'):
                        from .process import Cancelled
                        raise Cancelled('cancelled during archive extraction')
                    outgoing.write(data)
    output = {p.relative_to(destination).as_posix(): artifact(ctx.root, p, 'staged_input')
              for p in sorted(destination.rglob('*')) if p.is_file()}
    selected = [a for name, a in output.items() if not suffixes or PurePosixPath(name).suffix.lower() in suffixes]
    return dict(files=output, selected=selected, extracted_bytes=used, file_count=len(output),
                semantics='input_staging_not_scientific_processing')
