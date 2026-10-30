from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import uuid


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':'))


def new_id() -> str:
    return uuid.uuid4().hex


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(dumps(value)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> Any:
    def bad_constant(value):
        raise ValueError(f'nonfinite JSON number: {value}')
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=bad_constant)


def contained(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise ValueError('require a relative artifact path')
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('artifact path escapes session')
    return path


def artifact(root: Path, path: Path, role: str) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('artifact must be a real file inside the session')
    return dict(path=path.relative_to(root).as_posix(),
                size_bytes=path.stat().st_size, role=role)


def artifact_path(root: Path, value: dict[str, Any]) -> Path:
    if not isinstance(value, dict) or not {'path', 'size_bytes', 'role'} <= value.keys():
        raise ValueError('expected a verified artifact reference')
    path = contained(root, value['path'])
    if not path.is_file() or path.stat().st_size != value['size_bytes']:
        raise ValueError('artifact size changed or file is missing')
    return path


def pointer(value: Any, path: str) -> Any:
    if not path:
        return value
    if not path.startswith('/'):
        raise ValueError('JSON pointer must start with /')
    for token in path[1:].split('/'):
        key = token.replace('~1', '/').replace('~0', '~')
        if isinstance(value, list):
            if not key.isdigit() or (len(key) > 1 and key.startswith('0')):
                raise ValueError('invalid list index in JSON pointer')
            value = value[int(key)]
        elif isinstance(value, dict):
            value = value[key]
        else:
            raise ValueError('JSON pointer traverses a scalar')
    return value


@contextmanager
def exclusive_lock(path: Path):
    """POSIX advisory lock; a crash releases it. Linux/macOS/WSL are supported."""
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError('Use Linux, macOS or WSL for the agent execution lock') from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another runner holds this session') from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
