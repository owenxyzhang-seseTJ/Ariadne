"""Bounded process invocation: argv only, isolated attempts, file-backed logs."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Callable

from .util import atomic_json, now, file_hash


class Cancelled(RuntimeError):
    pass


class NeedsReview(RuntimeError):
    pass


def executable(name: str, mapping: dict[str, str]) -> str:
    configured = mapping.get(name, name)
    resolved = shutil.which(configured)
    if not resolved:
        raise RuntimeError(f'Required executable not found: {name}. Configure executables.{name}.')
    return resolved


def run_process(argv: list[str], directory: Path, timeout: float, cancelled: Callable[[], bool],
                on_pid: Callable[[int], None] | None = None, label: str = 'process',
                log_limit_bytes: int = 256*1024*1024) -> dict:
    if not argv or not all(isinstance(x, str) and '\x00' not in x for x in argv):
        raise ValueError('argv must be a list of strings')
    if timeout <= 0:
        raise TimeoutError('time budget exhausted')
    if cancelled():
        raise Cancelled('cancellation requested before process launch')
    atomic_json(directory/(label+'.command.json'), {'argv': argv, 'started': now(), 'timeout': timeout, 'executable_sha256': file_hash(Path(argv[0]).resolve())})
    stdout = directory/(label+'.stdout.log')
    stderr = directory/(label+'.stderr.log')
    started = time.monotonic()
    with stdout.open('wb') as out, stderr.open('wb') as err:
        child = subprocess.Popen(argv, cwd=directory, stdout=out, stderr=err,
                                 stdin=subprocess.DEVNULL, shell=False, start_new_session=True)
        try:
            if on_pid:
                on_pid(child.pid)
            while child.poll() is None:
                if cancelled():
                    raise Cancelled('cancellation requested')
                if time.monotonic()-started > timeout:
                    raise TimeoutError('external calculation exceeded its time budget')
                if stdout.stat().st_size+stderr.stat().st_size > log_limit_bytes:
                    raise RuntimeError('external logs exceeded the storage budget')
                time.sleep(.05)
            if stdout.stat().st_size+stderr.stat().st_size > log_limit_bytes:
                raise RuntimeError('external logs exceeded the storage budget')
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=2.)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
            raise
    result = dict(returncode=child.returncode, elapsed_seconds=time.monotonic()-started,
                  stdout=stdout.name, stderr=stderr.name)
    atomic_json(directory/(label+'.process.json'), result)
    if child.returncode:
        with stderr.open('rb') as stream:
            stream.seek(max(0, stderr.stat().st_size-3000))
            tail = stream.read().decode(errors='replace')
        raise RuntimeError(f'{Path(argv[0]).name} exited {child.returncode}: {tail}')
    return result
