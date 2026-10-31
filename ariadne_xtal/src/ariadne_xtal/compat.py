"""Explicit subprocess bridge to separately installed SHELXT and SHELXL.

This module does not contain, download, or emulate an official executable.  The
official process receives unmodified instruction and reflection bytes.  The
isolated working directory protects source files from ordinary program output;
it is not an operating-system security sandbox for an untrusted executable.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time
from typing import Sequence

from .shelx import ShelxDocument


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"Input file leaves the input project directory: {path.name}")
    if not resolved.is_file():
        raise ValueError(f"Input is not a regular file: {path.name}")
    return resolved


def _input_files(input_path: Path, auxiliary_paths: Sequence[str | Path] | None,
                 engine: str) -> list[tuple[Path, Path]]:
    root = input_path.parent.resolve()
    initial = _inside(input_path, root)
    result: dict[Path, Path] = {Path(input_path.stem + ".ins"): initial}
    visited: set[Path] = set()
    active: set[Path] = set()

    def visit(source: Path) -> None:
        resolved = _inside(source, root)
        if resolved in active:
            raise ValueError(f"Cyclic SHELX include reference involving {source.name}")
        if resolved in visited:
            return
        if len(visited) >= 1024:
            raise ValueError("More than 1024 include files")
        active.add(resolved)
        document = ShelxDocument.from_path(resolved)
        for instruction in document.instructions:
            if instruction.kind != "include" or not instruction.active:
                continue
            name = instruction.logical_text.lstrip("+").strip()
            if not name or "\x00" in name:
                raise ValueError("Empty or invalid SHELX include filename")
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("SHELX include paths must stay inside the input project directory")
            # SHELX opens filenames relative to the process cwd, including nested
            # includes. Preserve that layout under the isolated run directory.
            include = _inside(root / relative, root)
            if relative in result and result[relative] != include:
                raise ValueError(f"Conflicting include target {relative}")
            result[relative] = include
            visit(include)
        active.remove(resolved)
        visited.add(resolved)

    visit(initial)
    source_doc = ShelxDocument.from_path(initial)
    requires_hkl = engine == "shelxt" or not any(
        x.keyword == "HKLF" and x.arguments and x.arguments[0].startswith("-")
        for x in source_doc.instructions if x.active)
    hkl_path = root / (input_path.stem + ".hkl")
    if requires_hkl:
        if not hkl_path.exists():
            raise FileNotFoundError(f"Required companion reflection file is missing: {hkl_path.name}")
        result[Path(hkl_path.name)] = _inside(hkl_path, root)
    fab_path = root / (input_path.stem + ".fab")
    if fab_path.exists():
        result[Path(fab_path.name)] = _inside(fab_path, root)
    if any(x.keyword == "ABIN" and x.active for x in source_doc.instructions) and not fab_path.exists():
        raise FileNotFoundError(f"ABIN requires companion partial-structure-factor file {fab_path.name}")
    for extra in auxiliary_paths or []:
        supplied = Path(extra)
        source = _inside(supplied if supplied.is_absolute() else root / supplied, root)
        relative = source.relative_to(root)
        if relative in result and result[relative] != source:
            raise ValueError(f"Conflicting auxiliary target {relative}")
        result[relative] = source
    return list(result.items())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stop_process_tree(process: subprocess.Popen) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        process.kill()
    process.wait()


def run_legacy(engine: str, executable: str | Path, input_path: str | Path,
               output_dir: str | Path, timeout: float = 300,
               extra_args: Sequence[str] | None = None, *,
               auxiliary_paths: Sequence[str | Path] | None = None) -> dict:
    """Execute an explicitly selected installed backend and return provenance.

    Parameters
    ----------
    engine : {"shelxt", "shelxl"}
        Explicit routing, never inferred from native failure.
    executable : absolute path
        A caller-approved executable. Agent front ends must maintain their own
        allowlist; this function deliberately does not search PATH.
    input_path : INS or RES
        RES is copied to the same-stem INS name in the run directory. Original
        bytes, all includes and HKL/FAB companions are preserved.
    output_dir : directory
        A new child run directory is always allocated, never reused.
    extra_args : sequence of str
        Passed literally after the stem using argv, without a command shell.
    auxiliary_paths : sequence of paths
        Additional input files, restricted to the main input file's directory
        tree. They are copied without interpreting their contents.

    Returns
    -------
    dict
        Process status, return code, elapsed time, input/output hashes, working
        directory and manifest path. ``completed`` means process exit code 0;
        it does not certify a correct or converged crystal structure.
    """
    if engine not in {"shelxt", "shelxl"}:
        raise ValueError("engine must be explicitly 'shelxt' or 'shelxl'")
    supplied_executable = Path(executable).expanduser()
    if not supplied_executable.is_absolute():
        raise ValueError("Legacy executable must be an explicit absolute path")
    executable_path = supplied_executable.resolve(strict=True)
    if not executable_path.is_file() or not os.access(executable_path, os.X_OK):
        raise ValueError("Legacy executable is not an executable regular file")
    source = Path(input_path).expanduser().absolute()
    if source.suffix.lower() not in {".ins", ".res"}:
        raise ValueError("Legacy input must have .ins or .res extension")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,120}", source.stem):
        raise ValueError("Legacy stem must start with alphanumeric and contain only alphanumeric, dot, underscore or hyphen")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    args = list(extra_args or [])
    if any(not isinstance(arg,str) or "\x00" in arg for arg in args):
        raise ValueError("extra_args must contain strings without NUL")
    source_files = _input_files(source, auxiliary_paths, engine)
    destination = Path(output_dir).expanduser().absolute()
    destination.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"{engine}-{source.stem}-",dir=destination))
    inputs = []
    for relative,original in source_files:
        target = run_dir / relative
        target.parent.mkdir(parents=True,exist_ok=True)
        before = sha256_file(original)
        shutil.copyfile(original,target)
        copied = sha256_file(target)
        if before != copied:
            raise OSError(f"Input changed while copying: {original.name}")
        inputs.append({"source":str(original),"name":relative.as_posix(),
                       "sha256":copied,"size_bytes":target.stat().st_size})
    argv = [str(executable_path), source.stem, *args]
    manifest = {
        "schema":"ariadne_xtal.legacy_run/1", "engine":engine,
        "executable":str(executable_path),"executable_sha256":sha256_file(executable_path),
        "argv":argv,"working_directory":str(run_dir),"started_utc":_utc_now(),
        "status":"starting", "returncode":None, "inputs":inputs,
        "outputs":[],"scientific_validation_performed":False,
        "input_instruction_semantics":"delegated without rewriting to selected official executable",
        "timeout_seconds":timeout,
        "execution_environment":{name:os.environ[name] for name in ["OMP_NUM_THREADS","MKL_NUM_THREADS"] if name in os.environ},
    }
    start = time.monotonic()
    with (run_dir / "stdout.log").open("wb") as stdout, (run_dir / "stderr.log").open("wb") as stderr:
        try:
            process = subprocess.Popen(argv,cwd=run_dir,stdin=subprocess.DEVNULL,
                                       stdout=stdout,stderr=stderr,shell=False,
                                       start_new_session=(os.name == "posix"))
            try:
                returncode = process.wait(timeout=timeout)
                manifest["status"] = "completed" if returncode == 0 else "process_failed"
            except subprocess.TimeoutExpired:
                _stop_process_tree(process)
                manifest["status"] = "timed_out"
                returncode = process.returncode
            manifest["returncode"] = returncode
        except OSError as exc:
            manifest["status"] = "start_failed"
            manifest["error"] = str(exc)
    manifest["elapsed_seconds"] = time.monotonic()-start
    manifest["finished_utc"] = _utc_now()
    input_by_name = {entry["name"]:entry for entry in inputs}
    for item in sorted(run_dir.rglob("*")):
        if item.is_symlink():
            manifest.setdefault("warnings",[]).append(f"Output symlink not followed: {item.relative_to(run_dir)}")
            continue
        if item.is_file():
            name = item.relative_to(run_dir).as_posix()
            digest = sha256_file(item)
            if name not in input_by_name or digest != input_by_name[name]["sha256"]:
                manifest["outputs"].append({"name":name,"sha256":digest,"size_bytes":item.stat().st_size})
            if name in input_by_name and digest != input_by_name[name]["sha256"]:
                manifest.setdefault("warnings",[]).append(f"Backend modified copied input: {name}")
    manifest_path = run_dir / "manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return manifest
