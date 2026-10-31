#!/usr/bin/env python3
"""Verify a built wheel from a fresh target directory, without source imports.

Usage:
    python tools/check_release.py --wheel dist/ariadne_xtal-0.1.0-py3-none-any.whl \
        --sdist dist/ariadne_xtal-0.1.0.tar.gz --output validation/packaging_report.json

The interpreter must already have the declared runtime dependencies.  Wheel
installation uses --no-deps and never requests the network.  Both CLI entry
points and a small native refinement are exercised outside the source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile


PROBE = r'''
from importlib import metadata
import json
from pathlib import Path
import sys

import numpy as np
from packaging.requirements import Requirement

import ariadne_xtal
from ariadne_xtal.io import write_structure, write_reflections
from ariadne_xtal.model import Atom, Cell, Reflections, Structure
from ariadne_xtal.scattering import calculated_intensities

target = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
module_path = Path(ariadne_xtal.__file__).resolve()
assert module_path.is_relative_to(target), f"Source import leaked into test: {module_path}"
versions = {}
for raw in metadata.distribution('ariadne-xtal').requires or []:
    requirement = Requirement(raw)
    if requirement.marker is not None and not requirement.marker.evaluate({'extra': ''}):
        continue
    installed = metadata.version(requirement.name)
    assert installed in requirement.specifier, f"{requirement.name} {installed} is outside {requirement.specifier}"
    versions[requirement.name] = installed
truth = Structure(Cell(8, 9, 10), [
    Atom('C1', 'C', [.13, .22, .31], u_iso=.024, refine_xyz=False, refine_u=False),
    Atom('O1', 'O', [.37, .41, .55], u_iso=.031, refine_xyz=False, refine_u=False),
])
hkl = np.array([(h,k,l) for h in range(1,4) for k in range(-2,3) for l in range(-2,3)])
reflections = Reflections(hkl, 1.8 * calculated_intensities(truth, hkl), np.ones(len(hkl)),
                          free_flags=np.arange(len(hkl)) % 7 == 0)
write_structure(truth, work/'model.json')
write_reflections(reflections, work/'reflections.json')
(work/'job.json').write_text(json.dumps({'action':'refine', 'structure':'model.json',
    'hkl':'reflections.json', 'options':{'max_nfev':30}, 'graphics':False}))
print(json.dumps({'module_imported_from_target': True,
    'module_relative_path': str(module_path.relative_to(target)),
    'version': ariadne_xtal.__version__, 'runtime_versions': versions,
    'reflection_count': len(reflections)}))
'''


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def checked_run(argv, cwd, env):
    outcome = subprocess.run(argv, cwd=cwd, env=env, text=True, capture_output=True, timeout=60)
    if outcome.returncode:
        raise RuntimeError(f"Command failed ({outcome.returncode}): {argv!r}\n{outcome.stdout}\n{outcome.stderr}")
    return outcome.stdout


def check_wheel(wheel: Path) -> dict:
    wheel = wheel.resolve(strict=True)
    with zipfile.ZipFile(wheel) as archive:
        if archive.testzip() is not None:
            raise ValueError("Wheel ZIP member checksum failed")
        members = archive.namelist()
        if not any(name.endswith("entry_points.txt") for name in members):
            raise ValueError("Wheel has no console entry points")
        if not any(name.endswith("licenses/LICENSE") or name.endswith(".dist-info/LICENSE") for name in members):
            raise ValueError("Wheel does not include its license")
    with tempfile.TemporaryDirectory(prefix="ariadne-wheel-smoke-") as temporary:
        temporary = Path(temporary)
        target, work = temporary / "installed", temporary / "job"
        work.mkdir()
        environment = {**os.environ, "PYTHONPATH": str(target), "PYTHONNOUSERSITE": "1"}
        checked_run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-compile",
                     "--no-index", "--target", str(target), str(wheel)], temporary, environment)
        probe = temporary / "probe.py"
        probe.write_text(PROBE, encoding="utf-8")
        import_result = json.loads(checked_run([sys.executable, str(probe), str(target), str(work)], temporary, environment))
        candidates = [target / name for name in ("bin/ariadne-xtal", "Scripts/ariadne-xtal.exe",
                                                 "bin/ariadne-xtal.exe", "Scripts/ariadne-xtal-script.py")]
        console = next((candidate for candidate in candidates if candidate.is_file()), None)
        if console is None:
            raise ValueError("Installed wheel is missing its CLI script")
        console_argv = [str(console)] if console.suffix.lower() == ".exe" else [sys.executable, str(console)]
        capabilities = json.loads(checked_run([*console_argv, "capabilities"], work, environment))
        assert capabilities["native_full_shelx_parity"] is False
        module_capabilities = json.loads(checked_run([sys.executable, "-m", "ariadne_xtal", "capabilities"], work, environment))
        assert module_capabilities == capabilities
        response = json.loads(checked_run([*console_argv, "run", "job.json", "--root", str(work)], work, environment))
        assert response["ok"] is True, response
        result = json.loads((work / response["result"]).read_text())
        manifest = json.loads((work / response["manifest"]).read_text())
        assert result["success"] is True, result.get("message")
        assert abs(result["scale"] - 1.8) < 1e-9, result["scale"]
        assert result["metrics"]["r1_gt"] < 1e-10
        assert manifest["scientific_acceptance"] == "not_assessed"
        assert not result.get("artifact_warnings")
        run = (work / response["manifest"]).parent
        for output in manifest["outputs"]:
            assert digest(run / output["path"]) == output["sha256"]
        result_record = {
            "wheel": wheel.name, "wheel_sha256": digest(wheel), "wheel_size_bytes": wheel.stat().st_size,
            **import_result, "console_capabilities": "passed", "module_capabilities": "passed",
            "native_job": {"transport_ok": response["ok"], "optimizer_success": result["success"],
                           "scale": result["scale"], "r1_gt": result["metrics"]["r1_gt"],
                           "n_work": result["metrics"]["n_work"], "n_free": result["metrics"]["n_free"],
                           "scientific_acceptance": manifest["scientific_acceptance"],
                           "output_hashes_verified": len(manifest["outputs"])},
        }
    return result_record


def check_sdist(sdist: Path) -> dict:
    sdist = sdist.resolve(strict=True)
    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
        if any(member.issym() or member.islnk() for member in members):
            raise ValueError("Source distribution must not contain symbolic/hard links")
        relative = {"/".join(Path(member.name).parts[1:]) for member in members}
        required = {"README.md", "LICENSE", "pyproject.toml", "docs/API.md", "docs/COMPATIBILITY.md",
                    "docs/report.tex", "examples/generate_examples.py", "examples/refine_job.json",
                    "validation/run_real_fen4.py", "shelx_instruction_matrix.json"}
        missing = sorted(required - relative)
        if missing:
            raise ValueError(f"Source distribution omits documented deliverables: {missing}")
        if any("__pycache__" in name or name.endswith((".pyc", ".pyo")) for name in relative):
            raise ValueError("Source distribution contains interpreter cache files")
    return {"sdist": sdist.name, "sdist_sha256": digest(sdist), "sdist_size_bytes": sdist.stat().st_size,
            "member_count": len(members), "documented_source_paths_present": sorted(required), "symlinks": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--sdist", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"schema": "ariadne_xtal.release_verification/1", "status": "passed",
              "wheel_verification": check_wheel(args.wheel)}
    if args.sdist:
        report["sdist_verification"] = check_sdist(args.sdist)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
