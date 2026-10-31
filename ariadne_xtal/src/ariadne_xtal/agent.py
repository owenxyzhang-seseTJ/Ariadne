"""Structured agent tools with isolated outputs and immutable input hashes.

The host LLM proposes actions. Numerical modules execute declared operations.
No language-model output is interpreted as Python or shell code.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
import warnings

import gemmi
import numpy as np
import scipy

from . import __version__
from .capabilities import capabilities
from .io import read_structure, read_reflections, write_structure
from .model import Cell


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024*1024), b""):
            digest.update(part)
    return digest.hexdigest()


TOOLS = [
    {"name": "capabilities", "description": "Read exact native/legacy capability boundaries", "required": []},
    {"name": "inspect", "description": "Read a model and reflection statistics", "required": ["structure"]},
    {"name": "refine", "description": "Refine declared work F-squared observations; save model, diagnostics and maps", "required": ["structure", "hkl"]},
    {"name": "solve", "description": "Generate P1 atomic hypotheses from known cell composition", "required": ["cell", "elements", "hkl"]},
    {"name": "maps", "description": "Compute diagnostic Fourier maps and an offline dashboard", "required": ["structure", "hkl"]},
    {"name": "absolute", "description": "Estimate a conditional fixed-model Friedel-quotient parameter", "required": ["structure", "hkl"]},
    {"name": "legacy", "description": "Run an explicitly allowlisted installed SHELXT/SHELXL backend", "required": ["engine", "input"]},
]


class AgentWorkspace:
    def __init__(self, root, legacy_executables=None, atom_limit=500, reflection_limit=100_000):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError("Agent workspace must be an existing directory")
        self.legacy = {str(k).lower(): str(Path(v).resolve())
                       for k, v in (legacy_executables or {}).items()}
        self.atom_limit, self.reflection_limit = atom_limit, reflection_limit

    def _input(self, path):
        p = (self.root/str(path)).resolve()
        if not p.is_relative_to(self.root) or not p.is_file():
            raise ValueError("Input must be a regular file within the agent workspace")
        if p.stat().st_size > 200*1024*1024:
            raise ValueError("Input exceeds the agent file-size budget")
        return p

    def dispatch(self, request):
        if not isinstance(request, dict):
            raise ValueError("Tool request must be a JSON object")
        action = request.get("action")
        specs = {t["name"]: t for t in TOOLS}
        if action not in specs:
            raise ValueError(f"Unknown action {action!r}")
        fields = {
            "capabilities": set(),
            "inspect": {"structure", "hkl", "spacegroup_candidates"},
            "refine": {"structure", "hkl", "options", "restraints", "constraints", "graphics"},
            "solve": {"cell", "elements", "hkl", "options", "scale"},
            "maps": {"structure", "hkl", "scale"},
            "absolute": {"structure", "hkl", "options"},
            "legacy": {"engine", "input", "timeout", "extra_args", "auxiliary_paths"},
        }
        unknown = set(request) - fields[action] - {"action", "id"}
        if unknown:
            raise ValueError(f"Unknown request fields for {action}: {sorted(unknown)}")
        if action == "capabilities":
            return {"ok": True, "capabilities": capabilities(), "tools": TOOLS}
        for field in specs[action]["required"]:
            if field not in request:
                raise ValueError(f"Action {action!r} requires {field!r}")
        runs = self.root/"runs"
        if runs.is_symlink():
            raise ValueError("Run output directory cannot be a symlink")
        runs.mkdir(exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix=action+"_", dir=runs))
        started = time.time()
        manifest = {"schema_version": 1, "run_id": run.name, "action": action,
                    "request": request, "status": "running", "started_unix": started,
                    "scientific_acceptance": "not_assessed",
                    "environment": {"ariadne_xtal": __version__, "python": platform.python_version(),
                                    "numpy": np.__version__, "scipy": scipy.__version__, "gemmi": gemmi.__version__},
                    "inputs": [], "outputs": []}
        input_paths = {}
        write_json(run/"manifest.json", manifest)
        try:
            for key in ("structure", "hkl", "input"):
                if key in request:
                    p = self._input(request[key])
                    snapshot = run/"inputs"/(key+p.suffix)
                    snapshot.parent.mkdir(exist_ok=True)
                    shutil.copyfile(p, snapshot)
                    input_paths[key] = p if action == "legacy" else snapshot
                    manifest["inputs"].append({"role": key, "path": str(p.relative_to(self.root)),
                                               "snapshot": str(snapshot.relative_to(run)),
                                               "sha256": sha256(snapshot), "bytes": snapshot.stat().st_size})
            with warnings.catch_warnings(record=True) as import_warnings:
                warnings.simplefilter("always")
                s = read_structure(input_paths["structure"]) if "structure" in input_paths else None
                r = read_reflections(input_paths["hkl"]) if "hkl" in input_paths else None
            manifest["import_diagnostics"] = [str(w.message) for w in import_warnings]
            if s is not None and len(s.atoms) > self.atom_limit:
                raise ValueError("Model exceeds agent atom budget")
            if r is not None and len(r) > self.reflection_limit:
                raise ValueError("Data exceed agent reflection budget")
            if action == "inspect":
                from .symmetry import unique_sites, rank_spacegroups
                result = {"cell": s.cell.to_dict(), "cell_volume": s.cell.volume,
                          "asymmetric_unit_atoms": len(s.atoms),
                          "unit_cell_sites": sum(len(unique_sites(s, a)) for a in s.atoms),
                          "spacegroup_label": s.spacegroup, "operation_count": len(s.operations),
                          "explicit_operations_authoritative": s.sym_ops is not None}
                if r is not None:
                    result.update(n_reflections=len(r), negative_intensities=int(np.sum(r.intensity < 0)))
                    if request.get("spacegroup_candidates"):
                        result["spacegroup_diagnostics"] = rank_spacegroups(s.cell, r, request["spacegroup_candidates"])
            elif action == "refine":
                from .refinement import RefinementOptions, refine
                options = dict(request.get("options", {}))
                if options.get("max_nfev", 200) > 5000:
                    raise ValueError("Refinement exceeds agent evaluation budget")
                options["verbose"] = 0
                fit = refine(s, r, RefinementOptions(**options), request.get("restraints"), request.get("constraints"))
                result = fit.to_dict()
                write_structure(fit.structure, run/"refined.json")
                write_structure(fit.structure, run/"refined.cif")
                # The plain RES is a structural export; no unsupported directives are invented.
                try:
                    write_structure(fit.structure, run/"refined.res")
                except ValueError as error:
                    result.setdefault("artifact_warnings", []).append({"artifact": "refined.res", "reason": str(error)})
                if request.get("graphics", True):
                    try:
                        self._graphics(fit.structure, r, run, fit.scale, fit.history, fit.covariance,
                                       fit.parameter_names, options.get("twin_laws"),
                                       getattr(fit, "twin_fractions", options.get("twin_fractions")))
                    except ValueError as error:
                        result.setdefault("artifact_warnings", []).append({"artifact": "graphics", "reason": str(error)})
            elif action == "solve":
                from .phasing import PhasingOptions, solve_p1
                cell = Cell(**request["cell"])
                phasing_options = PhasingOptions(**request.get("options", {}))
                if phasing_options.trials * phasing_options.iterations > 20_000:
                    raise ValueError("Agent P1 phasing budget is 20,000 combined trial-iterations")
                result_obj = solve_p1(cell, r, list(request["elements"]),
                                      phasing_options, request.get("scale", 1.0))
                result = result_obj.to_dict()
                for i, candidate in enumerate(result_obj.candidates):
                    write_structure(candidate["structure"], run/f"candidate_{i+1}.json")
                    write_structure(candidate["structure"], run/f"candidate_{i+1}.cif")
            elif action == "maps":
                self._graphics(s, r, run, float(request.get("scale", 1.0)))
                result = {"map_type": "unweighted Fo-Fc", "scale": request.get("scale", 1.0)}
            elif action == "absolute":
                from .absolute import friedel_quotient
                result = friedel_quotient(s, r, **request.get("options", {}))
            else:
                from .compat import run_legacy
                engine = str(request["engine"]).lower()
                if engine not in self.legacy:
                    raise ValueError("Official engine was not enabled with an explicit executable at startup")
                auxiliary = [self._input(p) for p in request.get("auxiliary_paths", [])]
                result = run_legacy(engine, self.legacy[engine], input_paths["input"], run,
                                    timeout=min(float(request.get("timeout", 300)), 3600),
                                    extra_args=request.get("extra_args"), auxiliary_paths=auxiliary)
                if result.get("status") != "completed":
                    write_json(run/"result.json", result)
                    raise RuntimeError(f"Official process ended with {result.get('status')}; see saved backend manifest")
            for entry in manifest["inputs"]:
                if sha256(self.root/entry["path"]) != entry["sha256"]:
                    raise RuntimeError("An input file changed during the run")
            write_json(run/"result.json", result)
            manifest["status"] = "completed"
            manifest["outputs"] = [{"path": str(p.relative_to(run)), "sha256": sha256(p),
                                    "bytes": p.stat().st_size}
                                   for p in sorted(run.rglob("*"))
                                   if not p.is_symlink() and p.is_file() and p.name != "manifest.json"]
            manifest["elapsed_seconds"] = time.time()-started
            write_json(run/"manifest.json", manifest)
            return {"ok": True, "run_id": run.name, "manifest": str((run/"manifest.json").relative_to(self.root)),
                    "result": str((run/"result.json").relative_to(self.root)), "summary": self._summary(result)}
        except Exception as error:
            manifest.update(status="failed", elapsed_seconds=time.time()-started,
                            error={"type": type(error).__name__, "message": str(error)})
            write_json(run/"manifest.json", manifest)
            return {"ok": False, "run_id": run.name, "manifest": str((run/"manifest.json").relative_to(self.root)),
                    "error": manifest["error"]}

    @staticmethod
    def _summary(result):
        if "metrics" in result:
            return {"success": result.get("success"), "metrics": result["metrics"]}
        if "candidates" in result:
            return {"candidate_count": len(result["candidates"]), "seed": result.get("seed")}
        return {k: v for k, v in result.items() if k not in {"pair_hkl", "q_observed", "q_calculated"}}

    @staticmethod
    def _graphics(s, r, run, scale, history=None, covariance=None, parameter_names=None,
                  twin_laws=None, twin_fractions=None):
        from .maps import difference_map, patterson_map
        from .visualization import create_dashboard, static_figures
        density = None
        if not twin_laws:
            density = difference_map(s, r, scale)
            density.save_npz(run/"difference_map.npz")
            density.save_ccp4(run/"difference_map.ccp4")
        patterson = patterson_map(s.cell, r, scale, s)
        patterson.save_npz(run/"patterson_map.npz")
        create_dashboard(s, r, run/"dashboard.html", scale, history, covariance, parameter_names,
                         density, twin_laws, twin_fractions)
        static_figures(s, r, run/"figures", scale=scale, history=history, density=density,
                       covariance=covariance, parameter_names=parameter_names,
                       twin_laws=twin_laws, twin_fractions=twin_fractions)


def serve_jsonl(workspace, inp=None, out=None):
    inp, out = inp or sys.stdin, out or sys.stdout
    for line in inp:
        if not line.strip():
            continue
        try:
            if len(line) > 1_048_576:
                raise ValueError("JSON request exceeds 1 MiB")
            request = json.loads(line)
            result = workspace.dispatch(request)
        except Exception as error:
            result = {"ok": False, "error": {"type": type(error).__name__, "message": str(error)}}
        out.write(json.dumps(json_safe(result), ensure_ascii=False, allow_nan=False)+"\n")
        out.flush()
