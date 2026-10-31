"""Deterministic synthetic P1 phase retrieval followed by native F² refinement.

The truth is supplied only to simulate observations and to evaluate recovered
coordinates modulo origin, permutation and inversion. The solver receives the
cell, measured intensities, uncertainties and the unit-cell element list.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from ariadne_xtal.io import write_reflections, write_structure  # noqa: E402
from ariadne_xtal.model import Atom, Cell, Reflections, Structure  # noqa: E402
from ariadne_xtal.phasing import PhasingOptions, solve_p1  # noqa: E402
from ariadne_xtal.refinement import RefinementOptions, refine  # noqa: E402
from ariadne_xtal.scattering import calculated_intensities  # noqa: E402


def match_rms(reference, candidate):
    expected = np.array([atom.frac for atom in reference.atoms])
    recovered = np.array([atom.frac for atom in candidate.atoms])
    if len(expected) != len(recovered):
        raise ValueError("Matching requires the same number of atoms")
    best = (np.inf, None)
    for sign in (-1, 1):
        for permutation in itertools.permutations(range(len(expected))):
            target = expected[list(permutation)]
            shift = target[0] - sign * recovered[0]
            for _ in range(3):
                displacement = reference.cell.minimum_image(target - (sign * recovered + shift))
                shift += displacement.mean(axis=0)
            displacement = reference.cell.minimum_image(target - (sign * recovered + shift))
            rms = float(np.sqrt(np.mean(np.sum(reference.cell.to_cartesian(displacement)**2, axis=1))))
            if rms < best[0]:
                best = (rms, {"inversion_sign": sign, "permutation": list(permutation),
                              "origin_shift_fractional": shift.tolist()})
    return {"rms_cartesian_angstrom": best[0], **best[1]}


def make_case():
    cell = Cell(5.2, 5.7, 6.1)
    coords = [[.113, .239, .337], [.371, .628, .793], [.725, .159, .563], [.842, .771, .188]]
    truth = Structure(cell, [Atom(f"C{i + 1}", "C", xyz, u_iso=.012) for i, xyz in enumerate(coords)])
    all_h = np.array([h for h in itertools.product(range(-9, 10), range(-10, 11), range(-11, 12)) if h != (0, 0, 0)])
    hkl = all_h[cell.d_spacing(all_h) > .55]
    data = Reflections(hkl, calculated_intensities(truth, hkl), np.full(len(hkl), .2))
    return truth, data


def run_case(output_dir=None):
    output = Path(output_dir) if output_dir else PROJECT / "validation" / "phasing_recovery"
    output.mkdir(parents=True, exist_ok=True)
    truth, observations = make_case()
    input_before = observations.to_dict()
    options = PhasingOptions(trials=4, iterations=400, keep=4)
    solutions = solve_p1(truth.cell, observations, ["C"] * 4, options)
    results = []
    for rank, candidate in enumerate(solutions.candidates, 1):
        fit = refine(candidate["structure"], observations, RefinementOptions(max_nfev=150))
        results.append({"rank": rank, "trial": candidate["trial"],
                        "projection_residual": candidate["projection_residual"],
                        "before_refinement": match_rms(truth, candidate["structure"]),
                        "after_refinement": match_rms(truth, fit.structure),
                        "refinement": fit.to_dict()})
        write_structure(candidate["structure"], output / f"candidate_{rank}.json")
        write_structure(fit.structure, output / f"refined_{rank}.cif")
    report = {"experiment": "Synthetic four-carbon P1 phase retrieval and local refinement",
              "seed": options.seed, "trials": options.trials, "iterations": options.iterations,
              "composition_supplied": ["C"] * 4, "n_reflections": len(observations),
              "d_min_selection_angstrom": .55,
              "noise": "No added random noise. Assigned sigma=0.2 for objective scaling only.",
              "model_coordinates_available_to_solver": False,
              "comparison": "origin, inversion and same-element permutation are optimized for coordinate assessment",
              "observations_unchanged": observations.to_dict() == input_before,
              "results": results,
              "interpretation": "One deterministic algorithm-recovery test, not a success-rate estimate over experimental structures."}
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_structure(truth, output / "truth.json")
    write_reflections(observations, output / "synthetic_reflections.json")
    return report


if __name__ == "__main__":
    report = run_case()
    print(json.dumps({"n_reflections": report["n_reflections"], "unchanged": report["observations_unchanged"],
                      "candidates": [{"rank": r["rank"], "pre_rms": r["before_refinement"]["rms_cartesian_angstrom"],
                                      "post_rms": r["after_refinement"]["rms_cartesian_angstrom"],
                                      "wr2": r["refinement"]["metrics"]["wr2"],
                                      "success": r["refinement"]["success"]} for r in report["results"]]}, indent=2))
