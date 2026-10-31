"""Re-refine the openly published FeN4 measured-intensity fixture.

Run from the source checkout: python validation/run_real_fen4.py
The model is initialized from the deposited structure, not solved ab initio.
No Fcalc column or official SHELX executable enters the native optimization.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import gemmi
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from ariadne_xtal.io import read_structure, write_structure, write_reflections  # noqa: E402
from ariadne_xtal.model import Cell, Reflections  # noqa: E402
from ariadne_xtal.refinement import RefinementOptions, refine  # noqa: E402


def load_case():
    data_dir = PROJECT / "validation" / "public_data"
    structure = read_structure(data_dir / "2242624.cif")
    block = gemmi.cif.read_file(str(data_dir / "2242624.hkl")).sole_block()
    # The reflection archive retains more cell-angle digits than the rendered
    # structural CIF. Record this explicit choice instead of rounding twice.
    structure.cell = Cell(*[float(block.find_value("_cell_" + name)) for name in
                           ("length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma")])
    hkl = np.array([[int(value) for value in block.find_values("_refln_index_" + axis)]
                    for axis in "hkl"]).T
    observed = np.array([float(value) for value in block.find_values("_refln_F_squared_meas")])
    sigma = np.array([float(value) for value in block.find_values("_refln_F_squared_sigma")])
    # Deliberately do not read _refln_F_squared_calc.
    return structure, Reflections(hkl, observed, sigma)


def run_case(output_dir=None):
    output = Path(output_dir) if output_dir is not None else PROJECT / "validation" / "real_fen4"
    output.mkdir(parents=True, exist_ok=True)
    structure, reflections = load_case()
    options = RefinementOptions(weight_a=.0282, weight_b=.3122, weight_cycles=20,
                                covariance_scale="reduced_chi2")
    result = refine(structure, reflections, options)
    # A second declared start checks convergence away from the deposited N
    # coordinates; it is a local recovery experiment on the same data.
    perturbed = copy.deepcopy(structure)
    perturbed.atoms[1].frac += [.02, -.02, .015]
    perturbed.atoms[2].frac += [-.015, .02, -.02]
    perturbed.atoms[1].u_iso *= 1.5
    second = refine(perturbed, reflections, options)
    files = [PROJECT / "validation" / "public_data" / name
             for name in ("2242624.cif", "2242624.hkl")]
    report = {
        "experiment": "Native re-refinement of a deposited FeN4 model against measured F-squared",
        "article": "Bykov, M.; Khandarkhaeva, S.; Fedotenko, T.; Sedmak, P.; Dubrovinskaia, N.; Dubrovinsky, L. (2018). Acta Cryst. E74, 1392-1395.",
        "doi": "10.1107/S2056989018012161",
        "article_url": "https://journals.iucr.org/e/issues/2018/10/00/wm5459/index.html",
        "cod_id": "2242624",
        "sources": [{"file": str(path.relative_to(PROJECT)),
                     "download_url": "https://raw.githubusercontent.com/project-gemmi/gemmi/master/tests/" + path.name,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "bytes": path.stat().st_size} for path in files],
        "data_license": "Original IUCr article: CC-BY. COD fixture headers permit scientific-community reuse with attribution; original headers retained.",
        "observations": {"count": len(reflections), "kind": "published merged measured F_squared_meas and F_squared_sigma",
                         "computed_structure_factors_used_as_observations": False,
                         "raw_detector_images_used": False, "holdout": False},
        "model_choices": {"spacegroup": "P -1", "asymmetric_unit_atoms": 3,
                          "cell": structure.cell.to_dict(),
                          "cell_source": "higher-precision cell in published structure-factor CIF",
                          "anomalous_factors": "explicit atom-type dispersion terms from deposited model",
                          "weights": {"a": .0282, "b": .3122},
                          "restrictions": "Fe inversion-centre position exact; all Uiso; all occupancies fixed; no restraints",
                          "ignored_recipe": "CIF import is a structural import; no embedded SHELXL commands are executed"},
        "published_reference": {"r1_gt": .0403, "r1_all": .0413, "wr2": .0815, "gof": 1.183, "parameters": 10},
        "native": result.to_dict(),
        "perturbed_start": {"metrics": second.metrics, "success": second.success,
                            "maximum_fractional_difference_from_first_fit": float(np.max(np.abs(
                                np.array([a.frac for a in result.structure.atoms]) -
                                np.array([a.frac for a in second.structure.atoms]))))},
        "interpretation": "A measured-data regression for one known structure. It does not measure ab-initio success rates, compare execution speed to SHELXL, or demonstrate full instruction compatibility.",
    }
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_structure(result.structure, output / "refined.cif")
    write_structure(structure, output / "input_model.json")
    write_reflections(reflections, output / "measured_reflections.json")
    return report


if __name__ == "__main__":
    report = run_case()
    print(json.dumps({"native": report["native"]["metrics"],
                      "published_reference": report["published_reference"],
                      "perturbed_start": report["perturbed_start"]}, indent=2))
