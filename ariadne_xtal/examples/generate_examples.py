"""Recreate a labelled synthetic molecule, noisy intensities and native job.

This is illustrative synthetic data, not a measured crystal or new structure.
The separate validation directory contains real deposited FeN4 observations.
"""
from pathlib import Path
import copy
import json
import numpy as np
from scipy.spatial.transform import Rotation

from ariadne_xtal import Atom, Cell, Reflections, Structure, calculated_intensities
from ariadne_xtal.absolute import with_anomalous_factors
from ariadne_xtal.io import write_structure, write_reflections
from ariadne_xtal.validation import make_holdout


def main():
    root = Path(__file__).resolve().parent
    cell = Cell(9.6, 10.8, 12.2, 90, 103.5, 90)
    rotation = Rotation.from_euler("xyz", [23, -17, 31], degrees=True).as_matrix()
    local = [[1.40*np.cos(t), 1.40*np.sin(t), 0.] for t in np.arange(6)*np.pi/3]
    local += [[3.27, 0., 0.], [-2.87, 0., 0.], [-3.51, 1.04, 0.], [-3.51, -1.08, 0.]]
    elements = ["C"]*6+["Br", "C", "O", "O"]
    labels = ["C1", "C2", "C3", "C4", "C5", "C6", "Br1", "C7", "O1", "O2"]
    center = cell.to_cartesian([.5, .5, .5])
    coords = cell.to_fractional(np.asarray(local)@rotation.T+center)
    atoms = [Atom(label, element, xyz, u_iso=.025 if element=="C" else .032)
             for label, element, xyz in zip(labels, elements, coords)]
    atoms[6].u_aniso = rotation@np.diag([.018, .028, .041])@rotation.T
    truth = with_anomalous_factors(Structure(cell, atoms))
    hkl = np.array([(h,k,l) for h in range(-9,10) for k in range(-10,11) for l in range(-11,12)
                    if (h,k,l)!=(0,0,0)])
    hkl = hkl[cell.d_spacing(hkl) >= 1.05]
    intensity_scale = 2.4
    calculated = calculated_intensities(truth, hkl, intensity_scale)
    sigma = np.sqrt(4.+np.maximum(calculated,0)) + .01*calculated
    rng = np.random.default_rng(20261030)
    observed = calculated+rng.normal(0, sigma)
    refs = make_holdout(truth, Reflections(hkl, observed, sigma), .1, 20261030)
    start = copy.deepcopy(truth)
    for atom in start.atoms:
        atom.frac += rng.normal(0,.008,3)
        if atom.u_aniso is None:
            atom.u_iso = .04
        else:
            atom.u_aniso = np.eye(3)*.04
    write_structure(truth, root/"demo_truth.json")
    write_structure(start, root/"demo_start.json")
    write_structure(start, root/"demo_start.cif")
    write_reflections(refs, root/"demo_reflections.json")
    # HKLF4 intentionally has no automatic holdout coding; JSON is authoritative.
    write_reflections(Reflections(refs.hkl, refs.intensity, refs.sigma), root/"demo_reflections.hkl")
    job = {"action":"refine", "structure":"examples/demo_start.json",
           "hkl":"examples/demo_reflections.json", "graphics":True,
           "options":{"max_nfev":150,"covariance_scale":"absolute"}}
    (root/"refine_job.json").write_text(json.dumps(job,indent=2)+"\n")
    provenance={"kind":"synthetic demonstration", "seed":20261030,
                "description":"A synthetic planar brominated aromatic heavy-atom model; hydrogens omitted by design",
                "true_intensity_scale":intensity_scale, "n_reflections":len(refs),
                "n_holdout":int(refs.free_flags.sum()), "n_negative":int((refs.intensity<0).sum()),
                "noise_model":"independent Gaussian with sigma=sqrt(4+Icalc)+0.01 Icalc",
                "holdout_policy":"SHA256 over symmetry/Friedel orbit, seed fixed before refinement",
                "not_an_experimental_structure":True}
    (root/"PROVENANCE.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(json.dumps(provenance))


if __name__=="__main__":
    main()
