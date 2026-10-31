"""Independent audit of maps, absolute structure, feedback and full workflows."""

import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import gemmi
import numpy as np
import pytest

from ariadne_xtal.absolute import friedel_quotient, with_anomalous_factors
from ariadne_xtal.feedback import StructuralPrior, deblend_profiles
from ariadne_xtal.maps import difference_map, expand_coefficients, fourier_map, model_map, patterson_map
from ariadne_xtal.model import Atom, Cell, Reflections, Structure
from ariadne_xtal.phasing import PhasingOptions, solve_p1
from ariadne_xtal.refinement import RefinementOptions, refine
from ariadne_xtal.scattering import calculated_intensities, structure_factors
from ariadne_xtal.validation import make_holdout, reflection_orbit


def all_indices(limit=3):
    return np.array([h for h in itertools.product(range(-limit, limit + 1), repeat=3) if h != (0, 0, 0)])


def test_fourier_sign_peak_position_and_electron_integral():
    cell = Cell(8, 9, 10, 82, 97, 103)
    frac = np.array([3, 5, 7]) / 16
    hkl = np.array(list(itertools.product(range(-6, 7), repeat=3)))
    coefficients = 8 * np.exp(2j * np.pi * (hkl @ frac))
    density = fourier_map(cell, hkl, coefficients, shape=(16, 16, 16), f000=8)
    assert np.unravel_index(np.argmax(density.values), density.values.shape) == (3, 5, 7)
    assert density.values.mean() * cell.volume == pytest.approx(8, abs=1e-10)
    point = np.array([2, 8, 1]) / 16
    direct = np.sum(coefficients * np.exp(-2j * np.pi * (hkl @ point))) / cell.volume
    assert density.values[2, 8, 1] == pytest.approx(direct.real, abs=1e-11)
    assert density.metadata["max_imaginary_roundoff"] < 1e-10


def test_symmetry_expansion_uses_screw_translation_phases():
    structure = Structure(Cell(8, 9, 10), [Atom("C1", "C", [.123, .237, .319]),
                                          Atom("O1", "O", [.293, .419, .581])], "P 21 21 21")
    hkl = np.array([[1, 2, 3], [2, 3, 1], [1, 1, 0]])
    expanded, coefficients = expand_coefficients(hkl, structure_factors(structure, hkl), structure)
    oracle = structure_factors(structure, expanded)
    assert np.allclose(coefficients, oracle, atol=1e-12, rtol=1e-12)


def test_patterson_does_not_apply_complex_structure_factor_translation_phases():
    structure = Structure(Cell(8, 9, 10, 90, 101, 90),
                          [Atom("C1", "C", [.123, .237, .319]),
                           Atom("O1", "O", [.293, .419, .581])], "P 21/c")
    hkl = all_indices(3)
    data = Reflections(hkl, calculated_intensities(structure, hkl), np.ones(len(hkl)))
    expanded = patterson_map(structure.cell, data, structure=structure, shape=(16, 16, 16))
    explicit = patterson_map(structure.cell, data, structure=None, shape=(16, 16, 16))
    assert np.allclose(expanded.values, explicit.values, atol=1e-11)
    assert np.unravel_index(np.argmax(expanded.values), expanded.values.shape) == (0, 0, 0)
    for idx in [(2, 3, 4), (3, 1, 6), (7, 6, 2)]:
        inverse = tuple((-np.array(idx)) % 16)
        assert expanded.values[idx] == pytest.approx(expanded.values[inverse], abs=1e-11)


def test_difference_map_is_zero_for_matching_data_and_does_not_use_holdouts():
    structure = Structure(Cell(8, 9, 10), [Atom("C1", "C", [.123, .237, .319]),
                                          Atom("O1", "O", [.293, .419, .581])])
    hkl = all_indices(2)
    data = Reflections(hkl, calculated_intensities(structure, hkl, scale=1.7), np.ones(len(hkl)))
    held = make_holdout(structure, data, fraction=.2, seed=11)
    before = held.to_dict()
    density = difference_map(structure, held, scale=1.7)
    assert np.max(np.abs(density.values)) < 1e-12
    altered = copy.deepcopy(held)
    altered.intensity[altered.free_flags] += 10000
    assert np.array_equal(density.values, difference_map(structure, altered, scale=1.7).values)
    assert held.to_dict() == before


def test_ccp4_export_retains_map_grid_geometry_and_values(tmp_path):
    structure = Structure(Cell(8, 9, 10, 83, 101, 95), [Atom("O1", "O", [.25, .25, .5])])
    density = model_map(structure, all_indices(3), shape=(16, 18, 20))
    path = tmp_path / "model.ccp4"
    density.save_ccp4(path)
    exported = gemmi.read_ccp4_map(str(path))
    exported.setup(float("nan"))
    assert exported.grid.shape == (16, 18, 20)
    assert exported.grid.unit_cell.volume == pytest.approx(structure.cell.volume, rel=1e-6)
    assert np.allclose(np.asarray(exported.grid), density.values, rtol=2e-7, atol=2e-7)


def test_fourier_aliasing_and_noninteger_indices_are_rejected():
    cell = Cell(8, 9, 10)
    with pytest.raises(ValueError, match="alias"):
        fourier_map(cell, [[4, 1, 1]], [1], shape=(8, 8, 8))
    with pytest.raises(ValueError, match="integer"):
        fourier_map(cell, [[1.1, 0, 1]], [1])


def anomalous_structure():
    return Structure(Cell(8, 9, 10), [
        Atom("Fe1", "Fe", [.12, .23, .34], f_prime=-1.1, f_double_prime=3.2),
        Atom("O1", "O", [.27, .38, .45]), Atom("C1", "C", [.43, .51, .26])])


def test_friedel_quotient_recovers_declared_inversion_fraction_and_common_scale():
    structure = anomalous_structure()
    hkl = all_indices(3)
    alpha = .23
    fplus, fminus = structure_factors(structure, hkl), structure_factors(structure, -hkl)
    intensity = 7.3 * ((1 - alpha) * abs(fplus)**2 + alpha * abs(fminus)**2)
    data = Reflections(hkl, intensity, np.ones(len(hkl)))
    result = friedel_quotient(structure, data)
    assert result["x"] == pytest.approx(alpha, abs=1e-12)
    assert result["standard_uncertainty_conditional"] > 0
    assert result["chi_square"] < 1e-20


def test_friedel_quotient_does_not_invent_signal_without_anomalous_factors():
    structure = anomalous_structure()
    for atom in structure.atoms:
        atom.f_double_prime = 0
    hkl = all_indices(2)
    data = Reflections(hkl, calculated_intensities(structure, hkl), np.ones(len(hkl)))
    with pytest.raises(ValueError, match="identifiable"):
        friedel_quotient(structure, data)


def test_opt_in_anomalous_adapter_never_changes_input_model():
    structure = anomalous_structure()
    before = structure.to_dict()
    corrected = with_anomalous_factors(structure)
    assert structure.to_dict() == before
    expected = gemmi.cromer_liberman(26, gemmi.hc / structure.wavelength)
    assert corrected.atoms[0].f_prime == pytest.approx(expected[0])
    assert corrected.atoms[0].f_double_prime == pytest.approx(expected[1])


def test_profile_prior_information_cap_and_linear_normal_solution():
    x = np.linspace(-3, 3, 50)
    design = np.column_stack([np.exp(-.5 * (x + .3)**2), np.exp(-.5 * (x - .3)**2)])
    sigma = np.linspace(.8, 1.2, len(x))
    observations = design @ np.array([11., 18.])
    mean = np.array([15., 12.])
    covariance = np.array([[.8, .2], [.2, 1.1]])
    prior = StructuralPrior(mean, covariance, "train-excluding-A", ["A"])
    before = (design.copy(), observations.copy(), sigma.copy())
    result = deblend_profiles(design, observations, sigma, group_id="A", prior=prior,
                              strength=1, maximum_information_fraction=.15)
    aw, yw = design / sigma[:, None], observations / sigma
    information = aw.T @ aw
    precision = np.linalg.inv(covariance)
    effective = result["effective_prior_strength"]
    evals, evecs = np.linalg.eigh(information)
    inverse_sqrt = (evecs / np.sqrt(evals)) @ evecs.T
    ratio = np.linalg.eigvalsh(inverse_sqrt @ (effective * precision) @ inverse_sqrt).max()
    assert ratio == pytest.approx(.15, rel=1e-10)
    expected = np.linalg.solve(information + effective * precision, aw.T @ yw + effective * precision @ mean)
    assert np.allclose(result["intensity_estimate"], expected, rtol=1e-11)
    assert np.allclose(result["covariance_local"], np.linalg.inv(information + effective * precision), rtol=1e-10)
    assert all(np.array_equal(a, b) for a, b in zip(before, (design, observations, sigma)))
    assert result["raw_observations_modified"] is False


def test_profile_nullspace_cannot_be_presented_as_measured_precision():
    design = np.ones((5, 2))
    result = deblend_profiles(design, np.arange(5.0), np.ones(5), group_id="A")
    assert result["data_information_rank"] == 1
    assert result["covariance_local"] is None
    assert not result["identifiable_from_data"]
    assert len(result["data_information_nullspace"]) == 1
    prior = StructuralPrior(np.ones(2), np.eye(2), "independent-run", ["A"])
    with pytest.raises(ValueError, match="rank deficient"):
        deblend_profiles(design, np.arange(5.0), np.ones(5), group_id="A", prior=prior)


def test_profile_prior_requires_excluded_group_provenance():
    prior = StructuralPrior(np.ones(2), np.eye(2), "run-B", ["B"])
    with pytest.raises(ValueError, match="cross-fitted"):
        deblend_profiles(np.eye(2), [2, 3], [1, 1], group_id="A", prior=prior)


def test_holdout_partition_closes_spacegroup_friedel_and_twin_orbits():
    structure = Structure(Cell(8, 8, 10), [Atom("O1", "O", [.1, .2, .3])], "P 4")
    hkl = all_indices(3)
    law = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    data = Reflections(hkl, calculated_intensities(structure, hkl), np.ones(len(hkl)))
    split = make_holdout(structure, data, .2, seed=19, twin_laws=[law])
    assigned = {tuple(h): bool(flag) for h, flag in zip(split.hkl, split.free_flags)}
    for h in split.hkl:
        flags = {assigned[tuple(mate)] for mate in reflection_orbit(structure, h, [law]) if tuple(mate) in assigned}
        assert len(flags) == 1
    assert data.free_flags is None


@pytest.mark.parametrize("options", [{"trials": 1.5}, {"iterations": 2.5}, {"threshold_sigma": np.nan},
                                      {"minimum_peak_distance": np.inf}, {"oversampling": np.nan}])
def test_phasing_numerical_options_are_validated(options):
    with pytest.raises(ValueError):
        PhasingOptions(**options)


def test_phasing_rejects_friedel_split_leakage_before_search():
    hkl = all_indices(2)
    flags = np.zeros(len(hkl), dtype=bool)
    flags[0] = True
    data = Reflections(hkl, np.ones(len(hkl)), np.ones(len(hkl)), free_flags=flags)
    with pytest.raises(ValueError, match="Holdout leakage"):
        solve_p1(Cell(8, 9, 10), data, ["C"], PhasingOptions(trials=1, iterations=1, keep=1))


def test_published_fen4_measured_data_rerefines_to_archived_residuals():
    path = Path(__file__).resolve().parents[1] / "validation" / "run_real_fen4.py"
    spec = importlib.util.spec_from_file_location("real_case", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    structure, data = module.load_case()
    result = refine(structure, data, RefinementOptions(weight_a=.0282, weight_b=.3122,
                                                     weight_cycles=20, covariance_scale="reduced_chi2"))
    assert len(data) == 71
    assert result.success and result.metrics["jacobian_rank"] == 10
    assert result.metrics["r1_gt"] == pytest.approx(.0403, abs=5e-5)
    assert result.metrics["r1_all"] == pytest.approx(.0413, abs=5e-5)
    assert result.metrics["wr2"] == pytest.approx(.0815, abs=5e-5)
    assert result.metrics["gof"] == pytest.approx(1.183, abs=5e-4)


def test_phase_retrieval_then_refinement_recovers_four_atom_truth_without_input_mutation(tmp_path):
    path = Path(__file__).resolve().parents[1] / "validation" / "run_phasing_recovery.py"
    spec = importlib.util.spec_from_file_location("phasing_case", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run_case(tmp_path)
    assert result["n_reflections"] == 4540
    assert result["observations_unchanged"]
    assert len(result["results"]) == 4
    for candidate in result["results"]:
        assert candidate["refinement"]["success"]
        assert candidate["refinement"]["metrics"]["identifiable"]
        assert candidate["after_refinement"]["rms_cartesian_angstrom"] < 1e-5
        assert candidate["refinement"]["metrics"]["wr2"] < 1e-6


def test_agent_json_reflections_preserve_holdout_and_input_bytes(tmp_path):
    from ariadne_xtal.agent import AgentWorkspace
    from ariadne_xtal.io import write_reflections, write_structure

    structure = Structure(Cell(8, 9, 10), [Atom("C1", "C", [.13, .23, .31]),
                                          Atom("O1", "O", [.27, .39, .51])])
    hkl = all_indices(2)
    data = Reflections(hkl, calculated_intensities(structure, hkl), np.ones(len(hkl)))
    data = make_holdout(structure, data, .2, seed=19)
    write_structure(structure, tmp_path / "model.json")
    write_reflections(data, tmp_path / "reflections.json")
    paths = [tmp_path / "model.json", tmp_path / "reflections.json"]
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    response = AgentWorkspace(tmp_path).dispatch({"action": "refine", "structure": "model.json",
                                                "hkl": "reflections.json", "graphics": False})
    assert response["ok"], response
    result = json.loads((tmp_path / response["result"]).read_text())
    assert result["metrics"]["n_free"] == int(data.free_flags.sum())
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths} == before
    manifest = json.loads((tmp_path / response["manifest"]).read_text())
    assert manifest["status"] == "completed"
    assert len(manifest["inputs"]) == 2


def test_agent_rejects_inputs_outside_workspace(tmp_path):
    from ariadne_xtal.agent import AgentWorkspace
    response = AgentWorkspace(tmp_path).dispatch({"action": "inspect", "structure": "../outside.json"})
    assert not response["ok"]
    assert "within the agent workspace" in response["error"]["message"]
