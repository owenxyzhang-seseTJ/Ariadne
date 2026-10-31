"""Recovery, identifiability and conservation tests, not optimizer snapshots."""

import copy
import itertools
import json

import gemmi
import numpy as np
import pytest

from ariadne_xtal.model import Atom, Cell, Reflections, Structure
from ariadne_xtal.refinement import RefinementOptions, grouped_holdout_flags, refine
from ariadne_xtal.restraints import Restraint, restraint_residuals
from ariadne_xtal.scattering import calculated_intensities, structure_factors


def indices(limit=3):
    return np.array([h for h in itertools.product(range(-limit, limit + 1), repeat=3)
                     if h != (0, 0, 0) and (h[2] > 0 or (h[2] == 0 and h[1] > 0)
                                            or (h[2] == h[1] == 0 and h[0] > 0))])


def molecular_structure():
    return Structure(Cell(9.1, 10.2, 11.3, 83, 97, 103), [
        Atom("C1", "C", [.13, .21, .30], u_iso=.021),
        Atom("O1", "O", [.30, .24, .35], u_iso=.033),
        Atom("N1", "N", [.21, .40, .39], u_iso=.027),
    ])


def gemmi_intensities(structure, hkl, scale=1):
    """Generate the recovery observations with Gemmi's independent SF engine."""
    other = gemmi.SmallStructure()
    other.cell = gemmi.UnitCell(**structure.cell.to_dict())
    other.spacegroup = gemmi.find_spacegroup_by_name(structure.spacegroup)
    other.setup_cell_images()
    for atom in structure.atoms:
        site = gemmi.SmallStructure.Site()
        site.label = atom.label
        site.element = gemmi.Element(atom.element)
        site.fract = gemmi.Fractional(*atom.frac)
        site.occ = atom.occupancy
        site.u_iso = atom.u_iso
        if atom.u_aniso is not None:
            inverse = np.linalg.inv(structure.cell.matrix)
            ustar = inverse @ atom.u_aniso @ inverse.T
            lengths = np.linalg.norm(structure.cell.reciprocal, axis=0)
            cif_u = ustar / np.outer(lengths, lengths)
            site.aniso = gemmi.SMat33d(cif_u[0, 0], cif_u[1, 1], cif_u[2, 2],
                                     cif_u[0, 1], cif_u[0, 2], cif_u[1, 2])
        other.add_site(site)
    other.change_occupancies_to_crystallographic(max_dist=1e-6)
    calculator = gemmi.StructureFactorCalculatorX(other.cell)
    return scale * np.abs(np.array([calculator.calculate_sf_from_small_structure(other, h.tolist())
                                    for h in hkl]))**2


def test_recovers_coordinates_uiso_and_scale_from_independent_gemmi_data():
    truth = molecular_structure()
    hkl = indices()
    data = Reflections(hkl, gemmi_intensities(truth, hkl, 1.8), np.ones(len(hkl)))
    start = copy.deepcopy(truth)
    start.atoms[1].frac += [.012, -.007, .009]
    start.atoms[2].frac += [-.007, .011, .005]
    start.atoms[1].u_iso = .045
    saved = start.to_dict()
    result = refine(start, data)
    assert result.success and result.metrics["identifiable"]
    assert result.scale == pytest.approx(1.8, rel=2e-7)
    for atom, expected in zip(result.structure.atoms, truth.atoms):
        assert np.allclose(atom.frac, expected.frac, atol=2e-7)
        assert atom.u_iso == pytest.approx(expected.u_iso, abs=2e-7)
    assert start.to_dict() == saved
    assert result.metrics["wr2"] < 1e-6
    assert result.gauge["dimension"] == 3
    assert result.standard_uncertainties["C1.x"] is None
    assert result.standard_uncertainties["O1.x"] > 0
    assert result.covariance.shape == (10, 10)
    json.dumps(result.to_dict(), allow_nan=False)


def test_negative_intensities_are_included_in_the_scale_estimator():
    structure = molecular_structure()
    for atom in structure.atoms:
        atom.refine_xyz = atom.refine_u = False
    hkl = indices(2)
    ideal = calculated_intensities(structure, hkl)
    measured = ideal * 1.7
    measured[::7] = -30
    sigma = np.linspace(1, 2, len(hkl))
    expected = np.sum(measured * ideal / sigma**2) / np.sum(ideal**2 / sigma**2)
    result = refine(structure, Reflections(hkl, measured, sigma))
    assert result.scale == pytest.approx(expected, rel=1e-9)
    # The delta-method uncertainty must reduce to the exact linear-scale SE.
    expected_se = 1 / np.sqrt(np.sum(ideal**2 / sigma**2))
    assert result.standard_uncertainties["intensity_scale"] == pytest.approx(expected_se, rel=1e-7)
    assert result.metrics["negative_intensities"] == len(measured[::7])
    positive = measured > 0
    positive_only_scale = np.sum(measured[positive] * ideal[positive] / sigma[positive]**2) / np.sum(ideal[positive]**2 / sigma[positive]**2)
    assert abs(result.scale - positive_only_scale) > .05


def test_holdout_observations_never_affect_optimized_model_or_scale():
    truth = molecular_structure()
    hkl = indices(2)
    data = Reflections(hkl, calculated_intensities(truth, hkl, scale=1.3), np.ones(len(hkl)))
    data.free_flags = grouped_holdout_flags(truth, data, .2, seed=18)
    start = copy.deepcopy(truth)
    start.atoms[1].frac += [.005, -.005, .005]
    one = refine(start, data)
    altered = copy.deepcopy(data)
    altered.intensity[data.free_flags] += 10000
    altered.sigma[data.free_flags] *= 3
    two = refine(start, altered)
    assert np.array_equal(one.parameters, two.parameters)
    assert one.scale == two.scale
    assert two.metrics["free"]["wr2"] > .5
    assert one.metrics["free"]["wr2"] < 1e-7


def test_mixed_flags_for_friedel_mates_are_rejected():
    structure = molecular_structure()
    data = Reflections([[1, 2, 3], [-1, -2, -3]], [10, 11], [1, 1], free_flags=[False, True])
    with pytest.raises(ValueError, match="Holdout leakage"):
        refine(structure, data)
    flags = grouped_holdout_flags(structure, Reflections([[1, 2, 3], [-1, -2, -3], [2, 1, 3]], [10, 11, 12], [1, 1, 1]), .4)
    assert flags[0] == flags[1]


def test_cholesky_anisotropic_recovery_in_an_oblique_cell():
    truth = molecular_structure()
    u = np.array([[.027, .004, -.003], [.004, .044, .007], [-.003, .007, .035]])
    truth.atoms[1].u_aniso = u
    for atom in truth.atoms:
        atom.refine_xyz = False
        atom.refine_u = atom.label == "O1"
    hkl = indices(4)
    data = Reflections(hkl, gemmi_intensities(truth, hkl, 2.2), np.ones(len(hkl)))
    start = copy.deepcopy(truth)
    start.atoms[1].u_aniso = np.diag([.04, .04, .04])
    result = refine(start, data)
    assert result.success and result.metrics["identifiable"]
    assert np.allclose(result.structure.atoms[1].u_aniso, u, atol=2e-7)
    assert np.min(np.linalg.eigvalsh(result.structure.atoms[1].u_aniso)) > 0
    assert any("log_L11" in name for name in result.parameter_names)
    assert result.standard_uncertainties["O1.U12_cart"] > 0


def test_special_position_tangent_and_adp_symmetry_are_exact():
    cell = Cell(10, 11, 12, 90, 103, 90)
    # On a twofold axis, x=z=0 and Uxy=Uyz=0. Polar y is an origin freedom.
    tensor = np.array([[.027, 0, .004], [0, .044, 0], [.004, 0, .035]])
    truth = Structure(cell, [Atom("Zn1", "Zn", [0, .17, 0], u_aniso=tensor),
                             Atom("O1", "O", [.23, .34, .39], u_iso=.031)], "P 1 2 1")
    truth.atoms[1].refine_u = False
    hkl = indices(3)
    data = Reflections(hkl, gemmi_intensities(truth, hkl, 1.6), np.ones(len(hkl)))
    start = copy.deepcopy(truth)
    start.atoms[0].u_aniso = np.diag([.04, .04, .04])
    start.atoms[1].frac += [.005, -.005, .005]
    result = refine(start, data)
    assert result.success and result.metrics["identifiable"]
    assert result.gauge["dimension"] == 1
    assert np.array_equal(result.structure.atoms[0].frac, [0, .17, 0])
    refined_u = result.structure.atoms[0].u_aniso
    assert abs(refined_u[0, 1]) < 1e-12 and abs(refined_u[1, 2]) < 1e-12
    assert np.allclose(refined_u, tensor, atol=3e-7)
    assert sum("log_U_mode" in name for name in result.parameter_names) == 4


def test_inversion_twin_fraction_recovered_from_anomalous_data():
    structure = Structure(Cell(8, 9, 10), [
        Atom("Fe1", "Fe", [.12, .23, .34], f_prime=-1.1, f_double_prime=3.2, refine_xyz=False, refine_u=False),
        Atom("O1", "O", [.27, .38, .45], refine_xyz=False, refine_u=False),
        Atom("C1", "C", [.43, .51, .26], refine_xyz=False, refine_u=False),
    ])
    hkl = np.vstack([indices(3), -indices(3)])
    fplus, fminus = structure_factors(structure, hkl), structure_factors(structure, -hkl)
    obs = 1.7 * (.77 * abs(fplus)**2 + .23 * abs(fminus)**2)
    data = Reflections(hkl, obs, np.ones(len(hkl)))
    options = RefinementOptions(twin_laws=[(-np.eye(3, dtype=int)).tolist()], twin_fractions=[.55, .45], refine_twin_fractions=True)
    result = refine(structure, data, options)
    assert result.success and result.metrics["identifiable"]
    assert result.twin_fractions == pytest.approx([.77, .23], abs=1e-8)
    assert result.scale == pytest.approx(1.7, abs=1e-8)
    assert result.standard_uncertainties["twin_fraction[1]"] > 0


def test_inversion_twin_without_anomalous_signal_is_reported_unidentifiable():
    structure = molecular_structure()
    for atom in structure.atoms:
        atom.refine_xyz = atom.refine_u = False
    hkl = indices(2)
    data = Reflections(hkl, calculated_intensities(structure, hkl), np.ones(len(hkl)))
    options = RefinementOptions(twin_laws=[(-np.eye(3, dtype=int)).tolist()], twin_fractions=[.7, .3], refine_twin_fractions=True)
    result = refine(structure, data, options)
    assert not result.metrics["identifiable"]
    assert result.covariance is None
    assert result.nullspace.shape[0] >= 1
    assert result.standard_uncertainties["twin_fraction[1]"] is None


def test_disorder_constraints_preserve_exact_coordinate_adp_and_occupancy_sums():
    structure = Structure(Cell(8, 9, 10), [
        Atom("Zn1", "Zn", [.11, .21, .31], occupancy=.65, refine_occupancy=True, refine_u=False),
        Atom("Co1", "Co", [.11, .21, .31], occupancy=.35, refine_occupancy=True, refine_u=False),
        Atom("O1", "O", [.27, .38, .45], refine_u=False),
        Atom("C1", "C", [.43, .51, .26], refine_u=False),
    ])
    constraints = [{"type": "equal_xyz", "atoms": ["Zn1", "Co1"]},
                   {"type": "equal_adp", "atoms": ["Zn1", "Co1"]},
                   {"type": "occupancy_sum", "atoms": ["Zn1", "Co1"], "target": 1}]
    hkl = indices(3)
    data = Reflections(hkl, gemmi_intensities(structure, hkl, 1.4), np.ones(len(hkl)))
    start = copy.deepcopy(structure)
    start.atoms[0].occupancy = start.atoms[1].occupancy = .5
    start.atoms[2].frac += [.004, -.005, .004]
    result = refine(start, data, constraints=constraints)
    assert result.success and result.metrics["identifiable"]
    first, second = result.structure.atoms[:2]
    assert np.array_equal(first.frac, second.frac)
    assert first.u_iso == second.u_iso
    assert first.occupancy + second.occupancy == pytest.approx(1, abs=1e-14)
    assert first.occupancy == pytest.approx(.65, abs=1e-5)


def test_all_occupancies_and_scale_free_is_rejected_before_optimization():
    structure = molecular_structure()
    for atom in structure.atoms:
        atom.refine_occupancy = True
    hkl = indices(2)
    data = Reflections(hkl, calculated_intensities(structure, hkl), np.ones(len(hkl)))
    with pytest.raises(ValueError, match="All occupancies"):
        refine(structure, data)


def test_equal_adp_and_equal_occupancy_constraints_remove_degrees_of_freedom():
    truth = molecular_structure()
    for atom in truth.atoms:
        atom.refine_xyz = atom.refine_u = atom.refine_occupancy = False
    for atom in truth.atoms[:2]:
        atom.u_iso = .026
        atom.occupancy = .7
        atom.refine_u = atom.refine_occupancy = True
    hkl = indices(3)
    data = Reflections(hkl, calculated_intensities(truth, hkl, scale=1.4), np.ones(len(hkl)))
    start = copy.deepcopy(truth)
    for atom in start.atoms[:2]:
        atom.u_iso, atom.occupancy = .04, .5
    constraints = [{"type": "equal_adp", "atoms": ["C1", "O1"]},
                   {"type": "equal_occupancy", "atoms": ["C1", "O1"]}]
    result = refine(start, data, constraints=constraints)
    assert result.success and result.metrics["identifiable"]
    assert result.metrics["n_parameters"] == 3
    assert result.structure.atoms[0].u_iso == result.structure.atoms[1].u_iso
    assert result.structure.atoms[0].occupancy == result.structure.atoms[1].occupancy
    assert result.structure.atoms[0].u_iso == pytest.approx(.026, abs=1e-8)
    assert result.structure.atoms[0].occupancy == pytest.approx(.7, abs=1e-8)
    assert result.to_dict()["constraints"] == constraints


def test_without_origin_constraint_rank_deficiency_is_reported():
    structure = molecular_structure()
    hkl = indices(2)
    data = Reflections(hkl, calculated_intensities(structure, hkl), np.ones(len(hkl)))
    result = refine(structure, data, RefinementOptions(origin_gauge="none"))
    assert not result.metrics["identifiable"]
    assert result.metrics["n_parameters"] - result.metrics["jacobian_rank"] >= 3
    assert result.covariance is None


def test_restraints_have_physical_units_and_explicit_images():
    structure = Structure(Cell(10, 10, 10), [
        Atom("A", "C", [.95, .2, .3], u_aniso=np.diag([.02, .03, .04])),
        Atom("B", "C", [.05, .2, .3], u_aniso=np.diag([.025, .03, .04])),
        Atom("C", "C", [.05, .3, .3]),
        Atom("D", "C", [.95, .3, .31]),
    ])
    distance = {"type": "distance", "atoms": ["A", "B"], "target": 1, "sigma": .01}
    assert abs(restraint_residuals(structure, [distance])[0]) < 1e-12
    explicit = copy.deepcopy(distance)
    explicit["atoms"][1] = {"label": "B", "translation": [1, 0, 0]}
    assert abs(restraint_residuals(structure, [explicit])[0]) < 1e-12
    angle = {"type": "angle", "atoms": ["A", "B", "C"], "target": 90, "sigma": 2}
    assert abs(restraint_residuals(structure, [angle])[0]) < 1e-12
    rigid = {"type": "rigid_bond", "atoms": ["A", "B"], "sigma": .001}
    assert restraint_residuals(structure, [rigid])[0] == pytest.approx(-5)
    plane = {"type": "planarity", "atoms": ["A", "B", "C", "D"], "sigma": .01}
    assert np.linalg.norm(restraint_residuals(structure, [plane])) > 1


def test_unknown_restraint_or_wrong_symmetry_operation_is_not_ignored():
    structure = molecular_structure()
    with pytest.raises(ValueError, match="Unsupported"):
        Restraint.from_dict({"type": "riding_hydrogen", "atoms": ["C1"], "sigma": .01})
    with pytest.raises(ValueError, match="absent"):
        restraint_residuals(structure, [{"type": "distance", "atoms": ["C1", {"label": "O1", "symop": "-x,-y,-z"}], "target": 1.2, "sigma": .02}])


def test_distance_restraint_changes_a_weakly_determined_model():
    truth = molecular_structure()
    for atom in truth.atoms:
        atom.refine_u = False
    hkl = indices(2)
    data = Reflections(hkl, calculated_intensities(truth, hkl), np.full(len(hkl), 10000.0))
    target = np.linalg.norm(truth.cell.to_cartesian(truth.cell.minimum_image(truth.atoms[1].frac - truth.atoms[0].frac))) + .1
    result = refine(truth, data, restraints=[{"type": "distance", "atoms": ["C1", "O1"], "target": target, "sigma": .001}])
    measured = np.linalg.norm(truth.cell.to_cartesian(truth.cell.minimum_image(result.structure.atoms[1].frac - result.structure.atoms[0].frac)))
    assert measured == pytest.approx(target, abs=1e-5)
    assert result.metrics["restraint_residual_count"] == 1


def test_fixed_model_and_absolute_covariance_do_not_invent_precision():
    structure = molecular_structure()
    for atom in structure.atoms:
        atom.refine_xyz = atom.refine_u = False
    hkl = indices(2)
    data = Reflections(hkl, calculated_intensities(structure, hkl), np.ones(len(hkl)))
    result = refine(structure, data, RefinementOptions(scale=1, refine_scale=False))
    assert result.success
    assert result.metrics["n_parameters"] == 0
    assert all(x is None for x in result.standard_uncertainties.values())


def test_irls_uses_frozen_weights_and_records_remaining_weight_change():
    structure = molecular_structure()
    for atom in structure.atoms:
        atom.refine_xyz = atom.refine_u = False
    hkl = indices(2)
    expected = calculated_intensities(structure, hkl, scale=1.6)
    expected[::5] *= 1.2
    data = Reflections(hkl, expected, np.ones(len(hkl)))
    result = refine(structure, data, RefinementOptions(weight_a=.02, weight_b=.1, weight_cycles=5))
    assert result.success
    assert result.metrics["weighting"] == "frozen_weight_IRLS"
    assert result.metrics["final_weight_relative_change"] < 1e-3
    assert len(result.history) >= 3


@pytest.mark.parametrize("bad", [
    {"max_nfev": 0}, {"scale": -1}, {"weight_a": -1}, {"origin_gauge": "guess"},
    {"refine_scale": False}, {"covariance_scale": "automatic"},
])
def test_invalid_refinement_options_fail_explicitly(bad):
    structure = molecular_structure()
    data = Reflections([[1, 2, 3]], [10], [1])
    with pytest.raises(ValueError):
        refine(structure, data, bad)
