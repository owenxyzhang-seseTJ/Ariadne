"""Independent numerical/physical checks of the native scattering core."""

import copy
import itertools

import gemmi
import numpy as np
import pytest

from ariadne_xtal.model import Atom, Cell, Reflections, Structure
from ariadne_xtal.scattering import (
    atomic_form_factor, calculated_intensities, intensity_derivative,
    structure_factors, validate_twin_laws,
)
from ariadne_xtal.symmetry import (
    get_operations, rank_spacegroups, site_stabilizer,
    systematic_absences, unique_sites,
)


def test_nonorthogonal_reciprocal_geometry_matches_gemmi():
    cell = Cell(7.2, 8.3, 9.4, 75, 103, 68)
    gc = gemmi.UnitCell(7.2, 8.3, 9.4, 75, 103, 68)
    h = np.array([[1, 2, 3], [-2, 1, 0], [5, -4, 2]])
    assert np.allclose(cell.matrix.T @ cell.reciprocal, np.eye(3), atol=1e-13)
    assert cell.volume == pytest.approx(gc.volume, rel=1e-13)
    assert np.allclose(cell.d_spacing(h), [gc.calculate_d(x.tolist()) for x in h], rtol=1e-13)
    coords = np.array([[.12, -.3, 1.8], [.4, .1, .3]])
    assert np.allclose(cell.to_fractional(cell.to_cartesian(coords)), coords, atol=1e-14)


def test_minimum_image_is_exact_for_skew_lattice():
    cell = Cell(10, 10, 11, 90, 90, 13)
    rng = np.random.default_rng(110)
    displacements = rng.uniform(-2, 2, size=(50, 3))
    images = cell.minimum_image(displacements)
    shifts = np.array(list(itertools.product(range(-6, 7), repeat=3)))
    for original, image in zip(displacements, images):
        brute2 = np.min(np.sum(cell.to_cartesian(original - shifts) ** 2, axis=1))
        assert np.sum(cell.to_cartesian(image) ** 2) == pytest.approx(brute2, abs=1e-10)
    probe = np.array([.49, .49, .1])
    assert np.linalg.norm(cell.to_cartesian(cell.minimum_image(probe))) < np.linalg.norm(cell.to_cartesian(probe)) / 3


@pytest.mark.parametrize("args", [(0, 1, 2), (1, 1, 1, 5, 5, 170), (1, 2, 3, 90, 90, 0)])
def test_cell_rejects_invalid_metrics(args):
    with pytest.raises(ValueError):
        Cell(*args)


def test_structure_roundtrip_and_explicit_symmetry_authority():
    s = Structure(Cell(8, 9, 10, 81, 77, 69), [Atom("C1", "c", [.1, .2, .3])],
                  spacegroup="P 21/c", sym_ops=["x,y,z"])
    rebuilt = Structure.from_dict(s.to_dict())
    assert rebuilt.to_dict() == s.to_dict()
    assert len(rebuilt.operations) == 1
    assert rebuilt.atoms[0].element == "C"


def test_explicit_operation_groups_are_checked():
    with pytest.raises(ValueError, match="closed"):
        get_operations("P 1", ["x,y,z", "x,y+1/3,z"])
    with pytest.raises(ValueError, match="duplicate"):
        get_operations("P 1", ["x,y,z", "x+1,y,z"])
    with pytest.raises(ValueError, match="identity"):
        get_operations("P 1", ["-x,-y,-z"])
    with pytest.raises(ValueError, match="incompatible"):
        Structure(Cell(7, 8, 9), [], "P 4")


def test_special_position_occupancy_and_tensor_projection():
    u = np.array([[.04, .006, .004], [.006, .03, .005], [.004, .005, .02]])
    a = Atom("C1", "C", [.0, .31, .0], occupancy=.7, u_aniso=u)
    s = Structure(Cell(10, 11, 12), [a], "P 1 2 1")
    sites = unique_sites(s, a)
    assert len(sites) == 1 and len(site_stabilizer(s, a)) == 2
    assert sites[0][1][0, 1] == pytest.approx(0, abs=1e-15)
    assert sites[0][1][1, 2] == pytest.approx(0, abs=1e-15)
    assert sites[0][1][0, 2] == pytest.approx(.004)
    assert structure_factors(s, [0, 0, 0]) == pytest.approx(.7 * atomic_form_factor("C", 0))


def _gemmi_structure(s):
    g = gemmi.SmallStructure()
    g.cell = gemmi.UnitCell(**s.cell.to_dict())
    g.spacegroup = gemmi.find_spacegroup_by_name(s.spacegroup, s.cell.alpha, s.cell.gamma)
    g.setup_cell_images()
    for atom in s.atoms:
        site = gemmi.SmallStructure.Site()
        site.label, site.element = atom.label, gemmi.Element(atom.element)
        site.fract = gemmi.Fractional(*atom.frac)
        site.occ, site.u_iso = atom.occupancy, atom.u_iso
        if atom.u_aniso is not None:
            inv_a = np.linalg.inv(s.cell.matrix)
            u_star = inv_a @ atom.u_aniso @ inv_a.T
            reciprocal_lengths = np.linalg.norm(s.cell.reciprocal, axis=0)
            u_cif = u_star / np.outer(reciprocal_lengths, reciprocal_lengths)
            site.aniso = gemmi.SMat33d(u_cif[0, 0], u_cif[1, 1], u_cif[2, 2],
                                      u_cif[0, 1], u_cif[0, 2], u_cif[1, 2])
        g.add_site(site)
    # Gemmi sums symmetry images, so its SF API needs crystallographic SOF.
    g.change_occupancies_to_crystallographic(max_dist=1e-6)
    return g


@pytest.mark.parametrize("spacegroup, cell", [
    ("P 1", Cell(8.1, 9.2, 10.3, 72, 98, 79)),
    ("P -1", Cell(8.1, 9.2, 10.3, 72, 98, 79)),
    ("P 21/c", Cell(8.1, 9.2, 10.3, 90, 98, 90)),
    ("P 21 21 21", Cell(8.1, 9.2, 10.3)),
    ("P 4", Cell(8.1, 8.1, 10.3)),
    ("F m -3 m", Cell(8.1, 8.1, 8.1)),
])
def test_independent_gemmi_structure_factor_comparison(spacegroup, cell):
    atoms = [Atom("C1", "C", [.131, .267, .319], u_iso=.031),
             Atom("Zn1", "Zn", [.0, .0, .0], occupancy=.83, u_iso=.022)]
    s = Structure(cell, atoms, spacegroup)
    hkl = np.array(list(itertools.product(range(-2, 3), repeat=3)))
    g = _gemmi_structure(s)
    calc = gemmi.StructureFactorCalculatorX(g.cell)
    expected = np.array([calc.calculate_sf_from_small_structure(g, h.tolist()) for h in hkl])
    assert np.allclose(structure_factors(s, hkl), expected, rtol=3e-6, atol=2e-5)


@pytest.mark.parametrize("spacegroup", ["P 1", "P 21/c"])
def test_nonorthogonal_anisotropic_sf_against_gemmi(spacegroup):
    cell = Cell(8, 9, 10, 90, 103, 90)
    u = np.array([[.03, .003, -.002], [.003, .04, .005], [-.002, .005, .02]])
    s = Structure(cell, [Atom("Fe1", "Fe", [.123, .287, .313], u_aniso=u)], spacegroup)
    hkl = np.array(list(itertools.product(range(-3, 4), repeat=3)))
    g = _gemmi_structure(s)
    calc = gemmi.StructureFactorCalculatorX(g.cell)
    expected = np.array([calc.calculate_sf_from_small_structure(g, h.tolist()) for h in hkl])
    assert np.allclose(structure_factors(s, hkl), expected, rtol=3e-6, atol=2e-5)


def test_all_230_groups_systematic_absences_match_gemmi():
    hkl = np.array(list(itertools.product(range(-2, 3), repeat=3)))
    for number in range(1, 231):
        sg = gemmi.get_spacegroup_reference_setting(number)
        expected = np.array([sg.operations().is_systematically_absent(h.tolist()) for h in hkl])
        assert np.array_equal(systematic_absences(sg, hkl), expected), sg.xhm()


def test_friedel_law_and_explicit_anomalous_breaking():
    hkl = np.array([[1, 2, 3], [2, 1, 0], [4, 1, 2]])
    s = Structure(Cell(8, 9, 10), [Atom("Fe1", "Fe", [.11, .21, .31]),
                                  Atom("C1", "C", [.32, .43, .53])])
    assert np.allclose(structure_factors(s, -hkl), np.conjugate(structure_factors(s, hkl)))
    s.atoms[0].f_double_prime = 3.2
    assert not np.allclose(np.abs(structure_factors(s, -hkl)), np.abs(structure_factors(s, hkl)))


def test_global_origin_shift_has_no_effect_on_p1_intensities():
    s = Structure(Cell(8, 9, 10), [Atom("Fe1", "Fe", [.11, .21, .31]),
                                  Atom("C1", "C", [.32, .43, .53])])
    hkl = np.array([[1, 2, 3], [2, 1, 0], [4, 1, 2]])
    before = calculated_intensities(s, hkl)
    for a in s.atoms:
        a.frac += [.127, -.317, .613]
    assert np.allclose(calculated_intensities(s, hkl), before, rtol=1e-13)


def test_anisotropic_symmetry_matches_explicit_p1_expansion():
    u = np.array([[.04, .006, .004], [.006, .03, .005], [.004, .005, .02]])
    s = Structure(Cell(10, 11, 12, 90, 103, 90),
                  [Atom("C1", "C", [.17, .27, .37], u_aniso=u)], "P 21/c")
    expanded = Structure(s.cell, [Atom(f"C{i}", "C", xyz, u_aniso=ucart)
                                  for i, (xyz, ucart) in enumerate(unique_sites(s, s.atoms[0]))])
    hkl = np.array(list(itertools.product(range(-2, 3), repeat=3)))
    assert np.allclose(structure_factors(s, hkl), structure_factors(expanded, hkl), atol=1e-12)


def test_twins_are_incoherent_with_implicit_identity():
    s = Structure(Cell(8, 9, 10), [Atom("Fe1", "Fe", [.11, .21, .31], f_double_prime=3),
                                  Atom("C1", "C", [.32, .43, .53])])
    hkl = np.array([[1, 2, 3], [2, 1, 0], [4, 1, 2]])
    inversion = -np.eye(3, dtype=int)
    expected = 2.3 * (.7 * np.abs(structure_factors(s, hkl))**2 + .3 * np.abs(structure_factors(s, -hkl))**2)
    got = calculated_intensities(s, hkl, scale=2.3, twin_laws=[inversion], twin_fractions=[.7, .3])
    assert np.allclose(got, expected)
    with pytest.raises(ValueError, match="sum"):
        calculated_intensities(s, hkl, twin_laws=[inversion], twin_fractions=[.7, .4])
    with pytest.raises(ValueError, match="metric"):
        validate_twin_laws([[[0, 1, 0], [1, 0, 0], [0, 0, 1]]], [.5, .5], s.cell)
    with pytest.raises(ValueError, match="nonidentity"):
        validate_twin_laws([np.eye(3)], [.5, .5], s.cell)


def test_analytic_intensity_derivative_agrees_with_finite_difference():
    s = Structure(Cell(8, 9, 10), [Atom("Fe1", "Fe", [.11, .21, .31]),
                                  Atom("C1", "C", [.32, .43, .53])])
    hkl = np.array([[1, 2, 3], [2, 1, 0], [4, 1, 2]])
    carbon = Structure(s.cell, [copy.deepcopy(s.atoms[1])])
    df = 2j * np.pi * hkl[:, 0] * structure_factors(carbon, hkl)
    expected = intensity_derivative(structure_factors(s, hkl), df)
    eps = 1e-6
    plus, minus = copy.deepcopy(s), copy.deepcopy(s)
    plus.atoms[1].frac[0] += eps
    minus.atoms[1].frac[0] -= eps
    finite = (calculated_intensities(plus, hkl) - calculated_intensities(minus, hkl)) / (2 * eps)
    assert np.allclose(expected, finite, rtol=1e-8, atol=1e-6)


def test_reflections_preserve_negative_intensity_and_holdout_flags():
    r = Reflections([[1, 2, 3], [0, 1, 0]], [-3, 10], [2, 1], free_flags=[False, True])
    assert r.intensity[0] == -3
    assert r.subset([1]).free_flags[0]
    assert Reflections.from_dict(r.to_dict()).to_dict() == r.to_dict()
    for args in [([[1, 2, 3]], [1], [0]), ([[.5, 1, 2]], [1], [1]), ([], [], [])]:
        with pytest.raises(ValueError):
            Reflections(*args)


def test_extinction_rank_reports_ambiguity_instead_of_a_spacegroup_claim():
    r = Reflections([[0, 1, 0], [0, 2, 0], [1, 1, 1]], [100, 100, 50], [1, 1, 1])
    rows = rank_spacegroups(Cell(8, 9, 10, 90, 103, 90), r, ["P 1", "P 21/c", "P 4"])
    assert rows[0]["spacegroup"] == "P 1"
    assert rows[1]["n_absence_violations_3sigma"] >= 1
    assert rows[-1]["compatible_metric"] is False
    assert all(row["identification"] == "not_determined" for row in rows)


def test_invalid_atoms_rejected_before_scattering():
    with pytest.raises(ValueError):
        Atom("X1", "X", [0, 0, 0])
    with pytest.raises(ValueError):
        Atom("Ca1", "carbon", [0, 0, 0])
    with pytest.raises(ValueError):
        Atom("C1", "C", [0, 0, 0], occupancy=2)
    with pytest.raises(ValueError):
        Atom("C1", "C", [0, 0, 0], u_aniso=np.diag([.1, -.1, .1]))
    with pytest.raises(ValueError):
        Structure(Cell(10, 10, 10), [Atom("C1", "C", [0, 0, 0]), Atom("c1", "C", [.2, .3, .4])])
