"""Physics and delivery checks for scientific visualization."""

import re

import numpy as np
import pytest
from scipy.stats import chi2

from ariadne_xtal.maps import DensityMap
from ariadne_xtal.model import Atom, Cell, Reflections, Structure
from ariadne_xtal.scattering import calculated_intensities
from ariadne_xtal.visualization import (
    create_dashboard, density_surface_mesh, ellipsoid_mesh, expanded_sites,
    parameter_correlation, static_figures, unit_cell_edges,
)


def small_data():
    structure = Structure(Cell(8, 9, 10, 90, 102, 90),
                          [Atom("C1", "C", [.2, .3, .4]), Atom("O1", "O", [.3, .35, .5])])
    hkl = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 1], [2, 1, 0], [2, 2, 1]])
    intensities = calculated_intensities(structure, hkl)
    intensities[0] = -2
    return structure, Reflections(hkl, intensities, np.ones(len(hkl)), free_flags=[0, 0, 0, 0, 1])


def test_probability_ellipsoid_has_exact_mahalanobis_radius():
    angle = .61
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    u = rotation @ np.diag([.01, .04, .09]) @ rotation.T
    center = np.array([1.2, -.3, 2.1])
    vertices, triangles, radii = ellipsoid_mesh(center, u, probability=.5)
    delta = vertices - center
    mahalanobis2 = np.einsum("ni,ij,nj->n", delta, np.linalg.inv(u), delta)
    assert np.allclose(mahalanobis2, chi2.ppf(.5, 3), rtol=1e-12)
    assert np.allclose(radii, np.sqrt(chi2.ppf(.5, 3) * np.array([.01, .04, .09])))
    assert triangles.min() >= 0 and triangles.max() < len(vertices)
    with pytest.raises(ValueError):
        ellipsoid_mesh(center, np.diag([.01, -.02, .03]))


def test_skew_cell_edges_and_special_position_visualization():
    cell = Cell(8, 9, 10, 90, 102, 90)
    edges = unit_cell_edges(cell)
    assert len(edges) == 12
    lengths = sorted(np.linalg.norm(edge[1] - edge[0]) for edge in edges)
    assert np.allclose(lengths, [8] * 4 + [9] * 4 + [10] * 4)
    structure = Structure(cell, [Atom("Fe1", "Fe", [0, 0, 0])], "P -1")
    assert len(expanded_sites(structure)) == 1
    with pytest.raises(ValueError, match="budget"):
        expanded_sites(Structure(cell, [Atom("C1", "C", [.1, .2, .3])], "P -1"), 1)


def test_density_mesh_uses_fractional_grid_origin_and_skew_cartesian_transform():
    cell = Cell(8, 9, 10, 90, 102, 90)
    n = 16
    frac = np.indices((n, n, n)).transpose(1, 2, 3, 0) / n
    values = np.sin(2 * np.pi * frac[..., 0])
    density = DensityMap(values, cell)
    vertices, triangles = density_surface_mesh(density, .5)
    recovered = cell.to_fractional(vertices)
    assert np.all(recovered >= -1e-7) and np.all(recovered <= 1 + 1e-7)
    # Linear interpolation between x samples should produce two x-plane sheets.
    assert len(np.unique(np.round(recovered[:, 0], 5))) == 2
    assert np.allclose(np.sort(np.unique(np.round(recovered[:, 0], 5))), [1 / 12, 5 / 12], atol=.002)
    assert np.max(recovered[:, 1]) == pytest.approx(1)
    assert np.max(recovered[:, 2]) == pytest.approx(1)
    assert len(triangles) > 0


def test_correlation_is_dimensionless_and_undefined_zero_variance_stays_blank():
    cov = np.array([[4., 1., 0.], [1., 1., 0.], [0., 0., 0.]])
    correlation = parameter_correlation(cov)
    assert correlation[0, 1] == pytest.approx(.5)
    assert np.isnan(correlation[2, 2])


def test_dashboard_is_offline_and_escapes_labels(tmp_path):
    structure, reflections = small_data()
    structure.atoms[0].label = '<script>alert("x")</script>'
    path = tmp_path / "dashboard.html"
    create_dashboard(structure, reflections, path, history=[{"outer_cycle": 0, "chi_squared": 12},
                                                           {"outer_cycle": 1, "chi_squared": 4}])
    text = path.read_text()
    assert not re.search(r"<script[^>]+src\s*=", text, re.IGNORECASE)
    assert "plotly.js" in text.lower()
    assert 'alert("x")</script>' not in text
    assert text.count('id="ariadne-figure-0"') == 1
    assert "negative intensities remain visible" in text
    assert path.stat().st_size > 1_000_000


def test_static_figures_export_vector_documents(tmp_path):
    structure, reflections = small_data()
    paths = static_figures(structure, reflections, tmp_path,
                           history=[{"outer_cycle": 0, "chi_squared": 12}, {"outer_cycle": 1, "chi_squared": 4}])
    assert set(paths) == {"diagnostics_svg", "diagnostics_pdf", "structure_svg", "structure_pdf"}
    for path in paths.values():
        data = open(path, "rb").read()
        assert len(data) > 1000
        if path.endswith(".pdf"):
            assert data.startswith(b"%PDF")
        else:
            assert b"<svg" in data
