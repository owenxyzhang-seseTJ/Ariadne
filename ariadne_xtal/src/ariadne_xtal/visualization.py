"""Offline crystallographic graphics with physical Cartesian geometry.

The ellipsoid encloses a stated probability of a three-dimensional Gaussian
atomic displacement, not a confidence region for the refined coordinates.
Density meshes are extracted on the fractional grid and transformed by the
full cell matrix.  Contacts are geometric candidates, not inferred valences.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
from scipy.stats import chi2

from .model import Reflections, Structure
from .scattering import calculated_intensities
from .symmetry import unique_sites


CHARCOAL = "#343139"
BURGUNDY = "#943b4a"
OCHRE = "#b18340"
PLUM = "#76576f"
SILVER = "#b6b1ae"
PAPER = "#fcfaf7"
ELEMENT_COLORS = {"C": CHARCOAL, "H": SILVER, "D": SILVER, "O": BURGUNDY,
                  "N": PLUM, "S": OCHRE, "P": "#a66b4f", "Cl": "#8b7b59"}


def _element_color(element: str) -> str:
    return ELEMENT_COLORS.get(element, OCHRE if gemmi.Element(element).is_metal else "#807775")


def ellipsoid_mesh(center: Any, u_cart: Any, probability: float = 0.5,
                   resolution: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cartesian vertices, triangle indices and sorted principal semiaxes.

    A Gaussian displacement u has u^T U^{-1}u ~ chi-square(3), hence the
    semiaxes are sqrt(chi2.ppf(probability, 3) * eigenvalues(U)).
    """
    if not np.isfinite(probability) or not 0 < probability < 1:
        raise ValueError("ellipsoid probability must lie strictly between 0 and 1")
    if not isinstance(resolution, (int, np.integer)) or not 6 <= resolution <= 64:
        raise ValueError("ellipsoid mesh resolution must be an integer from 6 to 64")
    center = np.asarray(center, dtype=float)
    u = np.asarray(u_cart, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("ellipsoid center must be a finite Cartesian length-three vector")
    if u.shape != (3, 3) or not np.isfinite(u).all() or not np.allclose(u, u.T, atol=1e-12):
        raise ValueError("ellipsoid U must be a finite symmetric Cartesian matrix")
    eigenvalues, eigenvectors = np.linalg.eigh(u)
    if eigenvalues.min() < -1e-12:
        raise ValueError("a negative ADP eigenvalue cannot define a probability ellipsoid")
    radii = np.sqrt(chi2.ppf(probability, 3) * np.maximum(eigenvalues, 0))
    theta = np.linspace(0, np.pi, resolution + 1)
    phi = np.linspace(0, 2 * np.pi, 2 * resolution, endpoint=False)
    sphere = np.stack(np.broadcast_arrays(
        np.sin(theta)[:, None] * np.cos(phi)[None, :],
        np.sin(theta)[:, None] * np.sin(phi)[None, :],
        np.cos(theta)[:, None] * np.ones((1, len(phi))),
    ), axis=-1).reshape(-1, 3)
    vertices = (sphere * radii) @ eigenvectors.T + center
    triangles = []
    m = len(phi)
    for i in range(resolution):
        for j in range(m):
            a, b = i * m + j, i * m + (j + 1) % m
            c, d = (i + 1) * m + j, (i + 1) * m + (j + 1) % m
            if i > 0:
                triangles.append((a, c, b))
            if i < resolution - 1:
                triangles.append((b, c, d))
    return vertices, np.asarray(triangles, dtype=int), radii


def expanded_sites(structure: Structure, maximum_sites: int = 512) -> list[dict[str, Any]]:
    """Return one full conventional cell of independent symmetry images."""
    if maximum_sites < 1:
        raise ValueError("maximum_sites must be positive")
    sites = []
    for atom in structure.atoms:
        for image_number, (frac, u_cart) in enumerate(unique_sites(structure, atom), 1):
            sites.append({"label": f"{atom.label}:{image_number}", "source_label": atom.label,
                          "element": atom.element, "frac": frac,
                          "cart": structure.cell.to_cartesian(frac), "u_cart": u_cart,
                          "occupancy": atom.occupancy, "color": _element_color(atom.element)})
            if len(sites) > maximum_sites:
                raise ValueError(f"expanded model exceeds visualization budget ({maximum_sites} sites); select a smaller fragment")
    return sites


def unit_cell_edges(cell: Any) -> list[np.ndarray]:
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    cart = cell.to_cartesian(corners)
    return [cart[[i, j]] for i in range(8) for j in range(i + 1, 8)
            if np.sum(np.abs(corners[i] - corners[j])) == 1]


def geometric_contacts(structure: Structure, sites: list[dict[str, Any]],
                       maximum_contacts: int = 3000) -> list[np.ndarray]:
    """Metric-aware periodic contacts within 1.20 times covalent radii sum.

    Each segment starts at a visible site and ends at its nearest periodic
    image of the second site.  Segments can cross a cell face.  Disorder
    occupancy, chemistry and bond order are not inferred by this display.
    """
    contacts = []
    radii = [gemmi.Element(s["element"]).covalent_r for s in sites]
    for i, first in enumerate(sites):
        for j in range(i + 1, len(sites)):
            second = sites[j]
            delta = structure.cell.minimum_image(second["frac"] - first["frac"])
            vector = structure.cell.to_cartesian(delta)
            distance = np.linalg.norm(vector)
            if .4 < distance < 1.2 * (radii[i] + radii[j]):
                contacts.append(np.array([first["cart"], first["cart"] + vector]))
                if len(contacts) > maximum_contacts:
                    raise ValueError("geometric-contact count exceeds display budget; inspect the model for overlaps")
    return contacts


def density_surface_mesh(density: Any, level: float,
                         maximum_vertices: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
    """Extract a periodic cell surface, retaining its physical skew geometry."""
    from skimage.measure import marching_cubes

    values = np.asarray(density.values, dtype=float)
    if values.ndim != 3 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError("density must be a finite 3D grid of at least two samples per axis")
    if values.size > 2_097_152:
        raise ValueError("interactive isosurface grid exceeds 2,097,152 voxels; supply a coarser display map")
    if not np.isfinite(level) or not values.min() < level < values.max():
        raise ValueError("isosurface level must lie strictly within the map range")
    # Original sample i represents i/N, so append the periodic face at x=1.
    padded = np.pad(values, [(0, 1)] * 3, mode="wrap")
    frac_vertices, triangles, _, _ = marching_cubes(
        padded, level=level, spacing=tuple(1.0 / np.asarray(values.shape)),
        allow_degenerate=False,
    )
    if len(frac_vertices) > maximum_vertices:
        raise ValueError("isosurface exceeds vertex budget; raise the contour level or provide a coarser map")
    return density.cell.to_cartesian(frac_vertices), triangles


def parameter_correlation(covariance: Any) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or not np.isfinite(covariance).all():
        raise ValueError("covariance must be a finite square matrix")
    if not np.allclose(covariance, covariance.T, atol=1e-10, rtol=1e-8):
        raise ValueError("covariance must be symmetric")
    if np.any(np.diag(covariance) < -1e-12):
        raise ValueError("covariance diagonal must be nonnegative")
    sd = np.sqrt(np.maximum(np.diag(covariance), 0))
    denominator = np.outer(sd, sd)
    correlation = np.divide(covariance, denominator, out=np.full_like(covariance, np.nan), where=denominator > 0)
    finite = np.isfinite(correlation)
    if np.any(np.abs(correlation[finite]) > 1 + 1e-6):
        raise ValueError("covariance gives a correlation outside [-1, 1]")
    return np.clip(correlation, -1, 1)


def _line_trace(segments, name, color, width=2):
    import plotly.graph_objects as go
    xyz = []
    for segment in segments:
        xyz.extend([*segment, [None, None, None]])
    xyz = np.asarray(xyz, dtype=object).reshape(-1, 3)
    return go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines",
                        line={"color": color, "width": width}, name=name,
                        hoverinfo="skip", showlegend=True)


def _scene(title: str) -> dict[str, Any]:
    return {"title": title, "template": "plotly_white", "paper_bgcolor": PAPER,
            "font": {"family": "Arial, sans-serif", "color": CHARCOAL},
            "scene": {"xaxis_title": "x / Å", "yaxis_title": "y / Å", "zaxis_title": "z / Å",
                      "aspectmode": "data", "camera": {"projection": {"type": "orthographic"}}},
            "margin": {"l": 10, "r": 10, "t": 55, "b": 10}, "height": 620}


def structure_figure(structure: Structure, probability: float = .5, density: Any = None):
    import plotly.graph_objects as go

    sites = expanded_sites(structure)
    fig = go.Figure()
    fig.add_trace(_line_trace(unit_cell_edges(structure.cell), "Unit cell", SILVER))
    contacts = geometric_contacts(structure, sites)
    if contacts:
        fig.add_trace(_line_trace(contacts, "Geometric contacts", "#9b9390", 3))
    for site in sites:
        vertices, triangles, radii = ellipsoid_mesh(site["cart"], site["u_cart"], probability)
        hover = (f"{escape(site['label'])} ({escape(site['element'])})<br>"
                 f"occupancy = {site['occupancy']:.4f}<br>"
                 f"fractional = {site['frac'][0]:.5f}, {site['frac'][1]:.5f}, {site['frac'][2]:.5f}<br>"
                 f"ellipsoid semiaxes / Å = {radii[0]:.4f}, {radii[1]:.4f}, {radii[2]:.4f}")
        fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                               i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
                               color=site["color"], opacity=.78, name=escape(site["label"]),
                               hovertemplate=hover + "<extra></extra>", showlegend=False,
                               flatshading=False, lighting={"ambient": .65, "diffuse": .6}))
    if sites:
        coords = np.array([s["cart"] for s in sites])
        fig.add_trace(go.Scatter3d(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode="markers",
                                   marker={"size": 2.2, "color": [s["color"] for s in sites]},
                                   text=[escape(s["label"]) for s in sites], name="Sites",
                                   hovertemplate="%{text}<extra></extra>", showlegend=False))
    if density is not None:
        if density.cell != structure.cell:
            raise ValueError("density and structure must use exactly the same unit cell")
        contour = 2.5 * float(np.std(density.values))
        if contour > 0:
            for level, color in [(contour, OCHRE), (-contour, BURGUNDY)]:
                if density.values.min() < level < density.values.max():
                    vertices, triangles = density_surface_mesh(density, level)
                    label = f"{escape(density.kind)} {level:+.4g} {escape(density.units)}"
                    fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                                           i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
                                           color=color, opacity=.22, name=label, hoverinfo="name",
                                           showlegend=True))
    fig.update_layout(**_scene(f"Displacement ellipsoids · {100 * probability:g}% probability"))
    return fig


def _calculated(structure, reflections, scale, twin_laws, twin_fractions):
    predicted = calculated_intensities(structure, reflections.hkl, scale, twin_laws, twin_fractions)
    residual = (reflections.intensity - predicted) / reflections.sigma
    flags = np.zeros(len(reflections), dtype=bool) if reflections.free_flags is None else reflections.free_flags
    return predicted, residual, flags


def reciprocal_figure(structure, reflections, scale=1., twin_laws=None, twin_fractions=None):
    import plotly.graph_objects as go
    _, residual, flags = _calculated(structure, reflections, scale, twin_laws, twin_fractions)
    q = reflections.hkl @ structure.cell.reciprocal.T
    color_limit = max(3., float(np.quantile(np.abs(residual), .98)))
    text = [f"hkl = ({h}, {k}, {l})<br>I = {obs:.5g}<br>σ(I) = {sigma:.5g}<br>"
            f"standardized residual = {r:.4g}<br>{'held out' if flag else 'work'}"
            for (h, k, l), obs, sigma, r, flag in zip(reflections.hkl, reflections.intensity,
                                                     reflections.sigma, residual, flags)]
    fig = go.Figure(go.Scatter3d(x=q[:, 0], y=q[:, 1], z=q[:, 2], mode="markers",
                                 marker={"size": 3.2, "color": residual, "cmin": -color_limit,
                                         "cmax": color_limit, "colorscale": [[0, BURGUNDY], [.5, "#eee9e2"], [1, OCHRE]],
                                         "colorbar": {"title": "(Iobs−Icalc)/σ"}},
                                 text=text, hovertemplate="%{text}<extra></extra>", name="Reflections"))
    fig.update_layout(**_scene("Reciprocal-space residuals"))
    fig.update_scenes(xaxis_title="qₓ / Å⁻¹", yaxis_title="qᵧ / Å⁻¹", zaxis_title="q_z / Å⁻¹")
    return fig


def _history_xy(history):
    xs, ys = [], []
    for i, row in enumerate(history or []):
        key = next((key for key in ("objective", "chi_squared", "cost", "residual")
                    if key in row and row[key] is not None), None)
        value = None if key is None else row[key]
        if value is not None and np.isfinite(value):
            xs.append(row.get("outer_cycle", row.get("iteration", i)))
            ys.append(float(value) + (float(row.get("restraint_sum_squares", 0)) if key == "chi_squared" else 0))
    return xs, ys


def diagnostics_figure(structure, reflections, scale=1., history=None,
                       twin_laws=None, twin_fractions=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    predicted, residual, flags = _calculated(structure, reflections, scale, twin_laws, twin_fractions)
    reciprocal_resolution = 4 * structure.cell.stol_sq(reflections.hkl)
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Residual versus resolution", "Observed and calculated intensity",
                                                       "Standardized residual distribution", "Recorded optimization objective"),
                        horizontal_spacing=.12, vertical_spacing=.16)
    for mask, name, color, symbol in [(~flags, "work", CHARCOAL, "circle"),
                                     (flags, "held out", BURGUNDY, "circle-open")]:
        if np.any(mask):
            marker = {"size": 4, "color": color, "symbol": symbol, "opacity": .65}
            fig.add_trace(go.Scatter(x=reciprocal_resolution[mask], y=residual[mask], mode="markers",
                                     marker=marker, name=name, legendgroup=name), row=1, col=1)
            fig.add_trace(go.Scatter(x=predicted[mask], y=reflections.intensity[mask], mode="markers",
                                     marker=marker, name=name, legendgroup=name, showlegend=False), row=1, col=2)
    extent = [min(float(predicted.min()), float(reflections.intensity.min())),
              max(float(predicted.max()), float(reflections.intensity.max()))]
    fig.add_trace(go.Scatter(x=extent, y=extent, mode="lines", line={"color": OCHRE, "dash": "dash"},
                             name="Iobs=Icalc", showlegend=False), row=1, col=2)
    fig.add_trace(go.Histogram(x=residual[~flags], histnorm="probability density", nbinsx=36,
                               marker_color=BURGUNDY, opacity=.8, name="work residuals", showlegend=False), row=2, col=1)
    xs, ys = _history_xy(history)
    if ys:
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line_color=OCHRE,
                                 name="objective", showlegend=False), row=2, col=2)
    else:
        fig.add_annotation(text="No objective history supplied", xref="x4 domain", yref="y4 domain",
                           x=.5, y=.5, showarrow=False)
    for row, col, x, y in [(1, 1, "1/d² / Å⁻²", "(Iobs−Icalc)/σ(I)"),
                           (1, 2, "Icalc = k |Fcalc|²", "Iobs (negative values retained)"),
                           (2, 1, "(Iobs−Icalc)/σ(I)", "Probability density"),
                           (2, 2, "Outer-solve checkpoints", "Weighted sum of squares")]:
        fig.update_xaxes(title_text=x, row=row, col=col)
        fig.update_yaxes(title_text=y, row=row, col=col)
    fig.update_layout(template="plotly_white", paper_bgcolor=PAPER, height=760,
                      margin={"l": 65, "r": 35, "t": 70, "b": 55},
                      font={"family": "Arial, sans-serif", "color": CHARCOAL})
    return fig


def create_dashboard(structure: Structure, reflections: Reflections, output_html: Any,
                     scale: float = 1.0, history=None, covariance=None, parameter_names=None,
                     density=None, twin_laws=None, twin_fractions=None,
                     probability: float = .5) -> str:
    """Write a standalone HTML dashboard with one embedded Plotly runtime.

    The page makes no network requests and can be opened directly from disk.
    All user labels are escaped.  The figures show deterministic model outputs;
    no information is sent to an LLM or external visualization service.
    """
    import plotly.graph_objects as go

    figures = [structure_figure(structure, probability, density),
               reciprocal_figure(structure, reflections, scale, twin_laws, twin_fractions),
               diagnostics_figure(structure, reflections, scale, history, twin_laws, twin_fractions)]
    if covariance is not None and np.asarray(covariance).size:
        corr = parameter_correlation(covariance)
        if corr.shape[0] > 250:
            raise ValueError("parameter correlation display exceeds 250 parameters; provide a selected submatrix")
        names = [escape(str(name)) for name in (parameter_names if parameter_names is not None else [f"p{i}" for i in range(len(corr))])]
        if len(names) != len(corr):
            raise ValueError("parameter_names must match covariance dimensions")
        figure = go.Figure(go.Heatmap(z=corr, x=names, y=names, zmin=-1, zmax=1,
                                      colorscale=[[0, BURGUNDY], [.5, "#f8f5ef"], [1, OCHRE]],
                                      colorbar={"title": "Correlation"}))
        figure.update_layout(title="Local parameter correlation", template="plotly_white", paper_bgcolor=PAPER,
                              height=max(500, min(1000, 12 * len(corr))), margin={"l": 120, "b": 120})
        figures.append(figure)
    blocks = []
    config = {"responsive": True, "displaylogo": False,
              "toImageButtonOptions": {"format": "svg", "filename": "ariadne_xtal_figure"}}
    for idx, figure in enumerate(figures):
        blocks.append(figure.to_html(full_html=False, include_plotlyjs=idx == 0,
                                     config=config, div_id=f"ariadne-figure-{idx}"))
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ariadne-Xtal | Crystallographic analysis</title>
<style>body{margin:0;background:#f2eee8;color:#343139;font:16px/1.55 Arial,sans-serif}
main{max-width:1420px;margin:0 auto;padding:28px}header{border-bottom:2px solid #943b4a;margin-bottom:24px}
h1{font-size:32px;margin:6px 0 8px;font-weight:600}p{max-width:1050px}small{color:#645d59}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.card{background:#fcfaf7;border:1px solid #ded6ce;border-radius:10px;overflow:hidden;margin-bottom:20px}
.wide{grid-column:1/-1}.note{padding:0 22px 18px;font-size:14px;color:#645d59}
@media(max-width:960px){.grid{display:block}main{padding:12px}}
</style></head><body><main><header><small>ARIADNE-XTAL / SCIENTIFIC DIAGNOSTICS</small>
<h1>Single-crystal structure and refinement</h1><p>Cartesian atomic displacement ellipsoids, indexed reciprocal-space residuals, and intensity diagnostics. This offline page contains its plotting runtime and data.</p></header><div class="grid">"""
    page += '<section class="card">' + blocks[0] + '<p class="note">Ellipsoids enclose ' + f"{100 * probability:g}%" + ' of a Gaussian atomic displacement distribution. Lines denote geometric periodic contacts; they do not assign bond order. Density contours, if present, are at ±2.5 times the map RMS and carry the actual map units.</p></section>'
    page += '<section class="card">' + blocks[1] + '<p class="note">Reciprocal coordinates omit 2π. The diverging color range is clipped at the larger of 3 and the 98th percentile of |residual|; hover values retain the complete residual.</p></section>'
    page += '<section class="card wide">' + blocks[2] + '<p class="note">Standardized residuals use the supplied observation σ(I), including when the optimization uses other weights. All negative intensities remain visible. For iteratively updated weights, objectives from different outer cycles use their respective fixed weights.</p></section>'
    if len(blocks) > 3:
        page += '<section class="card wide">' + blocks[3] + '<p class="note">Correlation describes the supplied local covariance in its stated parameterization. Blank entries have zero variance and no defined correlation.</p></section>'
    page += "</div></main></body></html>"
    path = Path(output_html)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return str(path)


def static_figures(structure: Structure, reflections: Reflections, output_dir: Any,
                   scale: float = 1., history=None, covariance=None, parameter_names=None,
                   density=None, twin_laws=None, twin_fractions=None,
                   probability: float = .5) -> dict[str, str]:
    """Export publication SVG/PDF figures through Matplotlib, without a browser."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    style = {"font.family": "DejaVu Sans", "font.size": 10, "axes.labelcolor": CHARCOAL,
             "text.color": CHARCOAL, "axes.edgecolor": CHARCOAL, "svg.fonttype": "none",
             "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(style):
        predicted, residual, flags = _calculated(structure, reflections, scale, twin_laws, twin_fractions)
        inv_d2 = 4 * structure.cell.stol_sq(reflections.hkl)
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.3), layout="constrained")
        for mask, name, color in [(~flags, "work", CHARCOAL), (flags, "held out", BURGUNDY)]:
            if np.any(mask):
                axes[0, 0].scatter(inv_d2[mask], residual[mask], s=10, c=color, alpha=.6, label=name, edgecolors="none")
                axes[0, 1].scatter(predicted[mask], reflections.intensity[mask], s=10, c=color, alpha=.6, edgecolors="none")
        axes[0, 0].axhline(0, color=OCHRE, lw=.8)
        axes[0, 0].set(xlabel=r"$1/d^2$ / Å$^{-2}$", ylabel=r"$(I_{obs}-I_{calc})/\sigma(I)$", title="Residual versus resolution")
        axes[0, 0].legend(frameon=False, markerscale=1.5)
        lo, hi = min(predicted.min(), reflections.intensity.min()), max(predicted.max(), reflections.intensity.max())
        axes[0, 1].plot([lo, hi], [lo, hi], "--", color=OCHRE, lw=1)
        axes[0, 1].set(xlabel=r"Calculated intensity $k|F|^2$", ylabel="Observed intensity", title="Observed and calculated intensity")
        if np.any(~flags):
            axes[1, 0].hist(residual[~flags], bins=36, density=True, color=BURGUNDY, alpha=.85, edgecolor="white", linewidth=.4)
        axes[1, 0].set(xlabel=r"$(I_{obs}-I_{calc})/\sigma(I)$", ylabel="Probability density", title="Work residual distribution")
        xs, ys = _history_xy(history)
        if ys:
            axes[1, 1].plot(xs, ys, "o-", color=OCHRE, lw=1.4, ms=4)
        else:
            axes[1, 1].text(.5, .5, "No optimization history supplied", transform=axes[1, 1].transAxes, ha="center")
        axes[1, 1].set(xlabel="Outer-solve checkpoints", ylabel="Weighted sum of squares", title="Optimization objective")
        for label, ax in zip("abcd", axes.ravel()):
            ax.text(-.12, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=13)
        for extension in ("svg", "pdf"):
            path = destination / f"refinement_diagnostics.{extension}"
            fig.savefig(path, bbox_inches="tight", metadata={"Creator": "Ariadne-Xtal"})
            files[f"diagnostics_{extension}"] = str(path)
        plt.close(fig)

        sites = expanded_sites(structure)
        fig = plt.figure(figsize=(8.5, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_proj_type("ortho")
        ax.add_collection3d(Line3DCollection(unit_cell_edges(structure.cell), colors=SILVER, linewidths=.8))
        contacts = geometric_contacts(structure, sites)
        if contacts:
            ax.add_collection3d(Line3DCollection(contacts, colors="#a59d97", linewidths=1.1))
        all_vertices = []
        for site in sites:
            vertices, triangles, _ = ellipsoid_mesh(site["cart"], site["u_cart"], probability, resolution=8)
            ax.add_collection3d(Poly3DCollection(vertices[triangles], facecolors=site["color"],
                                                alpha=.72, edgecolors="none", linewidths=0))
            all_vertices.append(vertices)
        corners = np.concatenate(unit_cell_edges(structure.cell))
        bounds = np.concatenate([corners, *all_vertices]) if all_vertices else corners
        low, high = bounds.min(axis=0), bounds.max(axis=0)
        span = np.maximum(high - low, 1e-6)
        for setter, lo, hi, length in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), low, high, span):
            setter(lo - .05 * length, hi + .05 * length)
        ax.set_box_aspect(span)
        ax.view_init(elev=22, azim=32)
        ax.set(xlabel="x / Å", ylabel="y / Å", zlabel="z / Å",
               title=f"Atomic displacement ellipsoids ({100 * probability:g}% probability)")
        fig.text(.5, .02, "Periodic geometric contacts; full conventional-cell symmetry expansion", ha="center", fontsize=9)
        for extension in ("svg", "pdf"):
            path = destination / f"structure_ellipsoids.{extension}"
            fig.savefig(path, bbox_inches="tight", metadata={"Creator": "Ariadne-Xtal"})
            files[f"structure_{extension}"] = str(path)
        plt.close(fig)

        if density is not None:
            values = density.values
            fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list("ariadne_difference", [BURGUNDY, "#fcfaf7", OCHRE])
            limit = max(float(np.max(np.abs(values))), np.finfo(float).eps)
            nx, ny, nz = values.shape
            u, v = np.meshgrid(np.arange(nx + 1) / nx, np.arange(ny + 1) / ny, indexing="ij")
            for ax, fraction in zip(axes, [.25, .5, .75]):
                idx = min(int(round(fraction * nz)), nz - 1)
                frac = np.stack([u, v, np.full_like(u, idx / nz)], axis=-1)
                cart = density.cell.to_cartesian(frac)
                mesh = ax.pcolormesh(cart[..., 0], cart[..., 1], values[:, :, idx],
                                     cmap=cmap, vmin=-limit, vmax=limit, shading="flat", rasterized=True)
                ax.set(xlabel="x / Å", ylabel="y / Å", title=f"Fractional z = {idx / nz:.3f}", aspect="equal")
            fig.colorbar(mesh, ax=axes, label=density.units, shrink=.8)
            for extension in ("svg", "pdf"):
                path = destination / f"density_sections.{extension}"
                fig.savefig(path, bbox_inches="tight", metadata={"Creator": "Ariadne-Xtal"})
                files[f"density_{extension}"] = str(path)
            plt.close(fig)

        if covariance is not None and np.asarray(covariance).size:
            corr = parameter_correlation(covariance)
            fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list("ariadne_correlation", [BURGUNDY, "#fcfaf7", OCHRE])
            image = ax.imshow(corr, vmin=-1, vmax=1, cmap=cmap)
            if parameter_names is not None and len(corr) <= 40:
                if len(parameter_names) != len(corr):
                    raise ValueError("parameter_names must match covariance dimensions")
                ax.set_xticks(range(len(corr)), parameter_names, rotation=90, fontsize=6)
                ax.set_yticks(range(len(corr)), parameter_names, fontsize=6)
            ax.set_title("Local parameter correlation")
            fig.colorbar(image, ax=ax, label="Correlation")
            for extension in ("svg", "pdf"):
                path = destination / f"parameter_correlation.{extension}"
                fig.savefig(path, bbox_inches="tight", metadata={"Creator": "Ariadne-Xtal"})
                files[f"correlation_{extension}"] = str(path)
            plt.close(fig)
    return files
