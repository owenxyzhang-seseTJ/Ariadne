"""Constrained least-squares refinement of single-crystal intensities.

The native objective is ``sum w_h (I_h - k sum alpha_d |F(T_d h)|²)²``
plus explicit restraint residuals. Negative measured intensities enter intact.
The nonlinear Jacobian uses central three-point differences, and SciPy's
bounded trust-region solver. This module does not implement the full SHELXL
instruction language; its JSON constraints and restraints are defined here.

Coordinates on special positions are restricted to their site tangent space.
General-site anisotropic tensors use a positive Cholesky parameterization;
special-site tensors use a symmetry-invariant matrix logarithm. Exact native
constraints are ``equal_xyz``, ``equal_adp``, ``equal_occupancy`` (``atoms`` is
a list of labels) and ``occupancy_sum`` (``atoms`` and ``target`` in (0, 1]).
Overlapping occupancy-sum and equality groups are rejected.

Reported covariance is conditional on the supplied cell, scattering model,
fixed parameters, origin convention and final frozen weights. It is not a
Bayesian posterior, a model-selection probability or a cell-error estimate.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from numbers import Integral
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.special import softmax

from .model import Reflections, Structure
from .restraints import normalize_restraints, restraint_residuals
from .scattering import calculated_intensities


@dataclass
class RefinementOptions:
    """Explicit choices; all bounds and weights are in the archived result.

    ``twin_laws`` contains non-identity reciprocal column-vector matrices;
    ``twin_fractions`` includes identity first. Only fully overlapping
    (merohedral/pseudomerohedral metric-compatible) integer laws are accepted.
    ``weight_a`` and ``weight_b`` use P=(max(I,0)+2*Icalc)/3 and frozen-weight
    outer iterations. With both zero there is a single sigma-weighted solve.
    ``max_nfev`` limits objective evaluations per outer solve, excluding the
    finite-difference evaluations SciPy performs to form Jacobians.
    ``u_min`` and ``u_max`` bound Uiso and Cholesky diagonals/off-diagonals;
    for special sites they bound independent logarithmic modes. They do not
    impose eigenvalue-wise upper bounds on the final anisotropic tensor.
    """

    max_nfev: int = 300
    ftol: float = 1e-9
    xtol: float = 1e-9
    gtol: float = 1e-8
    scale: float | None = None
    refine_scale: bool = True
    weight_a: float = 0.0
    weight_b: float = 0.0
    weight_cycles: int = 4
    twin_laws: list = field(default_factory=list)
    twin_fractions: list[float] | None = None
    refine_twin_fractions: bool = False
    origin_gauge: str = "auto"
    covariance_rcond: float = 1e-9
    covariance_scale: str = "absolute"
    u_min: float = 1e-6
    u_max: float = 2.0
    verbose: int = 0

    def validate(self) -> None:
        if isinstance(self.max_nfev, bool) or not isinstance(self.max_nfev, Integral) or self.max_nfev < 1:
            raise ValueError("max_nfev must be a positive integer")
        if isinstance(self.weight_cycles, bool) or not isinstance(self.weight_cycles, Integral) or self.weight_cycles < 1:
            raise ValueError("weight_cycles must be a positive integer")
        for name in ("ftol", "xtol", "gtol", "covariance_rcond", "u_min", "u_max"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.covariance_rcond >= 1:
            raise ValueError("covariance_rcond must be less than one")
        if self.u_max <= self.u_min:
            raise ValueError("u_max must exceed u_min")
        if self.scale is not None and (not np.isfinite(self.scale) or self.scale <= 0):
            raise ValueError("Intensity scale must be finite and positive")
        if not self.refine_scale and self.scale is None:
            raise ValueError("A fixed intensity scale must be specified explicitly")
        for name in ("weight_a", "weight_b"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.origin_gauge not in {"auto", "none"}:
            raise ValueError("origin_gauge must be auto or none")
        if self.covariance_scale not in {"absolute", "reduced_chi2"}:
            raise ValueError("covariance_scale must be absolute or reduced_chi2")
        if self.verbose not in {0, 1, 2}:
            raise ValueError("verbose must be 0, 1 or 2")


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


@dataclass
class RefinementResult:
    structure: Structure
    scale: float
    metrics: dict
    history: list[dict]
    covariance: np.ndarray | None
    parameter_names: list[str]
    success: bool
    message: str
    calculated: np.ndarray
    parameters: np.ndarray
    twin_fractions: np.ndarray
    singular_values: np.ndarray
    nullspace: np.ndarray
    standard_uncertainties: dict
    warnings: list[str]
    gauge: dict
    options: dict
    normal_matrix_pseudoinverse: np.ndarray | None = None
    constraints: list[dict] = field(default_factory=list)
    restraints: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return _json_value({
            "schema_version": "ariadne_xtal.refinement/1",
            "structure": self.structure.to_dict(),
            "scale": self.scale,
            "metrics": self.metrics,
            "history": self.history,
            "covariance": self.covariance,
            "parameter_names": self.parameter_names,
            "success": self.success,
            "message": self.message,
            "calculated": self.calculated,
            "parameters": self.parameters,
            "twin_fractions": self.twin_fractions,
            "singular_values": self.singular_values,
            "nullspace": self.nullspace,
            "standard_uncertainties": self.standard_uncertainties,
            "warnings": self.warnings,
            "gauge": self.gauge,
            "options": self.options,
            "normal_matrix_pseudoinverse": self.normal_matrix_pseudoinverse,
            "constraints": self.constraints,
            "restraints": self.restraints,
        })


def _rotation(op) -> np.ndarray:
    return np.asarray(op.rot, dtype=float) / op.DEN


def _site_rotations(structure: Structure, atom) -> list[np.ndarray]:
    answer = []
    for op in structure.operations:
        mapped = np.asarray(op.apply_to_xyz(atom.frac.tolist()))
        delta = mapped - atom.frac
        if np.max(np.abs(delta - np.rint(delta))) <= 1e-8:
            answer.append(_rotation(op))
    return answer


def _kernel(matrices: list[np.ndarray]) -> np.ndarray:
    if not matrices:
        return np.eye(3)
    stacked = np.vstack(matrices)
    # An absolute threshold is needed for matrices that are identically zero.
    _, singular, vt = np.linalg.svd(stacked, full_matrices=True)
    rank = int(np.count_nonzero(singular > 1e-9))
    return vt[rank:].T


def _symmetric_basis() -> list[np.ndarray]:
    answer = []
    for i in range(3):
        matrix = np.zeros((3, 3))
        matrix[i, i] = 1
        answer.append(matrix)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        matrix = np.zeros((3, 3))
        matrix[i, j] = matrix[j, i] = 1 / np.sqrt(2)
        answer.append(matrix)
    return answer


def _invariant_tensor_basis(rotations: list[np.ndarray]) -> np.ndarray:
    base = np.asarray(_symmetric_basis())
    constraints = []
    for rotation in rotations:
        representation = np.array([
            [np.sum(left * (rotation @ right @ rotation.T)) for right in base]
            for left in base
        ])
        constraints.append(representation - np.eye(6))
    matrix = np.vstack(constraints) if constraints else np.zeros((6, 6))
    _, singular, vt = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.count_nonzero(singular > 1e-8))
    return np.einsum("ik,ijl->kjl", vt[rank:].T, base)


def _matrix_function(tensor: np.ndarray, function) -> np.ndarray:
    values, vectors = np.linalg.eigh((tensor + tensor.T) / 2)
    return (vectors * function(values)) @ vectors.T


def _groups(labels: list[str], constraints: list[dict], kind: str) -> list[list[str]]:
    parent = {x: x for x in labels}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for constraint in constraints:
        if constraint["type"] == kind:
            atoms = constraint["atoms"]
            lead = root(atoms[0])
            for atom in atoms[1:]:
                parent[root(atom)] = lead
    out = {}
    for label in labels:
        out.setdefault(root(label), []).append(label)
    return list(out.values())


def _validate_constraints(structure: Structure, values: list[dict] | None) -> list[dict]:
    labels = {atom.label for atom in structure.atoms}
    accepted = {"equal_xyz", "equal_adp", "equal_occupancy", "occupancy_sum"}
    answer = []
    for value in values or []:
        if not isinstance(value, dict) or value.get("type") not in accepted:
            raise ValueError(f"Unsupported exact native constraint: {value}")
        extra = set(value) - {"type", "atoms", "target"}
        if extra:
            raise ValueError(f"Unknown constraint keys: {sorted(extra)}")
        atoms = value.get("atoms", [])
        if not isinstance(atoms, (list, tuple)) or len(atoms) < 2 or len(set(atoms)) != len(atoms):
            raise ValueError("An exact constraint requires at least two distinct atom labels")
        if set(atoms) - labels:
            raise ValueError(f"Unknown constraint atoms: {sorted(set(atoms) - labels)}")
        if value["type"] == "occupancy_sum":
            target = float(value.get("target", 1.0))
            if not np.isfinite(target) or not 0 < target <= 1:
                raise ValueError("Native occupancy_sum target must be in (0,1]")
        elif "target" in value:
            raise ValueError("Only occupancy_sum uses a target field")
        answer.append(copy.deepcopy(value))
    occupancy_groups = [set(x["atoms"]) for x in answer
                        if x["type"] in {"occupancy_sum", "equal_occupancy"}]
    sum_groups = [set(x["atoms"]) for x in answer if x["type"] == "occupancy_sum"]
    for group in sum_groups:
        overlaps = sum(bool(group & other) for other in occupancy_groups)
        if overlaps > 1:
            raise ValueError("occupancy_sum groups cannot overlap any other occupancy constraint")
    return answer


def _validate_twins(structure: Structure, options: RefinementOptions) -> tuple[list[np.ndarray], np.ndarray]:
    laws = []
    metric = structure.cell.reciprocal.T @ structure.cell.reciprocal
    seen = {tuple(np.eye(3, dtype=int).ravel())}
    for value in options.twin_laws:
        law = np.asarray(value, dtype=float)
        if law.shape != (3, 3) or not np.all(np.isfinite(law)) or not np.allclose(law, np.rint(law)):
            raise ValueError("A native merohedral twin law must be a finite integer 3x3 matrix")
        law = np.rint(law).astype(int)
        if not np.isclose(abs(np.linalg.det(law)), 1):
            raise ValueError("A native twin law must be unimodular")
        if not np.allclose(law.T @ metric @ law, metric, atol=1e-9, rtol=1e-5):
            raise ValueError("Native twinning requires fully overlapping, metric-compatible reciprocal lattices")
        key = tuple(law.ravel())
        if key in seen:
            raise ValueError("Twin laws must be distinct and must exclude the implicit identity")
        seen.add(key)
        laws.append(law)
    n = len(laws) + 1
    fractions = np.full(n, 1 / n) if options.twin_fractions is None else np.asarray(options.twin_fractions, dtype=float)
    if fractions.shape != (n,) or not np.all(np.isfinite(fractions)) or np.any(fractions < 0) or not np.isclose(fractions.sum(), 1, atol=1e-8):
        raise ValueError("Twin fractions must include identity, be nonnegative and sum to one")
    if options.refine_twin_fractions and not laws:
        raise ValueError("Refining twin fractions requires at least one nonidentity law")
    if options.refine_twin_fractions and np.any(fractions <= 0):
        raise ValueError("Refined softmax twin fractions must start strictly above zero")
    return laws, fractions


def _reflection_keys(structure: Structure, hkl: np.ndarray, twin_laws=None) -> list[tuple[int, int, int]]:
    generators = [np.rint(_rotation(op).T).astype(int) for op in structure.operations]
    generators += [np.asarray(x, dtype=int) for x in (twin_laws or [])]
    generators += [-np.eye(3, dtype=int)]  # keep Friedel pairs together, also for anomalous data
    identity = np.eye(3, dtype=int)
    group = {tuple(identity.ravel()): identity}
    frontier = [identity]
    while frontier:
        current = frontier.pop()
        for generator in generators:
            candidate = generator @ current
            key = tuple(candidate.ravel())
            if key not in group:
                if len(group) >= 192:
                    raise ValueError("Reflection grouping operators do not form a finite crystallographic group")
                group[key] = candidate
                frontier.append(candidate)
    transformed = np.stack([np.asarray(hkl) @ matrix.T for matrix in group.values()], axis=1)
    return [min(tuple(int(v) for v in mate) for mate in orbit) for orbit in transformed]


def grouped_holdout_flags(structure: Structure, reflections: Reflections, fraction: float = 0.1,
                          seed: int = 0, twin_laws=None) -> np.ndarray:
    """A reproducible orbit-wise holdout split, including Friedel and twin mates.

    Use one predeclared seed for model selection. Repeatedly inspecting these
    holdout statistics consumes that holdout; a final external set is separate.
    """
    if not 0 < fraction < 1:
        raise ValueError("Holdout fraction must lie in (0,1)")
    keys = _reflection_keys(structure, reflections.hkl, twin_laws)
    unique = sorted(set(keys))
    if len(unique) < 2:
        raise ValueError("At least two independent reflection orbits are required")
    rng = np.random.default_rng(seed)
    count = min(len(unique) - 1, max(1, int(round(len(unique) * fraction))))
    selected = {unique[i] for i in rng.choice(len(unique), count, replace=False)}
    return np.array([key in selected for key in keys], dtype=bool)


def _validate_holdouts(structure: Structure, reflections: Reflections, laws) -> np.ndarray:
    flags = np.zeros(len(reflections), dtype=bool) if reflections.free_flags is None else np.asarray(reflections.free_flags, dtype=bool)
    if np.all(flags):
        raise ValueError("At least one work reflection is required")
    if np.any(flags):
        assigned = {}
        for key, flag in zip(_reflection_keys(structure, reflections.hkl, laws), flags):
            if key in assigned and assigned[key] != bool(flag):
                raise ValueError("Holdout leakage: equivalent/Friedel/twin-related reflections have different flags; use grouped_holdout_flags")
            assigned[key] = bool(flag)
    return flags


class _Parameterization:
    def __init__(self, structure: Structure, options: RefinementOptions, scale: float,
                 fractions: np.ndarray, constraints: list[dict]):
        self.base = copy.deepcopy(structure)
        self.options = options
        self.scale = float(scale)
        self.fractions = fractions.copy()
        self.names: list[str] = []
        self.start: list[float] = []
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.blocks: list[tuple] = []
        self.warnings: list[str] = []
        self.gauge: dict = {"dimension": 0, "anchor": None, "directions_fractional": []}
        self.index = {atom.label: i for i, atom in enumerate(self.base.atoms)}
        self.labels = list(self.index)
        self.constraints = constraints
        self._coordinates()
        self._adps()
        self._occupancies()
        if options.refine_scale:
            idx = self._add([np.log(scale)], ["log_intensity_scale"], [-30], [30])
            self.blocks.append(("scale", idx))
        if options.refine_twin_fractions:
            idx = self._add(np.log(fractions[1:] / fractions[0]),
                            [f"twin_logit[{i}]" for i in range(1, len(fractions))],
                            [-30] * (len(fractions) - 1), [30] * (len(fractions) - 1))
            self.blocks.append(("twins", idx))

    def _add(self, values, names, lower, upper) -> slice:
        start = len(self.start)
        values = list(values)
        if np.any(np.asarray(values) < lower) or np.any(np.asarray(values) > upper):
            raise ValueError(f"Starting parameter is outside its configured bounds: {names}")
        self.start.extend(float(x) for x in values)
        self.names.extend(names)
        self.lower.extend(float(x) for x in lower)
        self.upper.extend(float(x) for x in upper)
        return slice(start, len(self.start))

    def _coordinates(self):
        polar = _kernel([_rotation(op) - np.eye(3) for op in self.base.operations])
        groups = _groups(self.labels, self.constraints, "equal_xyz")
        free_origin = all(atom.refine_xyz for atom in self.base.atoms)
        anchor_used = False
        for labels in groups:
            atoms = [self.base.atoms[self.index[x]] for x in labels]
            anchor = atoms[0].frac.copy()
            offsets = []
            for atom in atoms:
                delta = atom.frac - anchor
                if np.max(np.abs(delta - np.rint(delta))) > 1e-8:
                    raise ValueError("equal_xyz requires initially coincident positions modulo lattice translations")
                offsets.append(np.rint(delta))
            site = [rotation - np.eye(3) for atom in atoms for rotation in _site_rotations(self.base, atom)]
            tangent = _kernel(site)
            if not all(atom.refine_xyz for atom in atoms):
                tangent = np.zeros((3, 0))
            elif free_origin and polar.shape[1] and not anchor_used and self.options.origin_gauge == "auto":
                # Polar translations lie in every site's tangent. Pin just their
                # projection on one common coordinate group.
                tangent = _kernel(site + [polar.T])
                anchor_used = True
                self.gauge = {"dimension": int(polar.shape[1]), "anchor": labels[0],
                              "directions_fractional": polar.T.tolist()}
                self.warnings.append("Continuous origin freedom fixed at the anchor; eliminated directions have no measured standard uncertainty")
            idx = self._add(np.zeros(tangent.shape[1]),
                            [f"{labels[0]}.position_mode[{j}]" for j in range(tangent.shape[1])],
                            [-np.inf] * tangent.shape[1], [np.inf] * tangent.shape[1])
            self.blocks.append(("xyz", idx, labels, anchor, offsets, tangent))

    def _adps(self):
        matrix = self.base.cell.matrix
        inverse = np.linalg.inv(matrix)
        for labels in _groups(self.labels, self.constraints, "equal_adp"):
            atoms = [self.base.atoms[self.index[x]] for x in labels]
            anisotropic = atoms[0].u_aniso is not None
            if any((atom.u_aniso is not None) != anisotropic for atom in atoms):
                raise ValueError("equal_adp cannot mix isotropic and anisotropic atoms")
            if not anisotropic:
                value = float(atoms[0].u_iso)
                if any(abs(atom.u_iso - value) > 1e-8 for atom in atoms):
                    raise ValueError("equal_adp requires equal initial Uiso values")
                if all(atom.refine_u for atom in atoms):
                    if value < self.options.u_min:
                        value = self.options.u_min
                        self.warnings.append(f"{labels[0]} initial Uiso raised to positive parameterization floor")
                    idx = self._add([np.log(value)], [f"{labels[0]}.log_Uiso"],
                                    [np.log(self.options.u_min)], [np.log(self.options.u_max)])
                    self.blocks.append(("uiso", idx, labels))
                else:
                    for atom in atoms:
                        atom.u_iso = value
                continue
            tensor = np.asarray(atoms[0].u_aniso).copy()
            if any(not np.allclose(atom.u_aniso, tensor, atol=1e-8, rtol=0) for atom in atoms):
                raise ValueError("equal_adp requires equal initial Cartesian tensors")
            rotations = [matrix @ rotation @ inverse for atom in atoms
                         for rotation in _site_rotations(self.base, atom)]
            basis = _invariant_tensor_basis(rotations)
            projected = np.einsum("i,ijk->jk", np.einsum("ijk,jk->i", basis, tensor), basis)
            if not np.allclose(tensor, projected, atol=1e-7, rtol=0):
                self.warnings.append(f"{labels[0]} initial anisotropic tensor projected onto site symmetry")
            tensor = projected
            if not all(atom.refine_u for atom in atoms):
                for atom in atoms:
                    atom.u_aniso = tensor.copy()
                continue
            eig = np.linalg.eigvalsh(tensor)
            if eig.min() < self.options.u_min:
                tensor = _matrix_function(tensor, lambda x: np.maximum(x, self.options.u_min))
                self.warnings.append(f"{labels[0]} initial ADP eigenvalues raised to positive parameterization floor")
            if basis.shape[0] == 6:
                chol = np.linalg.cholesky(tensor)
                values = [np.log(chol[0, 0]), np.log(chol[1, 1]), np.log(chol[2, 2]),
                          chol[1, 0], chol[2, 0], chol[2, 1]]
                names = [f"{labels[0]}.{x}" for x in
                         ("log_L11", "log_L22", "log_L33", "L21", "L31", "L32")]
                lower = [0.5 * np.log(self.options.u_min)] * 3 + [-np.sqrt(self.options.u_max)] * 3
                upper = [0.5 * np.log(self.options.u_max)] * 3 + [np.sqrt(self.options.u_max)] * 3
                idx = self._add(values, names, lower, upper)
                self.blocks.append(("chol", idx, labels))
            else:
                log_tensor = _matrix_function(tensor, np.log)
                values = np.einsum("ijk,jk->i", basis, log_tensor)
                # Bounds apply to independent log modes, not individual Uij.
                bound = np.sqrt(3) * max(abs(np.log(self.options.u_min)), abs(np.log(self.options.u_max)))
                idx = self._add(values, [f"{labels[0]}.log_U_mode[{i}]" for i in range(len(values))],
                                [-bound] * len(values), [bound] * len(values))
                self.blocks.append(("log_tensor", idx, labels, basis))

    def _occupancies(self):
        in_sum = set()
        scale_anchor = False
        for constraint in self.constraints:
            if constraint["type"] != "occupancy_sum":
                continue
            labels = constraint["atoms"]
            atoms = [self.base.atoms[self.index[x]] for x in labels]
            target = float(constraint.get("target", 1))
            values = np.array([atom.occupancy for atom in atoms])
            if not np.isclose(values.sum(), target, atol=1e-8):
                raise ValueError("occupancy_sum must be satisfied by the starting model")
            if not all(atom.refine_occupancy for atom in atoms):
                raise ValueError("Every occupancy_sum member must have refine_occupancy=True")
            if np.any(values <= 0):
                raise ValueError("Softmax occupancy groups require strictly positive initial members")
            idx = self._add(np.log(values[1:] / values[0]),
                            [f"occupancy_logit[{labels[0]},{label}]" for label in labels[1:]],
                            [-30] * (len(values) - 1), [30] * (len(values) - 1))
            self.blocks.append(("occupancy_sum", idx, labels, target))
            in_sum.update(labels)
            scale_anchor = True
        for labels in _groups(self.labels, self.constraints, "equal_occupancy"):
            if labels[0] in in_sum:
                continue
            atoms = [self.base.atoms[self.index[x]] for x in labels]
            value = float(atoms[0].occupancy)
            if any(abs(atom.occupancy - value) > 1e-8 for atom in atoms):
                raise ValueError("equal_occupancy requires equal initial occupancies")
            if all(atom.refine_occupancy for atom in atoms):
                idx = self._add([value], [f"{labels[0]}.occupancy"], [0], [1])
                self.blocks.append(("occupancy", idx, labels))
            else:
                scale_anchor |= value > 0
                for atom in atoms:
                    atom.occupancy = value
        if self.options.refine_scale and not scale_anchor:
            raise ValueError("All occupancies and scale are free: fix a nonzero occupancy or specify an exact occupancy_sum")

    def unpack(self, values: np.ndarray) -> tuple[Structure, float, np.ndarray]:
        result = copy.deepcopy(self.base)
        scale = self.scale
        fractions = self.fractions.copy()
        for block in self.blocks:
            kind, idx = block[:2]
            data = values[idx]
            if kind == "scale":
                scale = float(np.exp(data[0]))
            elif kind == "twins":
                fractions = softmax(np.r_[0.0, data])
            elif kind == "xyz":
                _, _, labels, anchor, offsets, tangent = block
                position = anchor + tangent @ data
                for label, offset in zip(labels, offsets):
                    result.atoms[self.index[label]].frac = position + offset
            elif kind in {"uiso", "chol", "log_tensor"}:
                labels = block[2]
                if kind == "uiso":
                    value = float(np.exp(data[0]))
                    for label in labels:
                        result.atoms[self.index[label]].u_iso = value
                else:
                    if kind == "chol":
                        chol = np.diag(np.exp(data[:3]))
                        chol[1, 0], chol[2, 0], chol[2, 1] = data[3:]
                        tensor = chol @ chol.T
                    else:
                        log_tensor = np.einsum("i,ijk->jk", data, block[3])
                        tensor = _matrix_function(log_tensor, np.exp)
                    for label in labels:
                        atom = result.atoms[self.index[label]]
                        atom.u_aniso = tensor.copy()
                        atom.u_iso = float(np.trace(tensor) / 3)
            elif kind in {"occupancy", "occupancy_sum"}:
                labels = block[2]
                occupancies = ([float(data[0])] * len(labels) if kind == "occupancy" else
                               block[3] * softmax(np.r_[0.0, data]))
                for label, value in zip(labels, occupancies):
                    result.atoms[self.index[label]].occupancy = float(value)
        return result, scale, fractions

    def physical_values(self, values: np.ndarray) -> tuple[list[str], np.ndarray]:
        model, scale, fractions = self.unpack(values)
        names = ["intensity_scale"]
        output = [scale]
        for atom in model.atoms:
            names += [f"{atom.label}.{axis}" for axis in ("x", "y", "z")]
            output += atom.frac.tolist()
            names.append(f"{atom.label}.occupancy")
            output.append(atom.occupancy)
            if atom.u_aniso is None:
                names.append(f"{atom.label}.Uiso")
                output.append(atom.u_iso)
            else:
                names += [f"{atom.label}.{axis}" for axis in ("U11_cart", "U22_cart", "U33_cart", "U12_cart", "U13_cart", "U23_cart")]
                tensor = atom.u_aniso
                output += [tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[0, 2], tensor[1, 2]]
        names += [f"twin_fraction[{i}]" for i in range(len(fractions))]
        output += fractions.tolist()
        return names, np.asarray(output)


def _weights(observed: np.ndarray, sigma: np.ndarray, calculated: np.ndarray,
             options: RefinementOptions) -> np.ndarray:
    p = (np.maximum(observed, 0) + 2 * calculated) / 3
    denominator = sigma**2 + (options.weight_a * p)**2 + options.weight_b * p
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0):
        raise ValueError("Nonfinite or nonpositive refinement weights")
    return 1 / denominator


def _r_metrics(observed, sigma, calculated, weights) -> dict:
    if len(observed) == 0:
        return {"n": 0, "r1_all": None, "r1_gt": None, "wr2": None,
                "chi_squared": None, "sigma_rmse": None, "negative_intensities": 0}
    fo = np.sqrt(np.maximum(observed, 0))
    fc = np.sqrt(np.maximum(calculated, 0))
    strong = observed > 2 * sigma

    def r1(mask):
        denominator = np.sum(fo[mask])
        return float(np.sum(np.abs(fo[mask] - fc[mask])) / denominator) if denominator > 0 else None

    numerator = np.sum(weights * (observed - calculated)**2)
    denominator = np.sum(weights * observed**2)
    return {"n": len(observed), "n_gt": int(np.count_nonzero(strong)),
            "r1_all": r1(np.ones(len(observed), dtype=bool)), "r1_gt": r1(strong),
            "wr2": float(np.sqrt(numerator / denominator)) if denominator > 0 else None,
            "chi_squared": float(numerator),
            "sigma_rmse": float(np.sqrt(np.mean(((observed - calculated) / sigma)**2))),
            "negative_intensities": int(np.count_nonzero(observed < 0))}


def _physical_uncertainties(parameterization, values, covariance, active_mask) -> dict:
    names, physical = parameterization.physical_values(values)
    if covariance is None:
        return dict.fromkeys(names)
    gradient = np.empty((len(physical), len(values)))
    step = np.cbrt(np.finfo(float).eps)
    for j in range(len(values)):
        delta = step * max(1.0, abs(values[j]))
        plus, minus = values.copy(), values.copy()
        plus[j] += delta
        minus[j] -= delta
        gradient[:, j] = (parameterization.physical_values(plus)[1] -
                          parameterization.physical_values(minus)[1]) / (2 * delta)
    variance = np.einsum("ij,jk,ik->i", gradient, covariance, gradient)
    answer = {}
    for i, name in enumerate(names):
        fixed = np.linalg.norm(gradient[i]) < 1e-12
        boundary = np.any(np.abs(gradient[i, active_mask != 0]) > 1e-10)
        answer[name] = None if fixed or boundary else float(np.sqrt(max(variance[i], 0)))
    return answer


def refine(structure: Structure, reflections: Reflections, options: RefinementOptions | dict | None = None,
           restraints=None, constraints=None) -> RefinementResult:
    """Refine a model without mutating inputs or using held-out reflections.

    ``success`` means numerical convergence. An additional
    ``metrics['identifiable']`` flag reports full Jacobian rank; neither is a
    claim that the structure has been solved correctly. A rank-deficient
    result exposes its nullspace and normal-matrix pseudoinverse but sets
    covariance and all standard uncertainties to None.
    """
    options = RefinementOptions() if options is None else (
        RefinementOptions(**options) if isinstance(options, dict) else copy.deepcopy(options))
    if not isinstance(options, RefinementOptions):
        raise ValueError("options must be RefinementOptions or a dictionary")
    options.validate()
    constraints = _validate_constraints(structure, constraints)
    restraints = normalize_restraints(restraints)
    laws, fractions = _validate_twins(structure, options)
    flags = _validate_holdouts(structure, reflections, laws)
    work = ~flags
    obs = reflections.intensity[work]
    sigma = reflections.sigma[work]
    hkl = reflections.hkl[work]
    initial = calculated_intensities(structure, hkl, twin_laws=laws, twin_fractions=fractions)
    if options.scale is None:
        denominator = np.sum((initial / sigma)**2)
        if denominator <= 0:
            raise ValueError("The initial model has no calculated intensity on the work reflections")
        scale = max(1e-12, float(np.sum(obs * initial / sigma**2) / denominator))
    else:
        scale = float(options.scale)
    parameterization = _Parameterization(structure, options, scale, fractions, constraints)
    values = np.asarray(parameterization.start)
    lower, upper = np.asarray(parameterization.lower), np.asarray(parameterization.upper)
    warnings = parameterization.warnings.copy()
    npar = len(values)
    nobs = int(work.sum())
    if nobs <= npar:
        warnings.append("Work reflection count does not exceed parameter count; diffraction-only goodness of fit is undefined")
    history = []
    evaluation_count = 0
    last_result = None
    cycles = options.weight_cycles if options.weight_a or options.weight_b else 1

    def predict(parameters, selected_hkl):
        model, current_scale, current_fractions = parameterization.unpack(parameters)
        calculated = calculated_intensities(model, selected_hkl, scale=current_scale,
                                           twin_laws=laws, twin_fractions=current_fractions)
        return model, calculated

    initial_model, initial_calc = predict(values, hkl)
    # Validate restraint labels and geometry before handing control to SciPy.
    restraint_residuals(initial_model, restraints)
    first_weights = _weights(obs, sigma, initial_calc, options)
    history.append({"stage": "initial", "outer_cycle": 0,
                    **_r_metrics(obs, sigma, initial_calc, first_weights),
                    "restraint_sum_squares": float(np.sum(restraint_residuals(initial_model, restraints)**2))})
    frozen = first_weights
    for outer in range(cycles):
        _, at_start = predict(values, hkl)
        frozen = _weights(obs, sigma, at_start, options)
        sqrt_weights = np.sqrt(frozen)

        def residual(parameters):
            nonlocal evaluation_count
            evaluation_count += 1
            model, calculated = predict(parameters, hkl)
            data = (calculated - obs) * sqrt_weights
            return np.r_[data, restraint_residuals(model, restraints)]

        if npar:
            last_result = least_squares(residual, values, jac="3-point", bounds=(lower, upper),
                                        method="trf", loss="linear", x_scale="jac",
                                        max_nfev=options.max_nfev, ftol=options.ftol,
                                        xtol=options.xtol, gtol=options.gtol, verbose=options.verbose)
            previous = values.copy()
            values = last_result.x
        model, current_calc = predict(values, hkl)
        restraint_vector = restraint_residuals(model, restraints)
        history.append({"stage": "refined", "outer_cycle": outer + 1,
                        **_r_metrics(obs, sigma, current_calc, frozen),
                        "restraint_sum_squares": float(np.sum(restraint_vector**2)),
                        "function_evaluations": int(last_result.nfev) if last_result is not None else 0})
        if npar and not last_result.success:
            break
        if npar and outer > 0 and np.max(np.abs(values - previous)) < 1e-8:
            break
    final_model, final_scale, final_fractions = parameterization.unpack(values)
    all_calculated = calculated_intensities(final_model, reflections.hkl, scale=final_scale,
                                            twin_laws=laws, twin_fractions=final_fractions)
    all_weights = _weights(reflections.intensity, reflections.sigma, all_calculated, options)
    # Metrics describe the weights actually optimized for work observations.
    all_weights[work] = frozen
    metrics = _r_metrics(obs, sigma, all_calculated[work], frozen)
    free_metrics = _r_metrics(reflections.intensity[flags], reflections.sigma[flags],
                              all_calculated[flags], all_weights[flags])
    restraint_vector = restraint_residuals(final_model, restraints)
    active = np.zeros(npar, dtype=int) if last_result is None else last_result.active_mask
    if npar:
        jacobian = np.asarray(last_result.jac)
        _, singular, vt = np.linalg.svd(jacobian, full_matrices=True)
        threshold = options.covariance_rcond * singular[0] if len(singular) else 0
        rank = int(np.count_nonzero(singular > threshold))
        estimable = vt[:rank].T
        pseudo = (estimable / singular[:rank]**2) @ estimable.T if rank else np.zeros((npar, npar))
        nullspace = vt[rank:]
        covariance = pseudo.copy() if rank == npar else None
        work_singular = np.linalg.svd(jacobian[:nobs], compute_uv=False)
        work_threshold = options.covariance_rcond * work_singular[0] if len(work_singular) else 0
        work_rank = int(np.count_nonzero(work_singular > work_threshold))
        condition = float(singular[0] / singular[-1]) if rank == npar else None
    else:
        singular = np.empty(0)
        nullspace = np.empty((0, 0))
        rank = work_rank = 0
        covariance = pseudo = np.empty((0, 0))
        condition = None
    if rank < npar:
        warnings.append(f"Jacobian rank {rank}/{npar}: non-identifiable parameter combinations; no covariance or standard uncertainties are reported")
    if work_rank < rank:
        warnings.append("Some parameter combinations are identified by restraints rather than diffraction alone")
    if np.any(active):
        warnings.append("Some variables are on optimization bounds; dependent physical standard uncertainties are omitted")
    dof = nobs - npar
    gof = float(np.sqrt(metrics["chi_squared"] / dof)) if dof > 0 else None
    if covariance is not None and options.covariance_scale == "reduced_chi2":
        if dof > 0:
            covariance *= metrics["chi_squared"] / dof
        else:
            covariance = None
            warnings.append("Reduced-chi-square covariance scaling requires positive diffraction degrees of freedom")
    metrics.update({"n_work": nobs, "n_free": int(flags.sum()), "n_parameters": npar,
                    "jacobian_rank": rank, "diffraction_jacobian_rank": work_rank,
                    "identifiable": rank == npar, "jacobian_condition": condition,
                    "normal_condition": None if condition is None else condition**2,
                    "degrees_of_freedom": dof, "gof": gof,
                    "restraint_residual_count": len(restraint_vector),
                    "restraint_sum_squares": float(np.dot(restraint_vector, restraint_vector)),
                    "free": free_metrics, "objective_evaluations_total": evaluation_count,
                    "weighting": "sigma" if not (options.weight_a or options.weight_b) else "frozen_weight_IRLS",
                    "covariance_scaling": options.covariance_scale})
    if options.weight_a or options.weight_b:
        new_weights = _weights(obs, sigma, all_calculated[work], options)
        metrics["final_weight_relative_change"] = float(np.max(np.abs(new_weights / frozen - 1)))
        if metrics["final_weight_relative_change"] > 1e-3:
            warnings.append("IRLS weights have not reached a fixed point; increase weight_cycles if these weights are intended")
    uncertainties = _physical_uncertainties(parameterization, values, covariance, active)
    numerical_success = True if last_result is None else bool(last_result.success)
    message = "All parameters fixed; residuals evaluated" if last_result is None else str(last_result.message)
    return RefinementResult(final_model, final_scale, metrics, history, covariance,
                            parameterization.names, numerical_success, message,
                            all_calculated, values, final_fractions, singular, nullspace,
                            uncertainties, warnings, parameterization.gauge, _json_value(asdict(options)), pseudo,
                            constraints=copy.deepcopy(constraints), restraints=[item.to_dict() for item in restraints])
