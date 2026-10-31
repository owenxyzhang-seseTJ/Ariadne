"""Finite crystallographic symmetry and explicitly diagnostic group scoring."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gemmi
import numpy as np

if TYPE_CHECKING:
    from .model import Atom, Cell, Reflections, Structure


SITE_TOLERANCE = 1e-8


def operation_parts(op: gemmi.Op) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(op.rot, dtype=float) / op.DEN, np.asarray(op.tran, dtype=float) / op.DEN


def _op_key(op: gemmi.Op) -> tuple[int, ...]:
    rotation = np.asarray(op.rot, dtype=np.int64)
    if np.any(rotation % op.DEN):
        raise ValueError("symmetry rotation must map integer lattice translations to integers")
    return tuple((rotation // op.DEN).ravel()) + tuple(np.asarray(op.tran, dtype=np.int64) % op.DEN)


def get_operations(spacegroup: str, sym_ops: list[str] | None = None,
                   cell: "Cell | None" = None) -> list[gemmi.Op]:
    """Obtain a full symmetry group; explicit operations have authority.

    Explicit lists must already contain the whole finite group (including
    centering).  A list of generators is rejected instead of being guessed.
    Translations are compared modulo the unit lattice.  Cell/group mismatch
    is a scientific error, not a reason to force an orthogonal metric.
    """
    if sym_ops is None:
        if cell is None:
            group = gemmi.find_spacegroup_by_name(spacegroup)
        else:
            group = gemmi.find_spacegroup_by_name(spacegroup, cell.alpha, cell.gamma)
        if group is None:
            raise ValueError(f"unknown space-group setting {spacegroup!r}")
        ops = list(group.operations())
    else:
        if not sym_ops:
            raise ValueError("explicit symmetry operation list must not be empty")
        ops = [gemmi.Op(s) for s in sym_ops]
        keys = [_op_key(op) for op in ops]
        keyset = set(keys)
        if len(keyset) != len(keys):
            raise ValueError("duplicate explicit symmetry operations")
        identity = tuple(np.eye(3, dtype=int).ravel()) + (0, 0, 0)
        if identity not in keyset:
            raise ValueError("explicit symmetry group lacks the identity operation")
        rotations = [np.array(k[:9], dtype=np.int64).reshape(3, 3) for k in keys]
        translations = [np.array(k[9:], dtype=np.int64) for k in keys]
        for r in rotations:
            if not np.isclose(abs(np.linalg.det(r)), 1.0, atol=1e-10):
                raise ValueError("symmetry rotations must be unimodular")
        den = gemmi.Op.DEN
        for r1, t1 in zip(rotations, translations):
            for r2, t2 in zip(rotations, translations):
                key = tuple((r1 @ r2).ravel()) + tuple((r1 @ t2 + t1) % den)
                if key not in keyset:
                    raise ValueError("explicit symmetry operations are not closed under composition")
    if cell is not None:
        metric = cell.metric
        for op in ops:
            rotation, _ = operation_parts(op)
            if not np.allclose(rotation.T @ metric @ rotation, metric, rtol=2e-6,
                               atol=2e-6 * max(np.max(np.abs(metric)), 1.0)):
                raise ValueError(f"cell metric is incompatible with symmetry operation {op.triplet()}")
    return ops


def site_stabilizer(structure: "Structure", atom: "Atom") -> list[gemmi.Op]:
    result = []
    for op in structure.operations:
        rotation, translation = operation_parts(op)
        delta = rotation @ atom.frac + translation - atom.frac
        if np.max(np.abs(delta - np.rint(delta))) <= SITE_TOLERANCE:
            result.append(op)
    return result


def unique_sites(structure: "Structure", atom: "Atom") -> list[tuple[np.ndarray, np.ndarray]]:
    """Unique images and projected Cartesian U tensors, without multiplicity.

    If several operations reach the same position, their rotated tensors are
    averaged.  This is the stabilizer projector at a special position.  The
    physical occupancy is applied once for each unique image by the caller.
    """
    a = structure.cell.matrix
    inverse = np.linalg.inv(a)
    u = np.eye(3) * atom.u_iso if atom.u_aniso is None else np.asarray(atom.u_aniso)
    positions: list[np.ndarray] = []
    tensors: list[list[np.ndarray]] = []
    for op in structure.operations:
        rotation, translation = operation_parts(op)
        position = (rotation @ atom.frac + translation) % 1.0
        rotation_cart = a @ rotation @ inverse
        transformed_u = rotation_cart @ u @ rotation_cart.T
        for idx, existing in enumerate(positions):
            delta = position - existing
            if np.max(np.abs(delta - np.rint(delta))) <= SITE_TOLERANCE:
                tensors[idx].append(transformed_u)
                break
        else:
            positions.append(position)
            tensors.append([transformed_u])
    return [(position, np.mean(values, axis=0)) for position, values in zip(positions, tensors)]


def systematic_absences(structure_or_spacegroup: "Structure | str | gemmi.SpaceGroup",
                        hkl: Any) -> np.ndarray:
    """Little-group translation-character test for integral Miller indices."""
    raw = np.asarray(hkl, dtype=float)
    one = raw.ndim == 1
    indices = np.atleast_2d(raw)
    if indices.shape[1] != 3 or not np.isfinite(indices).all() or not np.equal(indices, np.rint(indices)).all():
        raise ValueError("systematic-absence indices must be an N x 3 integer array")
    indices = indices.astype(np.int64)
    if isinstance(structure_or_spacegroup, str):
        ops = get_operations(structure_or_spacegroup)
    elif isinstance(structure_or_spacegroup, gemmi.SpaceGroup):
        ops = list(structure_or_spacegroup.operations())
    else:
        ops = structure_or_spacegroup.operations
    character = np.zeros(len(indices), dtype=complex)
    count = np.zeros(len(indices), dtype=int)
    for op in ops:
        rotation, translation = operation_parts(op)
        little = np.all(np.isclose(indices @ rotation, indices, atol=1e-9, rtol=0), axis=1)
        character[little] += np.exp(2j * np.pi * (indices[little] @ translation))
        count[little] += 1
    absent = np.abs(character) < 1e-7 * np.maximum(count, 1)
    return absent[0] if one else absent


def _equivalent_statistics(reflections: "Reflections", ops: list[gemmi.Op]) -> tuple[float, int]:
    # Friedel inversion is not added: it is invalid with appreciable anomalous scattering.
    rotations = [np.rint(operation_parts(op)[0]).astype(int) for op in ops]
    groups: dict[tuple[int, ...], list[int]] = {}
    for i, h in enumerate(reflections.hkl):
        key = min(tuple(h @ r) for r in rotations)
        groups.setdefault(key, []).append(i)
    chi2, dof = 0.0, 0
    for ids in groups.values():
        if len(ids) > 1:
            intensities = reflections.intensity[ids]
            weights = 1 / reflections.sigma[ids] ** 2
            mean = np.dot(weights, intensities) / weights.sum()
            chi2 += float(np.sum(weights * (intensities - mean) ** 2))
            dof += len(ids) - 1
    return chi2, dof


def rank_spacegroups(cell: "Cell", reflections: "Reflections", candidates: list[str]) -> list[dict[str, Any]]:
    """Rank *contradictions* to proposed groups, never infer a unique group.

    Each candidate is tested for metric compatibility, zero-mean standardized
    intensities in its forbidden classes, and intensity consistency among
    measured crystallographic equivalents.  The score is a diagnostic
    discrepancy, not a posterior probability or a complexity-comparable
    likelihood.  Groups without discriminating measured classes remain
    untested.  Phase-based and chemical comparisons belong to the solver.
    """
    results = []
    for name in candidates:
        try:
            ops = get_operations(name, cell=cell)
        except (ValueError, RuntimeError) as error:
            results.append({"spacegroup": name, "compatible_metric": False,
                            "score": None, "diagnostic": str(error), "identification": "not_determined"})
            continue
        group = gemmi.find_spacegroup_by_name(name, cell.alpha, cell.gamma)
        absent = systematic_absences(group, reflections.hkl)
        z = reflections.intensity[absent] / reflections.sigma[absent]
        violations = int(np.count_nonzero(np.abs(z) > 3))
        absent_excess = float(np.mean(np.maximum(np.abs(z) - 3, 0) ** 2)) if len(z) else 0.0
        equiv_chi2, equiv_dof = _equivalent_statistics(reflections, ops)
        equiv_reduced = equiv_chi2 / equiv_dof if equiv_dof else None
        score = absent_excess + max((equiv_reduced or 0) - 1, 0)
        results.append({
            "spacegroup": name, "compatible_metric": True, "score": score,
            "n_operations": len(ops), "n_tested_absences": int(len(z)),
            "n_absence_violations_3sigma": violations,
            "absence_chi2": float(np.sum(z * z)) if len(z) else None,
            "equivalent_chi2": equiv_chi2, "equivalent_degrees_of_freedom": equiv_dof,
            "equivalent_reduced_chi2": equiv_reduced,
            "identification": "not_determined",
            "diagnostic": "contradiction_score_only; absence classes cannot distinguish all space groups",
        })
    return sorted(results, key=lambda row: (not row["compatible_metric"], row["score"] if row["score"] is not None else float("inf")))
