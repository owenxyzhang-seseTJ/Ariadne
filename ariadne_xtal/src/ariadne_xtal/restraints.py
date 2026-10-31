"""Geometric and displacement pseudo-observations for native refinement.

Every residual is dimensionless: (calculated - target) / supplied sigma.
These are restraints, hence they add observations rather than remove unknowns.
Exact equalities and occupancy sums are implemented separately in refinement.

JSON examples::

    {"type": "distance", "atoms": ["C1", "O1"], "target": 1.23, "sigma": .02}
    {"type": "angle", "atoms": ["C1", "N1", "C2"], "target": 120, "sigma": 2}
    {"type": "similar_distance", "pairs": [["C1", "O1"], ["C2", "O2"]],
     "sigma": .02}
    {"type": "planarity", "atoms": ["C1", "C2", "C3", "N1"], "sigma": .02}
    {"type": "adp_similarity", "atoms": ["C1", "C2"], "sigma": .004}
    {"type": "rigid_bond", "atoms": ["C1", "C2"], "sigma": .004}
    {"type": "adp_isotropy", "atoms": ["C1"], "sigma": .01}

Lengths use angstroms, angles degrees and displacement tensors angstrom^2.
An atom reference can instead be {"label":"O1", "symop":"-x,y+1/2,-z",
"translation":[1,0,0]}.  Explicit references select the literal image.  Plain
labels select nearest periodic images.  For molecules larger than half the
cell, use explicit images rather than relying on nearest-image selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import gemmi
import numpy as np

from .model import Structure


@dataclass(frozen=True)
class SiteReference:
    label: str
    symop: str = "x,y,z"
    translation: tuple[int, int, int] = (0, 0, 0)
    explicit_image: bool = False

    @classmethod
    def from_value(cls, value: Any) -> "SiteReference":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(value)
        if not isinstance(value, dict) or "label" not in value:
            raise ValueError("An atom reference must be a label or a labelled image dictionary")
        extra = set(value) - {"label", "symop", "translation"}
        if extra:
            raise ValueError(f"Unknown atom-reference keys: {sorted(extra)}")
        shift = np.asarray(value.get("translation", [0, 0, 0]), dtype=float)
        if shift.shape != (3,) or not np.all(np.isfinite(shift)) or not np.allclose(shift, np.rint(shift)):
            raise ValueError("Atom-image translation must contain three integers")
        symop = str(value.get("symop", "x,y,z"))
        gemmi.Op(symop)  # syntax validation only; no unrestricted expression evaluation
        return cls(str(value["label"]), symop, tuple(int(x) for x in shift), True)


def _image(structure: Structure, ref: SiteReference) -> tuple[np.ndarray, np.ndarray]:
    atoms = {a.label: a for a in structure.atoms}
    if ref.label not in atoms:
        raise ValueError(f"Unknown atom in restraint: {ref.label}")
    atom = atoms[ref.label]
    op = gemmi.Op(ref.symop)
    desired_rotation = np.asarray(op.rot)
    desired_translation = np.asarray(op.tran, dtype=float) / op.DEN
    valid_operation = False
    for actual in structure.operations:
        difference = desired_translation - np.asarray(actual.tran, dtype=float) / actual.DEN
        if np.array_equal(desired_rotation, actual.rot) and np.allclose(difference, np.rint(difference), atol=1e-9):
            valid_operation = True
            break
    if not valid_operation:
        raise ValueError(f"Restraint image operation is absent from the structure symmetry: {ref.symop}")
    frac = np.asarray(op.apply_to_xyz(atom.frac.tolist())) + np.asarray(ref.translation)
    rot = np.asarray(op.rot, dtype=float) / op.DEN
    cart_rot = structure.cell.matrix @ rot @ np.linalg.inv(structure.cell.matrix)
    tensor = np.eye(3) * atom.u_iso if atom.u_aniso is None else np.asarray(atom.u_aniso)
    return frac, cart_rot @ tensor @ cart_rot.T


def _vector(structure: Structure, left: SiteReference, right: SiteReference) -> np.ndarray:
    a, _ = _image(structure, left)
    b, _ = _image(structure, right)
    delta = b - a
    if not (left.explicit_image or right.explicit_image):
        delta = structure.cell.minimum_image(delta)
    return structure.cell.to_cartesian(delta)


def _symmetric_components(matrix: np.ndarray) -> np.ndarray:
    """An orthonormal six-vector, so its norm is the Frobenius norm."""
    root2 = np.sqrt(2.0)
    return np.array([matrix[0, 0], matrix[1, 1], matrix[2, 2],
                     root2 * matrix[0, 1], root2 * matrix[0, 2], root2 * matrix[1, 2]])


@dataclass
class Restraint:
    """Validated restraint specification with an explicit, stable residual shape."""

    type: str
    atoms: tuple[SiteReference, ...] = ()
    sigma: float = 0.02
    target: float | None = None
    pairs: tuple[tuple[SiteReference, SiteReference], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        def site(ref):
            if not ref.explicit_image:
                return ref.label
            return {"label": ref.label, "symop": ref.symop,
                    "translation": list(ref.translation)}

        data = {"type": self.type, "sigma": self.sigma}
        if self.atoms:
            data["atoms"] = [site(ref) for ref in self.atoms]
        if self.pairs:
            data["pairs"] = [[site(ref) for ref in pair] for pair in self.pairs]
        if self.target is not None:
            data["target"] = self.target
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Restraint":
        if not isinstance(data, dict):
            raise ValueError("Restraints must be Restraint objects or dictionaries")
        unknown = set(data) - {"type", "atoms", "sigma", "target", "pairs"}
        if unknown:
            raise ValueError(f"Unknown restraint keys: {sorted(unknown)}")
        kind = str(data.get("type", ""))
        allowed = {"distance", "angle", "similar_distance", "planarity",
                   "adp_similarity", "rigid_bond", "adp_isotropy"}
        if kind not in allowed:
            raise ValueError(f"Unsupported native restraint: {kind}")
        if "sigma" not in data:
            raise ValueError("A restraint must specify sigma explicitly")
        sigma = float(data["sigma"])
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("A restraint sigma must be finite and positive")
        atoms = tuple(SiteReference.from_value(x) for x in data.get("atoms", []))
        pairs = tuple(tuple(SiteReference.from_value(x) for x in pair)
                      for pair in data.get("pairs", []))
        if any(len(pair) != 2 for pair in pairs):
            raise ValueError("Every similar-distance pair must have two atoms")
        counts = {"distance": 2, "angle": 3, "adp_similarity": 2,
                  "rigid_bond": 2, "adp_isotropy": 1}
        if kind in counts and len(atoms) != counts[kind]:
            raise ValueError(f"{kind} requires {counts[kind]} atoms")
        if kind == "planarity" and len(atoms) < 4:
            raise ValueError("A planarity restraint requires at least four atoms")
        if kind == "similar_distance" and len(pairs) < 2:
            raise ValueError("A similar-distance restraint requires at least two pairs")
        if kind != "similar_distance" and pairs:
            raise ValueError("pairs is used only for similar_distance")
        if kind == "similar_distance" and atoms:
            raise ValueError("similar_distance uses pairs, not atoms")
        target = None if data.get("target") is None else float(data["target"])
        if kind in {"distance", "angle"}:
            if target is None or not np.isfinite(target):
                raise ValueError(f"{kind} requires a finite target")
            if kind == "distance" and target <= 0:
                raise ValueError("A target distance must be positive")
            if kind == "angle" and not 0 < target < 180:
                raise ValueError("A target angle must lie strictly between 0 and 180 degrees")
        elif target is not None:
            raise ValueError(f"{kind} does not use a target field")
        return cls(kind, atoms, sigma, target, pairs)

    def residuals(self, structure: Structure) -> np.ndarray:
        if self.type == "distance":
            length = np.linalg.norm(_vector(structure, *self.atoms))
            return np.array([(length - self.target) / self.sigma])
        if self.type == "angle":
            a, b, c = self.atoms
            ba = _vector(structure, b, a)
            bc = _vector(structure, b, c)
            denom = np.linalg.norm(ba) * np.linalg.norm(bc)
            if denom < 1e-14:
                raise ValueError("An angle restraint contains coincident atoms")
            angle = np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1, 1)))
            return np.array([(angle - self.target) / self.sigma])
        if self.type == "similar_distance":
            distances = np.array([np.linalg.norm(_vector(structure, *pair)) for pair in self.pairs])
            # Each row is the declared difference against the first distance.
            # These are explicit pseudo-observations, not independent measured lengths.
            return (distances[1:] - distances[0]) / self.sigma
        if self.type == "planarity":
            reference = self.atoms[0]
            xyz = np.vstack([np.zeros(3)] + [_vector(structure, reference, x) for x in self.atoms[1:]])
            centred = xyz - xyz.mean(axis=0)
            _, singular, vt = np.linalg.svd(centred, full_matrices=False)
            if singular[1] < 1e-9:
                raise ValueError("A planarity restraint requires a non-collinear atom set")
            normal = vt[-1]
            normal *= 1 if normal[np.argmax(np.abs(normal))] >= 0 else -1
            return centred @ normal / self.sigma
        if self.type == "adp_isotropy":
            _, tensor = _image(structure, self.atoms[0])
            delta = tensor - np.eye(3) * np.trace(tensor) / 3.0
            return _symmetric_components(delta) / self.sigma
        _, left_u = _image(structure, self.atoms[0])
        _, right_u = _image(structure, self.atoms[1])
        if self.type == "adp_similarity":
            return _symmetric_components(left_u - right_u) / self.sigma
        if self.type == "rigid_bond":
            vector = _vector(structure, *self.atoms)
            length = np.linalg.norm(vector)
            if length < 1e-10:
                raise ValueError("A rigid-bond restraint contains coincident atoms")
            direction = vector / length
            return np.array([direction @ (left_u - right_u) @ direction / self.sigma])
        raise ValueError(f"Unsupported restraint type: {self.type}")


def normalize_restraints(values: Iterable[Restraint | dict] | None) -> list[Restraint]:
    return [Restraint.from_dict(x.to_dict() if isinstance(x, Restraint) else x)
            for x in (values or [])]


def restraint_residuals(structure: Structure, values: Iterable[Restraint | dict] | None) -> np.ndarray:
    restraints = normalize_restraints(values)
    if not restraints:
        return np.empty(0, dtype=float)
    result = np.concatenate([x.residuals(structure) for x in restraints])
    if not np.all(np.isfinite(result)):
        raise ValueError("Restraint residuals must remain finite")
    return result
