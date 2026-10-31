"""Lossless SHELX instruction documents and a deliberately strict native subset.

The text representation and the numerical model are separate.  Merely retaining a
directive in a document does not mean that the native solver implements it.
Public format reference: https://shelx.uni-goettingen.de/shelxl_html.php
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import gemmi
import numpy as np

from .model import Atom, Cell, Structure


SHELXL_KEYWORDS = frozenset("""
ABIN ACTA AFIX ANIS ANSC ANSR BASF BIND BLOC BOND BUMP CELL CGLS CHIV CONF
CONN DAMP DANG DEFS DELU DFIX DISP EADP END EQIV EXTI EXYZ FEND FLAT FMAP
FRAG FREE FVAR GRID HFIX HKLF HTAB ISOR LATT LAUE LIST L.S. MERG MORE MOVE
MPLA NCSY NEUT OMIT PART PLAN PRIG REM RESI RIGU RTAB SADI SAME SFAC SHEL
SIMU SIZE SPEC STIR SUMP SWAT SYMM TEMP TITL TWIN TWST UNIT WGHT WIGL WPDB
XNPD ZERR TIME HOPE MOLE
""".split())

# These contain structural information or non-executable documentation only.
NATIVE_KEYWORDS = frozenset({"CELL", "LATT", "SYMM", "SFAC", "UNIT", "TITL",
                             "REM", "END", "HKLF", "FVAR", "DISP"})
# These affect reporting, not the structure-factor model.  They are reported as
# non-translated metadata in audit(); no equivalent output is promised.
METADATA_KEYWORDS = frozenset({"ZERR", "TEMP", "SIZE", "MORE"})


class NativeConversionError(ValueError):
    """The instruction document cannot be represented without semantic loss."""

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("Native SHELX conversion refused:\n" + "\n".join(issues))


@dataclass(frozen=True)
class ShelxInstruction:
    keyword: str
    arguments: tuple[str, ...]
    logical_text: str
    line_number: int
    kind: str  # directive, atom, include, unknown
    active: bool = True


@dataclass
class ShelxDocument:
    raw_text: str
    instructions: list[ShelxInstruction]
    source_path: Path | None = None

    @classmethod
    def from_text(cls, text: str) -> "ShelxDocument":
        records: list[ShelxInstruction] = []
        lines = text.splitlines()
        active = True
        i = 0
        while i < len(lines):
            raw = lines[i]
            lineno = i + 1
            i += 1
            if not raw.strip() or raw[0].isspace():
                continue  # indented non-continuation lines are SHELX comments
            text_parts = []
            literal_text = raw.split(None, 1)[0].upper() in {"TITL", "REM"}
            while True:
                segment = raw if literal_text else raw.split("!", 1)[0]
                continuation = not literal_text and "=" in segment
                text_parts.append(segment.strip() if literal_text else segment.split("=", 1)[0].strip())
                if not continuation:
                    break
                if i >= len(lines) or not lines[i][:1].isspace():
                    raise ValueError(f"SHELX line {lineno}: continuation must be indented")
                raw = lines[i]
                i += 1
            logical = " ".join(text_parts).strip()
            if not logical:
                continue
            words = logical.split()
            keyword = words[0].upper()
            base = keyword.split("_", 1)[0]
            if logical.startswith("+"):
                kind = "include"
            elif base in SHELXL_KEYWORDS:
                kind = "directive"
            elif _looks_like_atom(words):
                kind = "atom"
            else:
                kind = "unknown"
            records.append(ShelxInstruction(keyword, tuple(words[1:]), logical,
                                             lineno, kind, active))
            if base == "END" and active:
                active = False
        return cls(text, records)

    @classmethod
    def from_path(cls, path: str | Path) -> "ShelxDocument":
        path = Path(path)
        result = cls.from_text(path.read_bytes().decode("utf-8", errors="surrogateescape"))
        result.source_path = path
        return result

    def write(self, path: str | Path) -> None:
        """Write byte-for-byte original text, including CRLF and comments."""
        Path(path).write_bytes(self.raw_text.encode("utf-8", errors="surrogateescape"))

    def audit(self) -> dict:
        """Classify directives; unsupported entries are never silently executed."""
        issues: list[str] = []
        metadata: list[str] = []
        fvars: list[str] = []
        for item in self.instructions:
            if not item.active:
                continue
            key = item.keyword.split("_", 1)[0]
            where = f"line {item.line_number}: {item.keyword}"
            if item.kind == "include":
                issues.append(f"{where}: native include expansion is unsupported; use legacy execution")
            elif item.kind == "unknown":
                issues.append(f"{where}: unknown instruction")
            elif item.kind == "directive":
                if key in METADATA_KEYWORDS:
                    metadata.append(f"{where}: retained as document metadata only")
                    if key == "ZERR":
                        try:
                            if any(_number(x) != 0 for x in item.arguments[1:]):
                                issues.append(f"{where}: nonzero cell uncertainties cannot be propagated by the native engine")
                        except ValueError:
                            issues.append(f"{where}: invalid cell uncertainty")
                elif key not in NATIVE_KEYWORDS:
                    issues.append(f"{where}: instruction has no native semantic translation")
                elif key == "HKLF" and item.arguments != ("4",):
                    issues.append(f"{where}: only untransformed HKLF 4 is supported")
                elif key == "FVAR":
                    fvars.extend(item.arguments)
                elif key == "SFAC":
                    if not item.arguments or any(not _is_element(x) for x in item.arguments):
                        issues.append(f"{where}: only named neutral-element SFAC lists are supported")
        if len(fvars) > 1:
            issues.append("FVAR: linked free variables require the legacy engine")
        if fvars:
            try:
                if _number(fvars[0]) <= 0:
                    issues.append("FVAR: overall amplitude scale must be positive")
            except ValueError:
                issues.append("FVAR: invalid overall amplitude scale")
            metadata.append("FVAR overall amplitude scale is not part of Structure; native refinement re-estimates intensity scale")
        if any(x.keyword == "TEMP" and x.active for x in self.instructions):
            if any(x.kind == "atom" and x.active and len(x.arguments) < 6 for x in self.instructions):
                issues.append("TEMP: temperature-dependent default atomic U is unsupported; supply explicit U values")
        return {"native_compatible": not issues, "issues": issues, "metadata": metadata,
                "instruction_count": len(self.instructions),
                "raw_text_preserved": True,
                "native_supported_keywords": sorted(NATIVE_KEYWORDS)}

    def to_structure(self) -> Structure:
        audit = self.audit()
        issues = list(audit["issues"])
        if issues:
            raise NativeConversionError(issues)
        active = [x for x in self.instructions if x.active]
        cells = [x for x in active if x.keyword == "CELL"]
        if len(cells) != 1 or len(cells[0].arguments) != 7:
            raise NativeConversionError(["Exactly one CELL wavelength a b c alpha beta gamma is required"])
        values = [_number(x) for x in cells[0].arguments]
        wavelength, cell = values[0], Cell(*values[1:])
        latt_items = [x for x in active if x.keyword == "LATT"]
        if len(latt_items) > 1:
            raise NativeConversionError(["More than one LATT instruction"])
        if latt_items and (len(latt_items[0].arguments) != 1 or _number(latt_items[0].arguments[0]) % 1):
            raise NativeConversionError(["LATT requires exactly one integer"])
        latt = int(_number(latt_items[0].arguments[0])) if latt_items else 1
        if not latt or abs(latt) > 7:
            raise NativeConversionError(["LATT must be in -7..-1 or 1..7"])
        generators = ["x,y,z"]
        generators.extend("".join(x.arguments) for x in active if x.keyword == "SYMM")
        ops = _shelx_operations(latt, generators)
        found = gemmi.find_spacegroup_by_ops(gemmi.GroupOps(ops))
        group_name = found.xhm() if found else "P 1"
        elements: list[str] = []
        for item in active:
            if item.keyword == "SFAC":
                elements.extend(gemmi.Element(x).name for x in item.arguments)
        if not elements:
            raise NativeConversionError(["No named SFAC elements"])
        dispersion: dict[str, tuple[float, float]] = {}
        for item in active:
            if item.keyword == "DISP":
                if len(item.arguments) not in (3, 4) or not _is_element(item.arguments[0]):
                    raise NativeConversionError([f"line {item.line_number}: expected DISP element fprime fdoubleprime [mu]"])
                dispersion[gemmi.Element(item.arguments[0]).name] = tuple(_number(x) for x in item.arguments[1:3])
        atoms: list[Atom] = []
        for item in active:
            if item.kind != "atom":
                continue
            try:
                args = [_number(x) for x in item.arguments]
                if len(args) not in (4, 5, 6, 11):
                    raise ValueError("expected sfac x y z [sof [Uiso or six Uij]]")
                sfac = int(args[0])
                if sfac != args[0] or not 1 <= sfac <= len(elements):
                    raise ValueError("invalid SFAC index")
                coordinate_values = [_decode_parameter(v) for v in args[1:4]]
                xyz_fixed = [x[1] for x in coordinate_values]
                if len(set(xyz_fixed)) > 1:
                    raise ValueError("mixed fixed/free coordinate components have no native translation")
                xyz = np.array([x[0] for x in coordinate_values])
                multiplicity = _multiplicity(ops, xyz)
                if len(args) >= 5:
                    sof, occupancy_fixed = _decode_parameter(args[4])
                    occupancy = sof * len(ops) / multiplicity
                else:
                    occupancy, occupancy_fixed = 1.0, True
                if not 0 <= occupancy <= 1 + 1e-10:
                    raise ValueError("physical occupancy lies outside [0,1] after special-position conversion")
                u_iso, u_aniso, u_fixed = 0.05, None, False
                if len(args) == 6:
                    if args[5] < 0:
                        raise ValueError("negative Uiso riding multiplier has no native translation")
                    u_iso, u_fixed = _decode_parameter(args[5])
                elif len(args) == 11:
                    decoded = [_decode_parameter(v) for v in args[5:]]
                    if len({x[1] for x in decoded}) != 1:
                        raise ValueError("mixed fixed/free anisotropic components have no native translation")
                    u_fixed = decoded[0][1]
                    u11, u22, u33, u23, u13, u12 = [x[0] for x in decoded]
                    u_cif = np.array([[u11,u12,u13],[u12,u22,u23],[u13,u23,u33]])
                    u_aniso = cif_u_to_cartesian(cell, u_cif)
                    u_iso = float(np.trace(u_aniso) / 3)
                element = elements[sfac - 1]
                fp, fpp = dispersion.get(element, (0.0, 0.0))
                atoms.append(Atom(item.keyword, element, xyz, occupancy=occupancy,
                                  u_iso=u_iso, u_aniso=u_aniso, f_prime=fp,
                                  f_double_prime=fpp, refine_xyz=not xyz_fixed[0],
                                  refine_u=not u_fixed, refine_occupancy=not occupancy_fixed))
            except (ValueError, IndexError) as exc:
                issues.append(f"line {item.line_number}: {item.keyword}: {exc}")
        if issues:
            raise NativeConversionError(issues)
        if not atoms:
            raise NativeConversionError(["No atoms in the native structural subset"])
        if any(x.keyword == "FVAR" for x in active):
            warnings.warn("FVAR overall amplitude scale is not part of Structure; native refinement re-estimates intensity scale", UserWarning, stacklevel=2)
        missing_dispersion = sorted(set(elements) - set(dispersion))
        if missing_dispersion:
            warnings.warn("Native neutral-atom import sets f'=f''=0 for " + ", ".join(missing_dispersion)
                          + "; supply explicit DISP terms to specify anomalous scattering. "
                          "This does not reproduce SHELXL's tabulated dispersion defaults.", UserWarning, stacklevel=2)
        return Structure(cell, atoms, spacegroup=group_name, wavelength=wavelength,
                         sym_ops=[x.triplet() for x in ops])


def _number(text: str) -> float:
    result = float(text.replace("D", "E").replace("d", "e"))
    if not np.isfinite(result):
        raise ValueError("nonfinite number")
    return result


def _is_element(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]{1,2}", text)) and gemmi.Element(text).atomic_number > 0


def _looks_like_atom(words: list[str]) -> bool:
    if len(words) < 5 or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,3}", words[0]):
        return False
    try:
        return all(np.isfinite(_number(x)) for x in words[1:])
    except ValueError:
        return False


def _decode_parameter(value: float) -> tuple[float, bool]:
    # SHELX p is within (-5,5); nearest multiple of ten supplies m.
    m = int(np.floor((value + 5) / 10))
    p = value - 10 * m
    if m == 0:
        return p, False
    if m == 1:
        return p, True
    raise ValueError(f"free-variable parameter code {value:g} has no native translation")


def _shelx_operations(latt: int, generators: list[str]) -> list[gemmi.Op]:
    centering = {
        1: [], 2: ["x+1/2,y+1/2,z+1/2"],
        3: ["x+2/3,y+1/3,z+1/3", "x+1/3,y+2/3,z+2/3"],
        4: ["x,y+1/2,z+1/2", "x+1/2,y,z+1/2", "x+1/2,y+1/2,z"],
        5: ["x,y+1/2,z+1/2"], 6: ["x+1/2,y,z+1/2"],
        7: ["x+1/2,y+1/2,z"],
    }
    strings = generators + centering[abs(latt)] + (["-x,-y,-z"] if latt > 0 else [])
    group = gemmi.GroupOps([gemmi.Op(x) for x in strings])
    group.add_missing_elements()
    return list(group)


def _multiplicity(operations: list[gemmi.Op], xyz: np.ndarray) -> int:
    unique: list[np.ndarray] = []
    for operation in operations:
        p = np.asarray(operation.apply_to_xyz(xyz.tolist())) % 1
        if not any(np.max(np.abs((p-q+0.5) % 1-0.5)) < 1e-7 for q in unique):
            unique.append(p)
    return len(unique)


def cif_u_to_cartesian(cell: Cell, u_cif: np.ndarray) -> np.ndarray:
    """CIF/SHELX Uij -> Cartesian U using A diag(|a*|,|b*|,|c*|)."""
    transform = cell.matrix @ np.diag(np.linalg.norm(cell.reciprocal, axis=0))
    return transform @ np.asarray(u_cif, dtype=float) @ transform.T


def cartesian_u_to_cif(cell: Cell, u_cartesian: np.ndarray) -> np.ndarray:
    transform = cell.matrix @ np.diag(np.linalg.norm(cell.reciprocal, axis=0))
    inverse = np.linalg.inv(transform)
    return inverse @ np.asarray(u_cartesian, dtype=float) @ inverse.T


def structure_to_shelx(structure: Structure) -> str:
    """Write a fresh structural RES; this is not an edit of a source document.

    Arbitrary source constraints, restraints and instructions can only be kept by
    ShelxDocument.write(). Centering/inversion are factored into LATT as required
    by the SHELX format; the remaining coset representatives are written as SYMM.
    """
    elements = list(dict.fromkeys(atom.element for atom in structure.atoms))
    operations = list(structure.operations)
    group = gemmi.GroupOps(operations)
    center_code = {"P":1,"I":2,"R":3,"F":4,"A":5,"B":6,"C":7}.get(group.find_centering())
    if center_code is None:
        raise ValueError("SHELX export does not support this centering setting; use CIF or JSON")
    centric = group.is_centrosymmetric()
    if centric and gemmi.Op("-x,-y,-z").triplet() not in {op.triplet() for op in operations}:
        raise ValueError("SHELX export requires an inversion centre at the origin; change origin explicitly or use CIF/JSON")
    lattice = center_code if centric else -center_code
    representatives = [op for op in group.sym_ops
                       if not centric or np.linalg.det(np.asarray(op.rot)/op.DEN) > 0]
    regenerated = _shelx_operations(lattice,[op.triplet() for op in representatives])
    if {op.triplet() for op in regenerated} != {op.triplet() for op in operations}:
        raise ValueError("Symmetry cannot be factored into SHELX LATT/SYMM without a setting change")
    cell = structure.cell
    lines = ["TITL Ariadne-Xtal native structural export",
             "REM Fresh structural model; source-file instructions are not represented",
             "CELL " + " ".join(f"{x:.12g}" for x in [structure.wavelength,cell.a,cell.b,cell.c,cell.alpha,cell.beta,cell.gamma]),
             f"LATT {lattice}"]
    identity = gemmi.Op("x,y,z").triplet()
    lines.extend("SYMM " + operation.triplet() for operation in representatives if operation.triplet() != identity)
    lines.append("SFAC " + " ".join(elements))
    for element in elements:
        values = {(a.f_prime, a.f_double_prime) for a in structure.atoms if a.element == element}
        if len(values) != 1:
            raise ValueError("SHELX export requires identical dispersion corrections for each element")
        fp, fpp = values.pop()
        # Explicit zeros avoid reintroducing a legacy engine's automatic
        # wavelength-dependent dispersion into the exported native model.
        # SHELXL requires the third numerical DISP field (mu). The native
        # model has no absorption term: explicit zero keeps it disabled; it
        # is not an estimate of the material's absorption coefficient.
        lines.append(f"DISP {element} {fp:.12g} {fpp:.12g} 0")
    composition = {el:0.0 for el in elements}
    for atom in structure.atoms:
        composition[atom.element] += _multiplicity(operations, atom.frac) * atom.occupancy
    lines.append("UNIT " + " ".join(f"{composition[el]:.12g}" for el in elements))
    for atom in structure.atoms:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,3}", atom.label):
            raise ValueError(f"SHELX atom label must have one to four alphanumeric characters: {atom.label}")
        if atom.label.upper() in SHELXL_KEYWORDS:
            raise ValueError(f"SHELX atom label collides with instruction: {atom.label}")
        coords = np.mod(atom.frac, 1) + (0 if atom.refine_xyz else 10)
        sof = atom.occupancy * _multiplicity(operations, atom.frac) / len(operations)
        sof += 0 if atom.refine_occupancy else 10
        args = [*coords, sof]
        if atom.u_aniso is None:
            args.append(atom.u_iso + (0 if atom.refine_u else 10))
        else:
            u = cartesian_u_to_cif(cell, atom.u_aniso)
            args.extend(float(u[i,j]) + (0 if atom.refine_u else 10) for i,j in [(0,0),(1,1),(2,2),(1,2),(0,2),(0,1)])
        header = f"{atom.label} {elements.index(atom.element)+1} "
        if atom.u_aniso is None:
            lines.append(header + " ".join(f"{x:.12g}" for x in args))
        else:
            lines.append(header + " ".join(f"{x:.12g}" for x in args[:7]) + " =")
            lines.append("  " + " ".join(f"{x:.12g}" for x in args[7:]))
    lines.extend(["HKLF 4", "END", ""])
    return "\n".join(lines)
