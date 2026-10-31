"""JSON, small-molecule CIF and SHELX structural/reflection input and output.

CIF and JSON contain models.  They do not reconstruct a SHELXL refinement recipe.
Use ShelxDocument for preservation of a complete INS/RES instruction stream.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import warnings

import gemmi
import numpy as np

from .model import Atom, Cell, Reflections, Structure
from .shelx import ShelxDocument, cartesian_u_to_cif, cif_u_to_cartesian, structure_to_shelx


class SymmetryMismatchWarning(UserWarning):
    """CIF symmetry operations disagree with a supplied group label."""


def read_structure(path: str | Path, *, block_name: str | None = None,
                   diagnostics: list[str] | None = None) -> Structure:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        # Refinement-result JSON deliberately has a nested structure object.
        return Structure.from_dict(obj["structure"] if "structure" in obj else obj)
    if suffix in {".ins", ".res"}:
        return ShelxDocument.from_path(path).to_structure()
    if suffix == ".cif":
        return read_cif(path, block_name=block_name, diagnostics=diagnostics)
    raise ValueError(f"Unsupported structural extension {suffix!r}; use JSON, CIF, INS or RES")


def write_structure(structure: Structure, path: str | Path) -> None:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        path.write_text(json.dumps(structure.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    elif suffix == ".cif":
        write_cif(structure, path)
    elif suffix in {".ins", ".res"}:
        path.write_text(structure_to_shelx(structure), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported structural extension {suffix!r}")


def read_reflections(path: str | Path) -> Reflections:
    """JSON preserves holdouts/precision; other paths are interpreted as HKLF4.

    HKLF4 batch signs are retained literally. Use ``read_hkl`` with its explicit
    ``negative_batch_is_free`` option to interpret the reference-set convention.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        return Reflections.from_dict(json.loads(path.read_text(encoding="utf-8")))
    return read_hkl(path)


def write_reflections(reflections: Reflections, path: str | Path) -> None:
    """Export JSON without numerical rounding or conventional fixed-width HKLF4."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(reflections.to_dict(),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    else:
        write_hkl(reflections,path)


def _cif_number(raw: str | None, *, default: float | None = None) -> float:
    if raw is None or raw in {"?", ".", ""}:
        if default is None:
            raise ValueError("Required CIF numerical value is missing")
        return default
    number = gemmi.cif.as_number(raw)
    if not np.isfinite(number):
        raise ValueError(f"Nonfinite or invalid CIF number {raw!r}")
    return float(number)


def _cif_text(raw: str | None, default: str = "") -> str:
    if raw is None or raw in {"?", ".", ""}:
        return default
    return gemmi.cif.as_string(raw)


def _column(block: gemmi.cif.Block, name: str, n: int, default: str = "?") -> list[str]:
    values = list(block.find_values(name))
    if values and len(values) != n:
        raise ValueError(f"CIF column {name} has {len(values)} values; expected {n}")
    return values if values else [default] * n


def read_cif(path: str | Path, *, block_name: str | None = None,
             diagnostics: list[str] | None = None) -> Structure:
    """Import one explicit small-molecule X-ray model, without guessing metadata.

    Requires a complete cell, symmetry setting/operations, one wavelength,
    fractional coordinates, explicit neutral element symbols and Uiso/Biso or
    complete anisotropic U tensors. Multiple structural blocks require a name.
    Explicit operations override contradictory Hermann-Mauguin labels with a
    warning, optionally appended to ``diagnostics``. Ionic scattering tables,
    neutron/multi-wavelength models, anisotropic B-only tables and reconstructing
    embedded SHELXL refinement recipes are outside this structural reader.
    """
    document = gemmi.cif.read_file(str(path))
    if block_name is not None:
        candidates = [b for b in document if b.name == block_name]
    else:
        candidates = [b for b in document if len(b.find_values("_atom_site_fract_x"))]
    if len(candidates) != 1:
        raise ValueError("CIF must select exactly one structural block; supply block_name for multi-block files")
    block = candidates[0]
    probe = _cif_text(block.find_value("_diffrn_radiation_probe")).lower()
    radiation_type = _cif_text(block.find_value("_diffrn_radiation_type")).lower()
    if any(kind in probe or kind in radiation_type for kind in ["neutron", "electron"]):
        raise ValueError("Native CIF import currently uses X-ray form factors, not neutron/electron scattering")
    cell = Cell(*[_cif_number(block.find_value("_cell_" + name)) for name in
                  ["length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma"]])
    sg_raw = block.find_value("_space_group_name_H-M_alt") or block.find_value("_symmetry_space_group_name_H-M")
    sg_name = _cif_text(sg_raw)
    symmetry_raw = list(block.find_values("_space_group_symop_operation_xyz"))
    if not symmetry_raw:
        symmetry_raw = list(block.find_values("_symmetry_equiv_pos_as_xyz"))
    explicit_ops = [_cif_text(x) for x in symmetry_raw] or None
    hall = _cif_text(block.find_value("_space_group_name_Hall") or block.find_value("_symmetry_space_group_name_Hall"))
    if not explicit_ops and not sg_name and hall:
        explicit_ops = [operation.triplet() for operation in gemmi.symops_from_hall(hall)]
    if not sg_name:
        if not explicit_ops:
            raise ValueError("CIF needs an explicit space-group setting or symmetry operation list; P1 is not assumed")
        inferred = gemmi.find_spacegroup_by_ops(gemmi.GroupOps([gemmi.Op(x) for x in explicit_ops]))
        sg_name = inferred.xhm() if inferred else "P 1"
    if explicit_ops:
        actual_ops = {gemmi.Op(x).triplet() for x in explicit_ops}
        declared_group = gemmi.find_spacegroup_by_name(sg_name)
        if declared_group and actual_ops != {x.triplet() for x in declared_group.operations()}:
            message = (f"CIF declares {sg_name!r}, but its explicit symmetry operations differ. "
                       "Explicit operations are authoritative.")
            warnings.warn(message, SymmetryMismatchWarning, stacklevel=2)
            if diagnostics is not None:
                diagnostics.append(message)
        if not declared_group:
            message = f"CIF group name {sg_name!r} is unrecognized; using explicit symmetry operations"
            warnings.warn(message, SymmetryMismatchWarning, stacklevel=2)
            if diagnostics is not None:
                diagnostics.append(message)
    elif not gemmi.find_spacegroup_by_name(sg_name):
        raise ValueError(f"Unrecognized CIF space group {sg_name!r} and no explicit symmetry operations")
    wavelengths = list(block.find_values("_diffrn_radiation_wavelength"))
    if len(wavelengths) != 1:
        raise ValueError("Native CIF import requires exactly one explicit X-ray wavelength")
    wavelength = _cif_number(wavelengths[0])
    n = len(block.find_values("_atom_site_fract_x"))
    if n == 0:
        raise ValueError("No fractional atom coordinates in CIF")
    labels = _column(block, "_atom_site_label", n)
    elements = _column(block, "_atom_site_type_symbol", n)
    frac_columns = [_column(block, "_atom_site_fract_"+c, n) for c in "xyz"]
    occ = _column(block, "_atom_site_occupancy", n, "1")
    uiso = _column(block, "_atom_site_U_iso_or_equiv", n)
    biso = _column(block, "_atom_site_B_iso_or_equiv", n)
    adp_type = _column(block, "_atom_site_adp_type", n)
    refine = {name:_column(block, "_ariadne_atom_site_"+name, n, str(default))
              for name,default in [("refine_xyz",1),("refine_u",1),("refine_occupancy",0)]}
    per_atom_fp = _column(block, "_ariadne_atom_site_f_prime", n)
    per_atom_fpp = _column(block, "_ariadne_atom_site_f_double_prime", n)
    atom_types = list(block.find_values("_atom_type_symbol"))
    type_dispersion = {}
    if atom_types:
        real = _column(block, "_atom_type_scat_dispersion_real", len(atom_types), "0")
        imag = _column(block, "_atom_type_scat_dispersion_imag", len(atom_types), "0")
        type_dispersion = {_cif_text(el):(_cif_number(a),_cif_number(b)) for el,a,b in zip(atom_types,real,imag)}
    aniso_labels = list(block.find_values("_atom_site_aniso_label"))
    aniso: dict[str,np.ndarray] = {}
    if aniso_labels:
        parts = [_column(block, "_atom_site_aniso_U_"+ij,len(aniso_labels)) for ij in ["11","22","33","12","13","23"]]
        for j,label in enumerate(aniso_labels):
            u11,u22,u33,u12,u13,u23 = [_cif_number(col[j]) for col in parts]
            u = np.array([[u11,u12,u13],[u12,u22,u23],[u13,u23,u33]])
            decoded_label = _cif_text(label)
            if decoded_label in aniso:
                raise ValueError(f"Duplicate CIF anisotropic label {decoded_label}")
            aniso[decoded_label] = cif_u_to_cartesian(cell,u)
    atoms = []
    for i in range(n):
        label = _cif_text(labels[i])
        element_text = _cif_text(elements[i])
        if not element_text:
            raise ValueError(f"CIF atom {label!r} needs an explicit type_symbol; elements are not guessed from labels")
        match = re.match(r"^([A-Za-z]{1,2})(?:[0-9]*[+-])?$", element_text)
        if not match or gemmi.Element(match.group(1)).atomic_number == 0:
            raise ValueError(f"Unsupported CIF scattering type {element_text!r}")
        if element_text != match.group(1):
            raise ValueError(f"Ionic CIF scattering type {element_text!r} has no native neutral-IT92 translation")
        element = gemmi.Element(match.group(1)).name
        u_cart = aniso.get(label)
        if _cif_text(adp_type[i]).lower() in {"uani","bani"} and u_cart is None:
            raise ValueError(f"CIF atom {label} declares anisotropic ADP but has no complete U tensor")
        if u_cart is not None:
            u_value = float(np.trace(u_cart)/3)
        elif uiso[i] not in {"?",".",""}:
            u_value = _cif_number(uiso[i])
        elif biso[i] not in {"?",".",""}:
            u_value = _cif_number(biso[i])/(8*np.pi**2)
        else:
            raise ValueError(f"CIF atom {label} has no displacement parameter")
        fp,fpp = type_dispersion.get(element_text, (0.0,0.0))
        fp = _cif_number(per_atom_fp[i], default=fp)
        fpp = _cif_number(per_atom_fpp[i], default=fpp)
        flags = {}
        for name,values in refine.items():
            flag = _cif_number(values[i])
            if flag not in (0,1):
                raise ValueError(f"CIF native refinement flag {name} must be 0 or 1")
            flags[name] = bool(flag)
        atoms.append(Atom(label,element,[_cif_number(col[i]) for col in frac_columns],
                          occupancy=_cif_number(occ[i]),u_iso=u_value,u_aniso=u_cart,
                          f_prime=fp,f_double_prime=fpp,**flags))
    unknown_aniso = set(aniso)-{a.label for a in atoms}
    if unknown_aniso:
        raise ValueError(f"Anisotropic CIF rows without matching atoms: {sorted(unknown_aniso)}")
    return Structure(cell,atoms,spacegroup=sg_name,wavelength=wavelength,sym_ops=explicit_ops)


def write_cif(structure: Structure, path: str | Path) -> None:
    document = gemmi.cif.Document()
    block = document.add_new_block("ariadne_xtal")
    for name in ["a","b","c","alpha","beta","gamma"]:
        prefix = "length_" if name in "abc" else "angle_"
        block.set_pair("_cell_"+prefix+name, f"{getattr(structure.cell,name):.15g}")
    operations = list(structure.operations)
    actual_group = gemmi.find_spacegroup_by_ops(gemmi.GroupOps(operations))
    # Downstream programs may ignore explicit operations. Correct the label from
    # the actual group, instead of propagating a contradictory source label.
    block.set_pair("_space_group_name_H-M_alt", gemmi.cif.quote(actual_group.xhm()) if actual_group else "?")
    block.set_pair("_diffrn_radiation_wavelength", f"{structure.wavelength:.15g}")
    block.set_pair("_audit_creation_method", gemmi.cif.quote("Ariadne-Xtal native structural export"))
    sym_loop = block.init_loop("_space_group_symop_", ["id","operation_xyz"])
    for j,operation in enumerate(operations,1):
        sym_loop.add_row([str(j),gemmi.cif.quote(operation.triplet())])
    atom_tags = ["_atom_site_label","_atom_site_type_symbol","_atom_site_fract_x","_atom_site_fract_y",
                 "_atom_site_fract_z","_atom_site_occupancy","_atom_site_U_iso_or_equiv","_atom_site_adp_type",
                 "_ariadne_atom_site_refine_xyz","_ariadne_atom_site_refine_u","_ariadne_atom_site_refine_occupancy",
                 "_ariadne_atom_site_f_prime","_ariadne_atom_site_f_double_prime"]
    atom_loop = block.init_loop("",atom_tags)
    for atom in structure.atoms:
        ueq = atom.u_iso if atom.u_aniso is None else float(np.trace(atom.u_aniso)/3)
        atom_loop.add_row([gemmi.cif.quote(atom.label),atom.element,
                           *[f"{v:.15g}" for v in atom.frac],f"{atom.occupancy:.15g}",f"{ueq:.15g}",
                           "Uiso" if atom.u_aniso is None else "Uani",str(int(atom.refine_xyz)),
                           str(int(atom.refine_u)),str(int(atom.refine_occupancy)),
                           f"{atom.f_prime:.15g}",f"{atom.f_double_prime:.15g}"])
    anisotropic = [a for a in structure.atoms if a.u_aniso is not None]
    if anisotropic:
        loop = block.init_loop("_atom_site_aniso_",["label","U_11","U_22","U_33","U_12","U_13","U_23"])
        for atom in anisotropic:
            u = cartesian_u_to_cif(structure.cell,atom.u_aniso)
            loop.add_row([gemmi.cif.quote(atom.label),*[f"{u[i,j]:.15g}" for i,j in [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]]])
    document.write_file(str(path))


def read_hkl(path: str | Path, *, negative_batch_is_free: bool = False) -> Reflections:
    """Read HKLF4 (3I4,2F8.2,I4), with an explicit whitespace-format fallback.

    Negative observations are retained.  No intensity cutoff, merging, or batch
    reinterpretation is automatic.  Optional ``negative_batch_is_free`` applies
    SHELXL's documented HKLF4 reference-set convention while retaining raw batch.
    In fixed F8.2 fields a number without a decimal/exponent has two implied
    decimal places, as required by Fortran formatted input.
    """
    hkl, intensities, sigmas, batches = [],[],[],[]
    for lineno,raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(),1):
        if not raw.strip():
            if hkl:
                break
            continue
        if raw.lstrip().startswith(("#","!")):
            continue
        parsed = None
        if len(raw) >= 28:
            try:
                indices = [int(raw[j:j+4]) for j in (0,4,8)]
                values = [_fixed_f82(raw[12:20]), _fixed_f82(raw[20:28])]
                batch = int(raw[28:32]) if raw[28:32].strip() else 1
                if raw[32:].strip():
                    raise ValueError("extra columns")
                parsed = indices,values,batch
            except ValueError:
                parsed = None
        if parsed is None:
            parts = raw.split()
            if len(parts) not in (5,6):
                raise ValueError(f"HKLF4 line {lineno}: expected five or six fields")
            try:
                indices = [int(x) for x in parts[:3]]
                values = [float(x.replace("D","E").replace("d","e")) for x in parts[3:5]]
                batch = int(parts[5]) if len(parts) == 6 else 1
                parsed = indices,values,batch
            except ValueError as exc:
                raise ValueError(f"HKLF4 line {lineno}: invalid numeric field") from exc
        indices,values,batch = parsed
        if indices == [0,0,0]:
            break
        hkl.append(indices);intensities.append(values[0]);sigmas.append(values[1]);batches.append(batch)
    if not hkl:
        raise ValueError("HKLF4 contains no reflections")
    batch_array = np.asarray(batches,dtype=int)
    free_flags = batch_array < 0 if negative_batch_is_free else None
    return Reflections(np.asarray(hkl,dtype=int),np.asarray(intensities),np.asarray(sigmas),
                       batch=batch_array,free_flags=free_flags)


def _fixed_f82(field: str) -> float:
    value = float(field.replace("D","E").replace("d","e"))
    if not any(char in field for char in ".EeDd"):
        value /= 100.0
    return value


def _float_field(value: float) -> str:
    """Eight-column decimal, then scientific notation; never silently rescale."""
    candidates = []
    for decimals in range(8,-1,-1):
        string = f"{value:.{decimals}f}"
        if decimals == 0:
            string += "."  # avoid implied Fortran decimal points
        elif string.startswith("0."):
            string = string[1:]
        elif string.startswith("-0."):
            string = "-" + string[2:]
        if len(string) <= 8 and (value == 0 or float(string) != 0):
            candidates.append(string)
    for decimals in range(6,-1,-1):
        string = f"{value:.{decimals}E}"
        string = re.sub(r"E\+0*", "E", string)
        string = re.sub(r"E-0*", "E-", string)
        if string.endswith(("E", "E-")):
            string += "0"
        if len(string) <= 8:
            candidates.append(string)
    if candidates:
        # Select the representation with least rounding error. A fixed decimal
        # wins exact ties, keeping conventional fields readable.
        return min(candidates,key=lambda candidate:abs(float(candidate)-value)).rjust(8)
    raise ValueError(f"Value {value} cannot be represented in HKLF4 F8.2 field")


def write_hkl(reflections: Reflections, path: str | Path, *, fixed_width: bool = True) -> None:
    """Write HKLF4; JSON is preferred when lossless floating-point roundtrip matters."""
    batches = np.ones(len(reflections),dtype=int) if reflections.batch is None else reflections.batch
    if reflections.free_flags is not None and np.any(reflections.free_flags):
        # Avoid silently losing independently supplied holdout assignments.
        expected = np.asarray(batches) < 0
        if not np.array_equal(expected,reflections.free_flags):
            raise ValueError("HKLF4 batch signs do not encode these free_flags; export JSON or set batch signs explicitly")
    lines = []
    for indices,intensity,sigma,batch in zip(reflections.hkl,reflections.intensity,reflections.sigma,batches):
        if fixed_width:
            integer_fields = [f"{int(v):4d}" for v in [*indices,batch]]
            if any(len(x) != 4 for x in integer_fields):
                raise ValueError("HKLF4 integer exceeds four-column width")
            lines.append("".join(integer_fields[:3])+_float_field(float(intensity))+_float_field(float(sigma))+integer_fields[3])
        else:
            lines.append(" ".join([*[str(int(x)) for x in indices],f"{float(intensity):.17g}",f"{float(sigma):.17g}",str(int(batch))]))
    lines.append("   0   0   0    0.00    0.00   0" if fixed_width else "0 0 0 0 0 0")
    Path(path).write_text("\n".join(lines)+"\n",encoding="utf-8")
