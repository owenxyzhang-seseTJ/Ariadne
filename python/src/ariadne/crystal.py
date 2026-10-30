"""Native CIF validation and explicit kinematic X-ray predictions using Gemmi.

The initial reconstruction supports real IT92 form factors (no anomalous terms),
CIF chemical occupancies, symmetry expansion and anisotropic displacement factors.
It rejects unspecified/conflicting symmetry and does not fill missing guest atoms.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any
import math

import numpy as np


def _gemmi():
    try:
        import gemmi
        return gemmi
    except ImportError as exc:
        raise RuntimeError('Native CIF operations require gemmi>=0.7.4,<0.8') from exc


def load_cif(path: str | Path, block: str | None = None):
    gemmi = _gemmi()
    doc = gemmi.cif.read_file(str(path))
    blocks = [b for b in doc if b.find_value('_cell_length_a') is not None]
    if block is None:
        if len(blocks) != 1:
            raise ValueError('select exactly one structural CIF block explicitly')
        selected = blocks[0]
    else:
        selected = doc.find_block(block)
        if selected is None:
            raise ValueError(f'CIF block not found: {block}')
    structure = gemmi.make_small_structure_from_block(selected)
    return structure, selected


def audit_cif(path: str | Path, *, block: str | None = None,
              expected_elements: list[str] | None = None) -> dict[str, Any]:
    gemmi = _gemmi()
    st, cif = load_cif(path, block)
    errors, warnings = [], []
    cell = list(st.cell.parameters)
    if not all(math.isfinite(v) for v in cell) or min(cell[:3]) <= 0 or not all(0 < v < 180 for v in cell[3:]) or not math.isfinite(st.cell.volume) or st.cell.volume <= 0:
        errors.append('invalid unit cell')
    if st.spacegroup is None:
        errors.append('space group is unresolved')
    if st.spacegroup is not None:
        message = st.check_spacegroup()
        if message:
            errors.append('conflicting symmetry declarations: '+message)
        if st.symops:
            observed = {gemmi.Op(x).triplet() for x in st.symops}
            expected = {op.triplet() for op in st.spacegroup.operations()}
            if observed != expected:
                errors.append('explicit symmetry operations disagree with the resolved space group')
    if not st.sites:
        errors.append('no atomic sites')
    names: set[str] = set()
    elements: set[str] = set()
    for site in st.sites:
        if site.label in names:
            errors.append(f'duplicate atom label: {site.label}')
        names.add(site.label)
        elements.add(site.element.name)
        vals = [site.fract.x, site.fract.y, site.fract.z, site.occ]
        if not all(math.isfinite(v) for v in vals):
            errors.append(f'nonfinite site: {site.label}')
        if site.element.atomic_number <= 0:
            errors.append(f'unknown element: {site.label}')
        if not 0 <= site.occ <= 1:
            errors.append(f'chemical occupancy outside [0,1]: {site.label}')
        an = site.aniso
        U = np.array([[an.u11, an.u12, an.u13], [an.u12, an.u22, an.u23], [an.u13, an.u23, an.u33]])
        if not np.all(np.isfinite(U)):
            errors.append(f'nonfinite anisotropic displacement: {site.label}')
        elif np.any(U):
            if np.linalg.eigvalsh(U)[0] <= 0:
                errors.append(f'non-positive anisotropic displacement tensor: {site.label}')
        elif not math.isfinite(site.u_iso):
            errors.append(f'nonfinite isotropic displacement: {site.label}')
        elif site.u_iso < 0:
            errors.append(f'negative isotropic displacement: {site.label}')
        elif site.u_iso == 0:
            warnings.append(f'zero isotropic displacement: {site.label}')
    if expected_elements is not None:
        unknown = elements-set(expected_elements)
        missing = set(expected_elements)-elements
        if unknown:
            errors.append('unexpected elements: '+', '.join(sorted(unknown)))
        if missing:
            warnings.append('expected but unmodelled elements: '+', '.join(sorted(missing)))
    metrics = {}
    for short, tag in [('R1', '_refine_ls_R_factor_gt'), ('wR2', '_refine_ls_wR_factor_ref'),
                       ('goodness_of_fit', '_refine_ls_goodness_of_fit_ref'),
                       ('difference_peak', '_refine_diff_density_max'), ('difference_hole', '_refine_diff_density_min')]:
        value = cif.find_value(tag)
        if value and value not in ('.', '?'):
            try:
                number = gemmi.cif.as_number(value)
                if math.isfinite(number):
                    metrics[short] = number
            except (ValueError, TypeError):
                warnings.append(f'unreadable metric: {tag}')
    return dict(schema='ariadne-crystal-audit/1', backend='gemmi', backend_version=gemmi.__version__,
                block=cif.name, cell=cell, spacegroup=st.spacegroup.xhm() if st.spacegroup else None,
                site_count=len(st.sites), elements=sorted(elements), errors=errors, warnings=warnings,
                metrics=metrics, model_eligible=not errors, scientific_acceptance='not_established')


def transform_hkl(hkl: list[int], matrix: list[list[Any]] | None) -> list[int]:
    if len(hkl) != 3 or any(isinstance(x, bool) or not isinstance(x, int) for x in hkl):
        raise ValueError('HKL must contain three exact integers')
    if matrix is None:
        return hkl
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError('index transform must be 3 by 3')
    M = [[Fraction(str(v)) for v in row] for row in matrix]
    det = (M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
           -M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
           +M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]))
    if det == 0:
        raise ValueError('singular index transform')
    mapped = [sum(M[i][j]*hkl[j] for j in range(3)) for i in range(3)]
    if any(v.denominator != 1 for v in mapped):
        raise ValueError('index transform yields a nonintegral reflection')
    return [int(v) for v in mapped]


def predict(cif_paths: list[str | Path], reflections: list[dict[str, Any]], *,
            blocks: list[str | None] | None = None, special_position_tolerance: float = 1e-5) -> dict[str, Any]:
    gemmi = _gemmi()
    if not cif_paths or not reflections:
        raise ValueError('CIF candidates and reflections are required')
    if not 0 < special_position_tolerance <= .01:
        raise ValueError('special-position tolerance must be in (0,0.01] angstrom')
    blocks = blocks or [None]*len(cif_paths)
    if len(blocks) != len(cif_paths):
        raise ValueError('one block selector is required per CIF')
    if len({r['id'] for r in reflections}) != len(reflections):
        raise ValueError('reflection IDs must be unique')
    all_values, audits = [], []
    for path, block in zip(cif_paths, blocks):
        audit = audit_cif(path, block=block)
        if not audit['model_eligible']:
            raise ValueError('CIF audit failed: '+'; '.join(audit['errors']))
        st, native_block = load_cif(path, block)
        probe = native_block.find_value('_diffrn_radiation_probe')
        if probe and str(probe).strip("'\"").lower() not in {'x-ray', 'xray', '.', '?'}:
            raise ValueError('this predictor is for X-ray scattering only')
        if audits and not np.allclose(audit['cell'], audits[0]['cell'], rtol=1e-6, atol=1e-6):
            raise ValueError('ensemble CIFs must share a declared unit-cell/index setting')
        # Explicitly convert CIF chemical occupancy once, on a fresh object.
        st.change_occupancies_to_crystallographic(special_position_tolerance)
        calculator = gemmi.StructureFactorCalculatorX(st.cell)
        values = []
        for item in reflections:
            hkl = transform_hkl(item['hkl'], item.get('index_transform'))
            if hkl == [0, 0, 0]:
                raise ValueError('F000 is not an experimental reflection')
            F = calculator.calculate_sf_from_small_structure(st, hkl)
            value = float(abs(F)**2)
            if not math.isfinite(value):
                raise ValueError('nonfinite calculated intensity')
            values.append(value)
        all_values.append(values)
        audits.append(audit)
    return dict(schema='ariadne-structure-predictions/1', ids=[r['id'] for r in reflections],
                predictions=np.asarray(all_values).T.tolist(), audits=audits,
                scattering='kinematic X-ray IT92, no anomalous or bulk-solvent correction',
                intensity_unit='electron_squared', occupancy_convention='CIF chemical -> crystallographic',
                special_position_tolerance_angstrom=special_position_tolerance)
