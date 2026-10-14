#!/usr/bin/env python3
"""Evidence-preserving DIALS spot curation and diagnostics; run via dials.python."""
from __future__ import annotations

import argparse
import csv
import copy
import json
from pathlib import Path
import sys

import numpy as np
from dials.array_family import flex
from dxtbx.model.experiment_list import ExperimentListFactory
from dxtbx.model.experiment_list import ExperimentList
from dxtbx.model.crystal import CrystalFactory


def save_json(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError('Refusing to overwrite ' + str(path))
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def load_pair(expt, refl):
    experiments = ExperimentListFactory.from_json_file(str(expt), check_format=False)
    reflections = flex.reflection_table.from_file(str(refl))
    if not experiments or not len(reflections):
        raise ValueError('Empty experiments or reflections')
    return experiments, reflections


def reciprocal(experiments, reflections):
    if 'rlp' not in reflections:
        # DIALS must have actual experiment/imageset associations for this mapping.
        reflections.centroid_px_to_mm(experiments)
        reflections.map_centroids_to_reciprocal_space(experiments)
    q = np.asarray(reflections['rlp'], dtype=float)
    if q.shape != (len(reflections), 3):
        raise ValueError('Unexpected reciprocal coordinate array')
    return q


def harvest(args):
    expts, refs = load_pair(args.expt, args.refl)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    q = reciprocal(expts, refs)
    reasons = np.zeros(len(refs), dtype=np.int32)
    norm = np.linalg.norm(q, axis=1)
    finite = np.all(np.isfinite(q), axis=1) & (norm > 0)
    reasons[~finite] |= 1
    d = np.full(len(refs), np.nan)
    d[finite] = 1 / norm[finite]
    if args.d_min is not None:
        reasons[finite & (d < args.d_min)] |= 2
    if args.d_max is not None:
        reasons[finite & (d > args.d_max)] |= 4
    if args.excluded_shells:
        for token in args.excluded_shells.split(','):
            lo, hi = map(float, token.split(':'))
            if not (0 < lo < hi):
                raise ValueError('Each shell must have 0 < low:high in Angstrom')
            reasons[finite & (d >= lo) & (d <= hi)] |= 8
    if args.min_snr is not None:
        for key in ('intensity.sum.value', 'intensity.sum.variance'):
            if key not in refs:
                raise ValueError('SNR filter requested but column unavailable: ' + key)
        val = np.asarray(refs['intensity.sum.value'], dtype=float)
        var = np.asarray(refs['intensity.sum.variance'], dtype=float)
        good = np.isfinite(val) & np.isfinite(var) & (var > 0)
        snr = np.full(len(refs), -np.inf)
        snr[good] = val[good] / np.sqrt(var[good])
        reasons[snr < args.min_snr] |= 16
    if args.min_span_frames is not None or args.max_span_frames is not None:
        if 'bbox' not in refs:
            raise ValueError('Frame-span filter requested but bbox unavailable')
        bbox = np.asarray(refs['bbox'], dtype=int)
        span = bbox[:, 5] - bbox[:, 4]
        if args.min_span_frames is not None:
            reasons[span < args.min_span_frames] |= 32
        if args.max_span_frames is not None:
            reasons[span > args.max_span_frames] |= 64
    if 'scxrd_source_row' in refs or 'scxrd_curation_reason' in refs:
        raise ValueError('Already curated input; use the original full table for a new branch')
    refs['scxrd_source_row'] = flex.int(list(range(len(refs))))
    refs['scxrd_curation_reason'] = flex.int(reasons.tolist())
    keep = reasons == 0
    refs.as_file(str(out / 'all_spots.refl'))
    refs.select(flex.bool(keep.tolist())).as_file(str(out / 'indexing_spots.refl'))
    refs.select(flex.bool((~keep).tolist())).as_file(str(out / 'excluded_spots.refl'))
    np.savetxt(out / 'curation.csv', np.column_stack((np.arange(len(refs)), reasons, d)),
               delimiter=',', header='source_row,reason_bits,d_A', comments='', fmt=['%d', '%d', '%.8g'])
    save_json(out / 'harvest.json', {
        'purpose': 'indexing subset only; not a final integration mask',
        'input_expt': str(args.expt.resolve()), 'input_refl': str(args.refl.resolve()),
        'spots_total': len(refs), 'spots_kept': int(keep.sum()),
        'parameters': {k: v for k, v in vars(args).items() if k not in ('func', 'action', 'expt', 'refl', 'output')},
        'reason_bits': {'1': 'nonfinite_or_zero_rlp', '2': 'high_resolution_cutoff',
                        '4': 'low_resolution_cutoff', '8': 'explicit_shell',
                        '16': 'explicit_snr', '32': 'explicit_min_span', '64': 'explicit_max_span'},
        'periodicity_filter': 'none', 'mosaicity_estimate': None})


def model_summary(args):
    expts, refs = load_pair(args.expt, args.refl)
    crystals = []
    crystal_objects = []
    experiment_rows = []
    ids = np.asarray(refs['id'], dtype=int) if 'id' in refs else np.full(len(refs), -1)
    for i, expt in enumerate(expts):
        crystal = expt.crystal
        domain = None
        if crystal is not None:
            domain = next((j for j, c in enumerate(crystal_objects) if c is crystal), None)
            if domain is None:
                domain = len(crystal_objects)
                crystal_objects.append(crystal)
                crystals.append({'crystal_model_id': domain, 'cell': list(crystal.get_unit_cell().parameters()),
                                 'space_group_model': str(crystal.get_space_group().info()),
                                 'A': list(crystal.get_A()), 'U': list(crystal.get_U()), 'B': list(crystal.get_B()),
                                 'matrix_convention': 'q=A*h, h and q column vectors; q in inverse Angstrom'})
        experiment_rows.append({'experiment_id': i, 'identifier': expt.identifier,
                                'crystal_model_id': domain, 'assigned_rows': int((ids == i).sum()),
                                'scan_oscillation': list(expt.scan.get_oscillation()) if expt.scan else None,
                                'wavelength_A': expt.beam.get_wavelength() if expt.beam else None})
    if 'miller_index' in refs:
        h = np.asarray(refs['miller_index'], dtype=int)
        assigned = (ids >= 0) & np.any(h != 0, axis=1)
    else:
        assigned = ids >= 0
    save_json(args.output, {'experiments': experiment_rows, 'crystal_models': crystals,
                           'rows_total': len(refs), 'rows_indexed': int(assigned.sum()),
                           'rows_unindexed': int((~assigned).sum()),
                           'indexing_fraction_of_this_table': float(assigned.mean()),
                           'twin_classification': 'requires independent lattice/overlap evidence'})


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    expts, refs = load_pair(args.expt, args.refl)
    q = reciprocal(expts, refs)
    args.output.mkdir(parents=True, exist_ok=False)
    finite = np.all(np.isfinite(q), axis=1)
    q = q[finite]
    ids = np.asarray(refs['id'], dtype=int)[finite] if 'id' in refs else np.full(len(q), -1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=180)  # 创建三幅正交投影，设置尺寸和导出分辨率。
    for ax, (i, j) in zip(axes, ((0, 1), (0, 2), (1, 2))):
        for label in np.unique(ids):
            sel = ids == label
            ax.scatter(q[sel, i], q[sel, j], s=2, alpha=0.4, label=f'experiment {label}')  # 按实验编号显示点，避免误称晶畴。
        ax.set_xlabel(f'q{i+1} (Å⁻¹)')  # 标注倒易坐标和单位。
        ax.set_ylabel(f'q{j+1} (Å⁻¹)')  # 标注倒易坐标和单位。
        ax.set_aspect('equal')  # 保持倒易距离尺度一致。
    axes[0].legend(fontsize=6)  # 显示实验编号图例。
    fig.tight_layout()  # 避免标签拥挤。
    fig.savefig(args.output / 'reciprocal_projections.png')  # 保存实际点云投影图。
    plt.close(fig)
    crystal_objects = []
    for expt in expts:
        if expt.crystal is not None and not any(c is expt.crystal for c in crystal_objects):
            crystal_objects.append(expt.crystal)
    if not (0 <= args.crystal < len(crystal_objects)):
        raise ValueError('Select an available unique crystal model number')
    A = np.asarray(crystal_objects[args.crystal].get_A(), dtype=float).reshape(3, 3)
    frac = np.linalg.solve(A, q.T).T
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=180)  # 为零层和半整数层创建四个切片面板。
    slices = ((0, 0., 1, 2, '0kl'), (1, 0., 0, 2, 'h0l'),
              (2, 0., 0, 1, 'hk0'), (0, .5, 1, 2, '0.5kl'))
    for ax, (normal, level, i, j, title) in zip(axes.flat, slices):
        sel = np.abs(frac[:, normal] - level) <= args.layer_half_width
        ax.scatter(frac[sel, i], frac[sel, j], s=3, alpha=.5)  # 用连续分数指数显示未指标化点和候选卫星点。
        ax.set_title(title + ' point slice')  # 明确它是点切片而非像素重构图。
        ax.set_xlabel('hkl'[i] + ' (fractional)')  # 横轴保留分数坐标定义。
        ax.set_ylabel('hkl'[j] + ' (fractional)')  # 纵轴保留分数坐标定义。
        ax.set_aspect('equal')  # 避免晶格形状被视觉拉伸。
    fig.tight_layout()  # 调整面板布局。
    fig.savefig(args.output / 'fractional_slices.png')  # 保存连续指数切片图。
    plt.close(fig)
    save_json(args.output / 'plot_context.json', {'crystal_model_id': args.crystal,
              'layer_half_width_fractional': args.layer_half_width,
              'geometry': 'static A matrix; scan-varying orientations require per-angle reconstruction',
              'limitations': 'spot cloud only; not raw-pixel precession or an extinction test'})


def scale_audit(args):
    refs = flex.reflection_table.from_file(str(args.refl))
    report = {'rows_total': len(refs), 'scope': 'rows retained in this table only',
              'note': 'Rejected-before-table observations need upstream logs; no physical cause inferred'}
    for name in ('outlier_in_scaling', 'bad_for_scaling', 'integrated_sum', 'integrated_prf'):
        flag = getattr(flex.reflection_table.flags, name, None)
        report[name] = int(np.asarray(refs.get_flags(flag), dtype=bool).sum()) if flag is not None and 'flags' in refs else None
    report['inverse_scale_factor'] = None
    if 'inverse_scale_factor' in refs:
        values = np.asarray(refs['inverse_scale_factor'], dtype=float)
        valid = np.isfinite(values) & (values > 0)
        report['invalid_scale_rows'] = int((~valid).sum())
        if valid.any():
            report['inverse_scale_factor'] = {'min': float(values[valid].min()), 'max': float(values[valid].max()),
                                             'quantiles_01_50_99': np.quantile(values[valid], [.01, .5, .99]).tolist()}
    if 'id' in refs:
        ids = np.asarray(refs['id'], dtype=int)
        report['experiment_row_counts'] = {str(i): int((ids == i).sum()) for i in np.unique(ids)}
    save_json(args.output, report)


def source_indices(refs):
    rows = np.asarray(refs['scxrd_source_row'], dtype=int) if 'scxrd_source_row' in refs else np.arange(len(refs))
    if len(set(rows.tolist())) != len(rows):
        raise ValueError('Nonunique source_row mapping')
    return rows


def export_q(args):
    expts, refs = load_pair(args.expt, args.refl)
    original_variance_available = 'xyzobs.px.variance' in refs
    covariances = None
    if args.with_covariance:
        if not original_variance_available:
            raise ValueError('No measured centroid variances; do not export DIALS fallback unit variances as measured covariance')
        refs.centroid_px_to_mm(expts)
        refs.map_centroids_to_reciprocal_space(expts)
        observed = np.asarray(refs['xyzobs.px.value'], dtype=float)
        variances = np.asarray(refs['xyzobs.px.variance'], dtype=float)
        if not np.all(np.isfinite(variances)) or np.any(variances < 0) or args.jacobian_step <= 0:
            raise ValueError('Invalid centroid variances or Jacobian step')
        jacobian = np.empty((len(refs), 3, 3))
        for axis in range(3):
            reciprocal_sides = []
            for sign in (-1, 1):
                displaced = refs.copy()
                coordinates = observed.copy()
                coordinates[:, axis] += sign * args.jacobian_step
                displaced['xyzobs.px.value'] = flex.vec3_double(coordinates.tolist())
                displaced.centroid_px_to_mm(expts)
                displaced.map_centroids_to_reciprocal_space(expts)
                reciprocal_sides.append(np.asarray(displaced['rlp'], dtype=float))
            jacobian[:, :, axis] = (reciprocal_sides[1] - reciprocal_sides[0]) / (2 * args.jacobian_step)
        covariances = np.einsum('nik,nk,njk->nij', jacobian, variances, jacobian)
    q = reciprocal(expts, refs)
    rows = source_indices(refs)
    valid = np.all(np.isfinite(q), axis=1) & (np.linalg.norm(q, axis=1) > 0)
    images = np.asarray(refs['imageset_id'], dtype=int) if 'imageset_id' in refs else np.asarray(refs['id'], dtype=int)
    if args.output.exists() or args.output.with_suffix('.context.json').exists():
        raise ValueError('Output already exists')
    with args.output.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        names = ['source_row', 'q_x', 'q_y', 'q_z', 'imageset_or_experiment_id']
        if covariances is not None:
            names += ['q_cov_xx', 'q_cov_yy', 'q_cov_zz', 'q_cov_xy', 'q_cov_xz', 'q_cov_yz']
        writer.writerow(names)
        for i in np.flatnonzero(valid):
            values = [int(rows[i]), *q[i], int(images[i])]
            if covariances is not None:
                c = covariances[i]
                values += [c[0, 0], c[1, 1], c[2, 2], c[0, 1], c[0, 2], c[1, 2]]
            writer.writerow(values)
    save_json(args.output.with_suffix('.context.json'), {
        'input_expt': str(args.expt.resolve()), 'input_refl': str(args.refl.resolve()),
        'q_units': 'inverse_Angstrom_without_2pi', 'coordinate_convention': 'DIALS rlp in common spindle-fixed reference frame',
        'rows_written': int(valid.sum()), 'invalid_source_rows': rows[~valid].tolist(),
        'source_mapping': 'scxrd_source_row if present; otherwise original table row index',
        'covariance': 'finite-difference J diag(centroid_variance) J^T; fixed instrument geometry; centroid correlations unavailable' if covariances is not None else 'not propagated',
        'constraint': 'do not change reflection row mapping between q export and candidate split'})


def split_candidates(args):
    _, refs = load_pair(args.expt, args.refl)
    result = json.loads(args.domains.read_text(encoding='utf-8'))
    if result.get('schema') != 'scxrd-domain-candidates/1':
        raise ValueError('Unsupported domain candidates schema')
    source_context = Path(result['source_csv']).with_suffix('.context.json')
    if not source_context.is_file():
        raise ValueError('Missing q export context; cannot verify reflection source mapping')
    context = json.loads(source_context.read_text())
    if Path(context['input_expt']).resolve() != args.expt.resolve() or Path(context['input_refl']).resolve() != args.refl.resolve():
        raise ValueError('Use the exact experiments and reflection table that supplied q export')
    rows = source_indices(refs)
    if 'scxrd_source_row' not in refs:
        refs['scxrd_source_row'] = flex.int(rows.tolist())
    existing = set(rows.tolist())
    args.output.mkdir(parents=True, exist_ok=False)
    union = set()
    manifest = []
    for candidate in result['candidates']:
        wanted = set(candidate['source_rows'])
        if not wanted <= existing:
            raise ValueError('Candidate references unknown source rows')
        union.update(wanted)
        selected = np.asarray([int(r) in wanted for r in rows], dtype=bool)
        name = f'candidate_{candidate["candidate_id"]:03d}.refl'
        refs.select(flex.bool(selected.tolist())).as_file(str(args.output / name))
        manifest.append({'candidate_id': candidate['candidate_id'], 'refl': name,
                         'rows': int(selected.sum()), 'index_with_expt': str(args.expt.resolve()),
                         'status': 'requires_independent_DIALS_indexing'})
    unused = np.asarray([int(r) not in union for r in rows], dtype=bool)
    refs.select(flex.bool(unused.tolist())).as_file(str(args.output / 'unassigned.refl'))
    save_json(args.output / 'manifest.json', {'candidates': manifest,
              'unassigned_rows': int(unused.sum()), 'ambiguity': 'candidate sets may overlap; not disjoint observations'})


def validate_domain(args):
    expts, refs = load_pair(args.expt, args.refl)
    q = reciprocal(expts, refs)
    ids = np.asarray(refs['id'], dtype=int)
    hkl = np.asarray(refs['miller_index'], dtype=int)
    models = []
    grouped = {}
    for i, exp in enumerate(expts):
        if exp.crystal is None:
            continue
        model_id = next((j for j, c in enumerate(models) if c is exp.crystal), None)
        if model_id is None:
            model_id = len(models)
            models.append(exp.crystal)
        selected = (ids == i) & np.any(hkl != 0, axis=1) & np.all(np.isfinite(q), axis=1)
        grouped.setdefault(model_id, []).append(np.flatnonzero(selected))
    checks = []
    for model_id, crystal in enumerate(models):
        positions = np.concatenate(grouped[model_id])
        A = np.asarray(crystal.get_A(), dtype=float).reshape(3, 3)
        indices = hkl[positions]
        residual = np.linalg.norm(q[positions] - indices @ A.T, axis=1)
        frac_residual = np.linalg.norm(np.linalg.solve(A, q[positions].T).T - indices, axis=1)
        unique_indices = np.unique(indices, axis=0)
        rank = int(np.linalg.matrix_rank(unique_indices - np.mean(unique_indices, axis=0))) if len(indices) else 0
        checks.append({'crystal_model_id': model_id, 'indexed_rows': len(positions),
                       'unique_hkl': len(unique_indices), 'hkl_rank': rank,
                       'three_independent_index_directions': rank == 3,
                       'rlp_residual_median_p95_invA': np.quantile(residual, [.5, .95]).tolist() if len(residual) else None,
                       'fractional_index_residual_median_p95': np.quantile(frac_residual, [.5, .95]).tolist() if len(residual) else None,
                       'cell': list(crystal.get_unit_cell().parameters()), 'A': A.tolist()})
    indexed_total = sum(c['indexed_rows'] for c in checks)
    input_count = None
    if args.candidate_refl:
        candidate = flex.reflection_table.from_file(str(args.candidate_refl))
        input_count = len(candidate)
        if 'scxrd_source_row' not in candidate or 'scxrd_source_row' not in refs:
            raise ValueError('Input/output source_row mapping required for acceptance denominator')
        if not set(source_indices(refs).tolist()) <= set(source_indices(candidate).tolist()):
            raise ValueError('Indexing output contains source rows outside its candidate input')
    save_json(args.output, {'schema': 'scxrd-domain-index-check/1', 'rows_in_indexed_output': len(refs),
              'rows_in_input_candidate': input_count,
              'indexed_fraction_of_input_table': indexed_total / input_count if input_count else None,
              'indexed_fraction_of_output_table': indexed_total / len(refs), 'crystal_models': checks,
              'diagnostic_only': True, 'interpretation': 'rank3 alone does not certify a lattice; inspect residuals, weak-domain coverage and alternative cells',
              'limitations': 'static A; scan-varying models require per-angle calculation; dropped input spots require input table comparison'})


def build_domain_experiments(args):
    original = ExperimentListFactory.from_json_file(str(args.expt), check_format=False)
    models = json.loads(args.models.read_text())
    if models.get('schema') != 'scxrd-accepted-models/1' or not models.get('models'):
        raise ValueError('Require accepted crystal models')
    if models.get('phase_relation') != 'same_phase_supported' or not models.get('common_structure_basis') or not models.get('phase_basis_provenance'):
        raise ValueError('Joint twin predictions require supported same phase and common structural basis; multi-phase models must stay separate')
    args.output.mkdir(parents=True, exist_ok=False)
    expanded = ExperimentList()
    mapping = {}
    for model in models['models']:
        if model.get('accepted') is not True:
            continue
        inverse = np.linalg.inv(np.asarray(model['A'], dtype=float).reshape(3, 3))
        crystal = CrystalFactory.from_dict({'real_space_a': inverse[0].tolist(), 'real_space_b': inverse[1].tolist(),
                                           'real_space_c': inverse[2].tolist(), 'space_group_hall_symbol': ' P 1'})
        for exp in original:
            cloned = copy.deepcopy(exp)
            cloned.crystal = crystal
            cloned.profile = None
            cloned.identifier = f'crystal-{model["model_id"]}-sweep-{len(expanded)}'
            mapping[str(len(expanded))] = int(model['model_id'])
            expanded.append(cloned)
    expanded.as_file(str(args.output / 'multi_domain.expt'))
    save_json(args.output / 'domain_map.json', {'experiment_to_domain': mapping,
              'basis': models['common_structure_basis'], 'phase_basis_provenance': models['phase_basis_provenance'],
              'source_expt': str(args.expt.resolve()), 'source_models': str(args.models.resolve()),
              'next': 'predict all reflections for these models before patch grouping; never use harvested strong spots as the complete prediction list'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('harvest')
    p.add_argument('expt', type=Path)
    p.add_argument('refl', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--d-min', type=float)
    p.add_argument('--d-max', type=float)
    p.add_argument('--min-snr', type=float)
    p.add_argument('--min-span-frames', type=int)
    p.add_argument('--max-span-frames', type=int)
    p.add_argument('--excluded-shells', default='', help='Explicit low:high intervals in Angstrom, comma separated')
    p.set_defaults(func=harvest)
    for action, function in (('summary', model_summary), ('plot', plot)):
        p = sub.add_parser(action)
        p.add_argument('expt', type=Path)
        p.add_argument('refl', type=Path)
        p.add_argument('output', type=Path)
        if action == 'plot':
            p.add_argument('--crystal', type=int, default=0)
            p.add_argument('--layer-half-width', type=float, default=.08)
        p.set_defaults(func=function)
    p = sub.add_parser('scale-audit')
    p.add_argument('refl', type=Path)
    p.add_argument('output', type=Path)
    p.set_defaults(func=scale_audit)
    for action, function in (('q-export', export_q), ('validate-domain', validate_domain)):
        p = sub.add_parser(action)
        p.add_argument('expt', type=Path)
        p.add_argument('refl', type=Path)
        p.add_argument('output', type=Path)
        if action == 'q-export':
            p.add_argument('--with-covariance', action='store_true')
            p.add_argument('--jacobian-step', type=float, default=1e-3)
        else:
            p.add_argument('--candidate-refl', type=Path)
        p.set_defaults(func=function)
    p = sub.add_parser('split-candidates')
    p.add_argument('expt', type=Path)
    p.add_argument('refl', type=Path)
    p.add_argument('domains', type=Path)
    p.add_argument('output', type=Path)
    p.set_defaults(func=split_candidates)
    p = sub.add_parser('build-domain-experiments')
    p.add_argument('expt', type=Path)
    p.add_argument('models', type=Path)
    p.add_argument('output', type=Path)
    p.set_defaults(func=build_domain_experiments)
    args = parser.parse_args()
    try:
        args.func(args)
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print('error: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
