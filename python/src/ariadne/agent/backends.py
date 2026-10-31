"""Executable adapters. Command construction is independent of the planner."""
from __future__ import annotations

from pathlib import Path
import re
import shutil
import sys

from .process import NeedsReview, executable, run_process
from .util import artifact, artifact_path, atomic_json, read_json
from ..crystal import audit_cif


DIALS_OUTPUTS = {
    'import': {'experiments': 'imported.expt'},
    'find_spots': {'reflections': 'strong.refl'},
    'index': {'experiments': 'indexed.expt', 'reflections': 'indexed.refl'},
    'refine': {'experiments': 'refined.expt', 'reflections': 'refined.refl'},
    'integrate': {'experiments': 'integrated.expt', 'reflections': 'integrated.refl'},
    'scale': {'experiments': 'scaled.expt', 'reflections': 'scaled.refl'},
    'export': {'hkl': 'compound.hkl', 'ins': 'compound.ins'},
    'predict': {'reflections': 'predicted.refl'},
}
DIALS_OPTIONS = {
    'import': {},
    'find_spots': {'spotfinder.threshold.algorithm': ('dispersion', 'dispersion_extended', 'radial_profile', 'ariadne'),
                   'spotfinder.threshold.ariadne.algorithm': ('matched', 'multiscale'),
                   'spotfinder.threshold.ariadne.matched.calibrate': bool,
                   'spotfinder.threshold.ariadne.matched.false_alarms': float,
                   'spotfinder.threshold.ariadne.matched.peak_threshold': 'optional_positive',
                   'spotfinder.threshold.ariadne.matched.local_cells': 'nonnegative_integer',
                   'spotfinder.threshold.ariadne.matched.region_threshold': 'finite',
                   'spotfinder.threshold.ariadne.matched.contrast': 'nonnegative',
                   'spotfinder.threshold.ariadne.matched.tile': int,
                   'spotfinder.threshold.ariadne.matched.background_rounds': int,
                   'spotfinder.threshold.ariadne.matched.radial_background': bool,
                   'spotfinder.threshold.ariadne.matched.zinger_sigma': 'nonnegative',
                   'spotfinder.threshold.ariadne.matched.component_min_size': int,
                   'spotfinder.threshold.ariadne.gain_adu_per_photon': 'optional_positive',
                   'spotfinder.threshold.ariadne.offset_adu': 'finite',
                   'spotfinder.threshold.ariadne.readout_variance_adu': 'nonnegative',
                   'spotfinder.threshold.ariadne.noise_model': ('poisson', 'poisson_gaussian'),
                   'spotfinder.threshold.ariadne.statistic': ('standardized', 'deviance'),
                   'spotfinder.threshold.ariadne.score_threshold': float,
                   'spotfinder.threshold.ariadne.support_sigma': float,
                   'spotfinder.threshold.ariadne.component_min_size': int,
                   'spotfinder.filter.min_spot_size': int},
    'index': {'indexing.method': ('fft3d', 'fft1d', 'real_space_grid_search'),
              'indexing.basis_vector_combinations.solution_scorer': ('weighted', 'filter', 'ariadne'),
              'indexing.basis_vector_combinations.ariadne.q_sigma': float,
              'indexing.multiple_lattice_search.max_lattices': int, 'indexing.max_cell': float},
    'refine': {'refinement.parameterisation.scan_varying': bool},
    'integrate': {'profile.gaussian_rs.fitting.fit_method': ('reciprocal_space', 'detector_space'),
                  'profile.gaussian_rs.fitting.detector_space.deconvolution': bool},
    'scale': {'model': ('physical', 'array', 'KB'), 'physical.absorption_correction': bool},
    'export': {'intensity': ('scale', 'sum', 'profile'), 'shelx.composition': str},
    'predict': {'d_min': float},
}
ANALYSES = {
    'joint_fit': ('overlap_integrator.py', 1, 'integration',
                  {'estimator': ('signed-wls', 'nnls', 'poisson', 'poisson-gaussian'), 'fit-positions': bool,
                   'confidence-level': float, 'max-components': int}),
    'global_scale': ('global_scale.py', 2, 'scaling',
                     {'optimizer': ('alternating', 'profiled'), 'max-iterations': int, 'max-seconds': float}),
    'export_hklf5': ('export_hklf5.py', 1, 'compound.hkl', {}),
    'learn_profiles': ('learn_profile_library.py', 2, 'profiles', {'min-references': int}),
    'build_overlaps': ('overlap_prepare.py', 3, 'patches', {'position-derivatives': bool, 'max-voxels': int}),
    'soft_assign': ('soft_multi_indexer.py', 5, 'compatibility', {'max-candidates-per-spot': int}),
    'joint_refine': ('joint_geometry_refine.py', 3, 'geometry', {'method': ('anchors',), 'max-iterations': int}),
    'scaling_geometry': ('scaling_geometry.py', 4, 'geometry', {}),
    'overlap_observability': ('overlap_observability.py', 1, 'observability.json', {'budget': int}),
}


def typed_options(options: dict, schema: dict) -> dict:
    if not isinstance(options, dict) or set(options)-set(schema):
        raise ValueError('unknown operation option')
    result = {}
    for key, value in options.items():
        rule = schema[key]
        if isinstance(rule, tuple):
            if value not in rule:
                raise ValueError('invalid option: '+key)
        elif rule is bool:
            if not isinstance(value, bool):
                raise ValueError('require boolean: '+key)
        elif rule is int:
            if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= 10_000_000:
                raise ValueError('require bounded positive integer: '+key)
        elif rule == 'nonnegative_integer':
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10_000_000:
                raise ValueError('require bounded nonnegative integer: '+key)
        elif rule is float or rule == 'optional_positive':
            if value is None and rule == 'optional_positive':
                result[key] = value
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float('inf'):
                raise ValueError('require positive finite number: '+key)
        elif rule in {'finite', 'nonnegative'}:
            import math
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (rule == 'nonnegative' and value < 0):
                raise ValueError('require finite calibration value: '+key)
        elif rule is str:
            if not isinstance(value, str) or re.fullmatch(r'[A-Za-z0-9 ().,+-]{1,200}', value) is None:
                raise ValueError('invalid composition string')
        result[key] = value
    return result


def invoke(ctx, argv, label='process'):
    def pid(value):
        ctx.store.db.execute('UPDATE attempts SET pid=? WHERE node=? AND number=?',
                             (value, ctx.node_id, ctx.store.node(ctx.node_id)['attempts']))
    return run_process(argv, ctx.directory, ctx.remaining(),
                       lambda: ctx.store.get('cancel_requested', False), pid, label)


def dials_command(operation, paths, executables):
    """One process-local raw reader for the agent and expert MCP entrypoints."""
    py = executable('scientific_python', {'scientific_python': executables.get('scientific_python', sys.executable)})
    runner = plugin_root()/'skills/ariadne-reduction/scripts/dials_runtime.py'
    return [py, str(runner), executable('dials.'+operation, executables), *map(str, paths)]


def spotfinding_options(options):
    options = typed_options(options, DIALS_OPTIONS['find_spots'])
    options.setdefault('spotfinder.threshold.algorithm', 'ariadne')
    if options['spotfinder.threshold.algorithm'] == 'ariadne':
        options.setdefault('spotfinder.threshold.ariadne.algorithm', 'matched')
    return options


def run_dials(ctx, args: dict) -> dict:
    operation = args['operation']
    if operation not in DIALS_OUTPUTS:
        raise ValueError('unregistered DIALS operation')
    inputs = args['inputs']
    if not isinstance(inputs, list):
        raise ValueError('inputs must be an artifact list')
    expected_count = {'find_spots': 1, 'predict': 1}.get(operation, 2)
    if operation == 'import':
        if not 1 <= len(inputs) <= 10000:
            raise ValueError('import requires 1 to 10000 files')
    elif len(inputs) != expected_count:
        raise ValueError('wrong number of inputs for '+operation)
    paths = [artifact_path(ctx.root, value) for value in inputs]
    options = typed_options(args.get('options', {}), DIALS_OPTIONS[operation])
    if operation == 'find_spots':
        options = spotfinding_options(options)
    # Every DIALS process must use the same raw decoder, especially integrate.
    # Run the installed dispatcher in its scientific Python without changing
    # site-packages or detector gain to match a threshold-derived noise fit.
    argv = dials_command(operation, paths, ctx.executables)
    outputs = DIALS_OUTPUTS[operation]
    if operation == 'export':
        if 'shelx.composition' not in options:
            raise ValueError('declare chemical composition before SHELX export')
        argv += ['format=shelx', 'shelx.hklout=compound.hkl', 'shelx.ins=compound.ins']
    elif operation == 'predict':
        argv += ['output=predicted.refl']
    else:
        argv += [f'output.{key}={name}' for key, name in outputs.items()]
    argv += [f'{key}={value}' for key, value in options.items()]
    invoke(ctx, argv)
    result = {role: artifact(ctx.root, ctx.directory/name, role) for role, name in outputs.items()}
    if operation in {'index', 'refine', 'integrate', 'scale'}:
        inspection = ctx.directory/'native_inspection.json'
        helper = Path(__file__).with_name('native_inspect.py')
        py = executable('scientific_python', {'scientific_python': ctx.executables.get('scientific_python', sys.executable)})
        invoke(ctx, [py, str(helper), str(ctx.directory/outputs['experiments']),
                     str(ctx.directory/outputs['reflections']), str(inspection)], 'inspection')
        result['inspection'] = read_json(inspection)
        if ctx.request['mode'] == 'reduce' and result['inspection']['n_crystals'] != 1:
            atomic_json(ctx.directory/'needs_review.json', result)
            raise NeedsReview('automatic single-crystal route cannot continue: use the joint overlap workflow')
    result['raw_data_adapter'] = {
        'bruker100': 'signed absolute ADU; baseline only on non-underflow pixels',
        'scope': 'process-local; same decoder for import, find_spots and integrate',
        'pixel_unit': 'raw ADU; DIALS corrected data retain (raw-pedestal)/gain',
        'threshold_calibration': 'selection only; does not modify detector gain or integrated intensities',
    }
    if operation == 'find_spots':
        result['threshold_options'] = options
    result['scientific_acceptance'] = 'not_established'
    return result


def plugin_root() -> Path:
    import os
    if os.environ.get('ARIADNE_PLUGIN_ROOT'):
        root = Path(os.environ['ARIADNE_PLUGIN_ROOT']).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parents[4]
    if not (root/'skills/ariadne-reduction/scripts').is_dir():
        raise RuntimeError('set ARIADNE_PLUGIN_ROOT to the complete source checkout')
    return root


def run_analysis(ctx, args: dict) -> dict:
    name = args['analysis']
    if name not in ANALYSES:
        raise ValueError('analysis is not registered')
    script, count, output, option_schema = ANALYSES[name]
    if not isinstance(args['inputs'], list) or len(args['inputs']) != count:
        raise ValueError('wrong analysis input count')
    paths = [artifact_path(ctx.root, x) for x in args['inputs']]
    options = typed_options(args.get('options', {}), option_schema)
    target = ctx.directory/output
    py = executable('scientific_python', {'scientific_python': ctx.executables.get('scientific_python', sys.executable)})
    scripts = plugin_root()/'skills/ariadne-reduction/scripts'
    argv = [py]
    if name in {'learn_profiles', 'build_overlaps'}:
        argv.append(str(scripts/'dials_runtime.py'))
    argv += [str(scripts/script), *map(str, paths)]
    argv += ['--output', str(target)] if name == 'overlap_observability' else [str(target)]
    for key, value in options.items():
        if isinstance(value, bool):
            if value:
                argv.append('--'+key)
        else:
            argv += ['--'+key, str(value)]
    invoke(ctx, argv)
    files = sorted(target.rglob('*')) if target.is_dir() else [target]
    if name == 'export_hklf5':
        # The export map and covariance-loss statement are part of this result.
        files.append(target.with_suffix('.export.json'))
        if not all(p.is_file() for p in files):
            raise RuntimeError('HKLF5 export must include its observation mapping sidecar')
    result = dict(artifacts=[artifact(ctx.root, p, 'analysis') for p in files if p.is_file()],
                  scientific_acceptance='not_established')
    if not result['artifacts']:
        raise RuntimeError('analysis produced no output files')
    return result


def _prepare_shelx(source: Path, target: Path, cycles: int | None = None):
    lines = source.read_text(errors='strict').splitlines()
    if any(line.lstrip().startswith('+') for line in lines):
        raise ValueError('external SHELX includes must be explicitly staged; automatic route does not follow them')
    if any(line.split() and line.split()[0].upper() == 'ABIN' for line in lines):
        raise NeedsReview('ABIN requires a paired FAB file; use an explicitly staged expert workflow')
    hklf = [line.split() for line in lines if line.split() and line.split()[0].upper() == 'HKLF']
    if not hklf:
        raise ValueError('SHELX input has no HKLF instruction')
    if len(hklf) != 1 or len(hklf[0]) < 2 or hklf[0][1] not in {'4', '5'}:
        raise NeedsReview('automatic structure backend requires one explicit HKLF 4 or 5 instruction')
    if cycles is None and hklf[0][1] != '4':
        raise NeedsReview('SHELXT solving requires HKLF 4; composite HKLF 5 is reserved for model refinement')
    if hklf[0][1] == '5' and any(line.split() and line.split()[0].upper() == 'TWIN' for line in lines):
        raise NeedsReview('HKLF 5 already declares component indices; do not combine it with TWIN')
    if cycles is None:
        shutil.copy2(source, target)
        return
    if isinstance(cycles, bool) or not isinstance(cycles, int) or not 1 <= cycles <= 100:
        raise ValueError('refinement cycles must be 1..100')
    # Do not silently edit restraints, atom types, TWIN/BASF, OMIT, SWAT or EXTI.
    result = []
    inserted = False
    for line in lines:
        token = line.split()[0].upper() if line.split() else ''
        if token in {'L.S.', 'CGLS', 'ACTA'}:
            if line.rstrip().endswith('='):
                raise ValueError('continued refinement control requires manual normalization')
            continue
        if token == 'HKLF' and not inserted:
            result += [f'L.S. {cycles}', 'ACTA']
            inserted = True
        result.append(line)
    target.write_text('\n'.join(result)+'\n')


def solve(ctx, args: dict) -> dict:
    ins, hkl = artifact_path(ctx.root, args['ins']), artifact_path(ctx.root, args['hkl'])
    _prepare_shelx(ins, ctx.directory/'job.ins')
    shutil.copy2(hkl, ctx.directory/'job.hkl')
    invoke(ctx, [executable('shelxt', ctx.executables), 'job'])
    candidates = []
    for res in sorted(ctx.directory.glob('job*.res')):
        partner = res.with_suffix('.hkl')
        paired = partner.is_file()
        candidates.append(dict(name=res.stem, res=artifact(ctx.root, res, 'model'),
                               hkl=artifact(ctx.root, partner, 'observed_intensities') if paired else None,
                               eligible_for_refinement=paired,
                               source_hkl=args['hkl'],
                               reason=None if paired else 'candidate-specific HKL missing; no fallback to source data'))
    if not candidates:
        raise RuntimeError('SHELXT completed without a candidate RES')
    return dict(candidates=candidates, source_hkl=args['hkl'],
                scientific_acceptance='not_established')


def refine(ctx, args: dict) -> dict:
    res, hkl = artifact_path(ctx.root, args['res']), artifact_path(ctx.root, args['hkl'])
    cycles = args.get('cycles', 8)
    _prepare_shelx(res, ctx.directory/'job.ins', cycles)
    shutil.copy2(hkl, ctx.directory/'job.hkl')
    invoke(ctx, [executable('shelxl', ctx.executables), 'job'])
    produced_res, cif = ctx.directory/'job.res', ctx.directory/'job.cif'
    result = dict(res=artifact(ctx.root, produced_res, 'refined_model'),
                  hkl=artifact(ctx.root, ctx.directory/'job.hkl', 'observed_intensities'),
                  input_hkl=args['hkl'], scientific_acceptance='not_established')
    if not cif.is_file():
        raise RuntimeError('SHELXL did not produce the requested CIF')
    result['cif'] = artifact(ctx.root, cif, 'refined_cif')
    result['audit'] = audit_cif(cif, expected_elements=ctx.request['settings'].get('expected_elements'))
    result['metrics'] = result['audit']['metrics']
    result['model_eligible'] = result['audit']['model_eligible']
    return result
