#!/usr/bin/env python3
"""Bounded real-data validation of production reduction adapters; no input guessing.

Run in the patched DIALS environment, e.g. python -I this_file.py --help.
Raw source data are copied to a temporary input tree and removed at completion;
real command logs and outputs remain under --output. No structure is accepted.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python/src'), str(ROOT/'skills/ariadne-reduction/scripts')]
# Isolated Python intentionally ignores PYTHONPATH; only an explicit operator
# overlay is allowed here, and is inherited by the production subprocesses.
if os.environ.get('ARIADNE_VALIDATION_OVERLAY'):
    sys.path.insert(0, os.environ['ARIADNE_VALIDATION_OVERLAY'])


def write_reusable_experiments(output, source_expt):
    """Keep command outputs immutable; create an explicitly relinked EXPT copy."""
    from ariadne.agent.util import atomic_json
    source = json.loads(source_expt.read_text())
    # Validation imports the first imageset's consecutive source filenames.
    template = source['imageset'][source['experiment'][0]['imageset']]['template']
    reusable = output/'reusable'
    reusable.mkdir(exist_ok=True)
    manifest = []
    for path in sorted((output/'attempts').rglob('*.expt')):
        if not any(part in {'import', 'index', 'refine', 'integrate'} for part in path.relative_to(output/'attempts').parts):
            continue
        document = json.loads(path.read_text())
        changed = False
        for imageset in document.get('imageset', []):
            if 'input_frames' in imageset.get('template', ''):
                imageset['template'] = template
                changed = True
        if changed:
            name = path.parent.parent.name+'_'+path.name
            target = reusable/name
            atomic_json(target, document)
            manifest.append(dict(executed_expt=str(path.relative_to(output)), reusable_expt=str(target.relative_to(output)),
                                 original_source_template=template,
                                 semantics='Post-run path relink only; geometry, raw ADU and reflection mapping unchanged'))
    atomic_json(reusable/'provenance.json', manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-expt', required=True, type=Path)
    p.add_argument('--source-refl', required=True, type=Path)
    p.add_argument('--composition-cif', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--frames', type=int, default=32)
    args = p.parse_args()
    if not 8 <= args.frames <= 64:
        p.error('frames must be in 8..64')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['ARIADNE_PLUGIN_ROOT'] = str(ROOT)
    # This is a test-owned deployment config, never a task executable override.
    deployment = args.output/'executables.json'
    deployment.write_text(json.dumps({'scientific_python': sys.executable}))
    os.environ['ARIADNE_EXECUTABLES_CONFIG'] = str(deployment)
    from ariadne.agent.backends import run_dials, solve
    from ariadne.agent.actions import Context
    from ariadne.agent.store import Store
    from ariadne.agent.util import artifact, atomic_json
    from dials_runtime import install_bruker_adapter
    install_bruker_adapter()
    from ariadne.threshold_adapter import corrected_to_adu
    from dxtbx.model.experiment_list import ExperimentListFactory, ExperimentList
    from dials.array_family import flex
    import numpy as np
    import fabio
    import gemmi
    import ariadne.spots as spots

    report = dict(source_expt=str(args.source_expt), source_refl=str(args.source_refl),
                  composition_source=str(args.composition_cif), frames=args.frames,
                  scientific_acceptance='not_established', stages={},
                  overlay=os.environ.get('ARIADNE_VALIDATION_OVERLAY'),
                  overlay_semantics='Patched DIALS Python overlay and project entry_points metadata; equivalent to normal pip target registration, no installed source changed')
    store = Store(args.output, create=True)
    store.set('request', dict(mode='custom', settings={}, budget=dict(action_seconds=180., wall_seconds=900., max_actions=20, max_attempts=1)))
    expts = ExperimentListFactory.from_json_file(str(args.source_expt), check_format=True)
    imageset = expts[0].imageset
    raw = imageset.get_raw_data(0)[0].as_numpy_array()
    corrected = imageset.get_corrected_data(0)[0].as_numpy_array()
    restored, gain = corrected_to_adu(corrected, imageset)
    official = fabio.open(imageset.get_path(0)).data
    np.testing.assert_array_equal(raw, official)
    np.testing.assert_allclose(restored, raw, rtol=3e-16, atol=1e-12)
    mask = imageset.get_mask(0)[0].as_numpy_array()
    settings = dict(gain=gain, offset=64., readout_variance=9., calibrate=False,
                    peak_threshold=5., local_cells=0)
    first = spots.propose_matched(raw, mask, **settings)
    second = spots.propose_matched(restored, mask, **settings)
    np.testing.assert_array_equal(first, second)
    report['raw_contract'] = dict(shape=list(raw.shape), dtype=str(raw.dtype), min=float(raw.min()),
        panel_gain=gain, pedestal=imageset.get_detector()[0].get_pedestal(), fabio_pixelwise_equal=True,
        raw_vs_restored_matched_mask_equal=True, calibrate=False, offset_adu=64., readout_variance_adu=9.,
        saved_trusted_range=list(imageset.get_detector()[0].get_trusted_range()),
        note='Explicit offset/readout exercise units; not an experimental calibration claim')

    results = {}
    def action(name, operation, inputs, options=None):
        store.add_nodes([dict(id=name, tool='dials.run', args=dict(operation=operation, inputs=[]), depends_on=[])])
        work = store.begin(name)
        started = time.monotonic()
        ctx = Context(store, name, work, started)
        try:
            result = (solve(ctx, dict(ins=inputs[0], hkl=inputs[1])) if operation == 'solve'
                      else run_dials(ctx, dict(operation=operation, inputs=inputs, options=options or {})))
            store.finish(name, 'succeeded', time.monotonic()-started, result=result)
            results[name] = result
            report['stages'][name] = dict(status='succeeded', seconds=time.monotonic()-started, result=result)
            print(name, 'succeeded', flush=True)
            return result
        except Exception as exc:
            store.finish(name, 'failed', time.monotonic()-started, error=str(exc))
            report['stages'][name] = dict(status='failed', seconds=time.monotonic()-started, error=str(exc))
            print(name, 'failed:', str(exc), flush=True)
            return None
        finally:
            atomic_json(args.output/'reduction_path.json', report)

    try:
        with tempfile.TemporaryDirectory(prefix='ariadne-raw-contract-') as temporary:
            # artifact references require files beneath the session root. The
            # temporary input directory is removed after this bounded run.
            staged = args.output/'input_frames'
            staged.mkdir()
            try:
                frames = []
                for i in range(args.frames):
                    source = Path(imageset.get_path(i))
                    target = staged/source.name
                    shutil.copy2(source, target)
                    frames.append(artifact(args.output, target, 'raw_frame'))
                imported = action('import', 'import', frames)
                strong = action('find_spots', 'find_spots', [imported['experiments']]) if imported else None
                indexed = action('index', 'index', [imported['experiments'], strong['reflections']]) if strong else None
                refined = action('refine', 'refine', [indexed['experiments'], indexed['reflections']]) if indexed else None
                integrated = action('integrate', 'integrate', [refined['experiments'], refined['reflections']]) if refined else None
                report['bounded_raw_pipeline_reached'] = 'integrate' if integrated else ('refine' if refined else 'index' if indexed else 'find_spots' if strong else 'import' if imported else 'none')
                if imported:
                    new_expts = ExperimentListFactory.from_json_file(str(args.output/imported['experiments']['path']), check_format=True)
                    report['imported_trusted_range'] = list(new_expts[0].detector[0].get_trusted_range())
                    new_raw = new_expts[0].imageset.get_raw_data(0)[0].as_numpy_array()
                    np.testing.assert_array_equal(new_raw, raw)
                    report['imported_raw_equals_source_absolute_adu'] = True
            finally:
                report['reusable_experiments'] = write_reusable_experiments(args.output, args.source_expt)
                shutil.rmtree(staged)
        # Multi-domain source data must be separated for HKLF4 solution input.
        # Select exactly one original crystal object and preserve measured signed
        # intensities and metadata; do not pretend composite data are HKLF4.
        refs = flex.reflection_table.from_file(str(args.source_refl))
        ids = [i for i, e in enumerate(expts) if e.crystal is expts[0].crystal]
        selected = flex.bool([int(i) in ids for i in refs['id']])
        subset = refs.select(selected)
        mapping = {old: new for new, old in enumerate(ids)}
        subset['id'] = flex.int([mapping[int(i)] for i in subset['id']])
        for key in list(subset.experiment_identifiers().keys()):
            del subset.experiment_identifiers()[key]
        chosen = ExperimentList([expts[i] for i in ids])
        for i, exp in enumerate(chosen):
            subset.experiment_identifiers()[i] = exp.identifier
        source_dir = args.output/'existing_integrated'
        source_dir.mkdir()
        chosen.as_file(str(source_dir/'single_crystal.expt'))
        subset.as_file(str(source_dir/'single_crystal.refl'))
        block = gemmi.cif.read_file(str(args.composition_cif)).sole_block()
        composition = gemmi.cif.as_string(block.find_value('_chemical_formula_sum'))
        if not composition or composition in {'?', '.'}:
            raise ValueError('No composition in source CIF')
        report['existing_integrated'] = dict(selected_experiments=ids, rows=len(subset),
            negative_sum_intensities=int((subset['intensity.sum.value'] < 0).count(True)),
            composition=composition, caveat='Original domain integration retained; overlap/absorption acceptance not established')
        exported = action('export_existing', 'export', [artifact(args.output, source_dir/'single_crystal.expt', 'experiments'),
                            artifact(args.output, source_dir/'single_crystal.refl', 'reflections')],
                          {'intensity': 'sum', 'shelx.composition': composition})
        if exported:
            hkl = args.output/exported['hkl']['path']
            values = [float(line[12:20]) for line in hkl.read_text().splitlines() if len(line) >= 28 and line[:12].strip() != '0   0   0']
            report['shelx_hkl_negative_rows'] = sum(v < 0 for v in values)
            action('solve_existing', 'solve', [exported['ins'], exported['hkl']])
        report['not_covered'] = ['Full multi-sweep raw reduction', 'Raw-route scaling and full-data structure solution',
            'Multi-domain overlap/absorption scientific acceptance', 'Experimental calibration of supplied threshold offset/readout']
    finally:
        atomic_json(args.output/'reduction_path.json', report)
        store.close()
    print(args.output/'reduction_path.json', flush=True)


if __name__ == '__main__':
    main()
