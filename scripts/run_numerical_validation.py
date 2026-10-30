#!/usr/bin/env python3
"""Run reproducible mathematical regressions or the complete native suite.

Use the installed ariadne-env launcher for --scope native.  No image files are
downloaded and no data-processing job is submitted.  Native tests use synthetic
DIALS experiments and isolated temporary process/contract fixtures.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
NUMERICAL = [
    'test_centroid_em', 'test_composite_scaling', 'test_conditional_gaussian',
    'test_constrained_pixels', 'test_numeric_optimization', 'test_profile_positions',
    'test_lattice_principles', 'test_pixel_principles', 'test_scaling_principles',
    'test_information_modes', 'test_overlap_observability',
    'test_streamed_wls', 'test_export_round2', 'test_variable_projection',
    'test_geometry_round2', 'test_profiles_round2', 'test_likelihood_round2',
]


def version_evidence(package):
    """Keep conda release versions distinct from upstream Python metadata.

    The pinned DIALS 3.30.1 release reports Python distribution metadata of 3.30;
    the matching conda record is the installed binary release evidence.
    """
    try:
        distribution = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        distribution = None
    conda = None
    for path in (Path(sys.prefix) / 'conda-meta').glob(package + '-*.json'):
        record = json.loads(path.read_text())
        if record.get('name') == package:
            conda = record.get('version')
            break
    return {'distribution': distribution, 'conda': conda,
            'resolved': conda or distribution}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=('numerical', 'native'), default='numerical')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/local_validation.json')
    args = parser.parse_args()
    sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'python/src'),
                   str(ROOT / 'skills/ariadne-reduction/scripts')]
    if args.scope == 'native':
        for name in ('dials', 'dxtbx', 'cctbx', 'mcp'):
            importlib.import_module(name)
        version = version_evidence('dials')['resolved']
        if version != '3.30.1':
            raise SystemExit(f'Native patch target is DIALS 3.30.1; found {version}')
    os.environ.setdefault('ARIADNE_PLUGIN_ROOT', str(ROOT))
    with tempfile.TemporaryDirectory(prefix='ariadne_validation_') as temporary:
        if not os.environ.get('SCXRD_RUNTIME_CONFIG'):
            config = Path(temporary) / 'runtime.json'
            config.write_text(json.dumps({'schema': 'ariadne-runtime/1',
                'purpose': 'test fixture for capability serialization',
                'plugin_root': str(ROOT), 'python': sys.executable}))
            os.environ['SCXRD_RUNTIME_CONFIG'] = str(config)
        suite = (unittest.defaultTestLoader.discover(str(ROOT / 'tests')) if args.scope == 'native'
                 else unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in NUMERICAL))
        started = time.perf_counter()
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        environment = {'python': sys.version, 'platform': platform.platform(), 'executable': sys.executable}
        environment['version_evidence'] = {}
        for package in ('numpy', 'scipy', 'dials', 'dxtbx', 'cctbx-base', 'mcp'):
            evidence = version_evidence(package)
            environment[package] = evidence['resolved']
            environment['version_evidence'][package] = evidence
        report = {
            'schema': 'ariadne-first-principles-validation/1',
            'revision': 'deep-optimization-round2-20261029',
            'comparison_baseline': 'first-principles delivery of 2026-10-29',
            'baseline_commit': '797ac22ffa9b81da2dd977e013bea8f114b49eb2',
            'scope': args.scope, 'tests_run': result.testsRun,
            'failures': len(result.failures), 'errors': len(result.errors),
            'skipped': len(result.skipped), 'successful': result.wasSuccessful(),
            'elapsed_seconds': time.perf_counter() - started,
            'environment': environment,
            'failure_details': [{'test': str(test), 'traceback': detail} for test, detail in result.failures + result.errors],
            'skipped_details': [{'test': str(test), 'reason': reason} for test, reason in result.skipped],
            'data_scope': 'synthetic numerical and native-interface regressions; no experimental raw-frame accuracy claim',
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(args.output)
        return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
