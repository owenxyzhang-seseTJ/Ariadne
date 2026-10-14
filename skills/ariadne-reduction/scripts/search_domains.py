#!/usr/bin/env python3
"""Budgeted propose/index/accept/joint-assign/residual-search loop on real data.

This command executes crystallography tools when explicitly invoked. No synthetic
fixtures or automatic validation runs are performed during skill installation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def search(args):
    scripts = Path(__file__).resolve().parent
    python = sys.executable
    indexer = shutil.which('dials.index')
    if not indexer:
        raise ValueError('DIALS environment is not activated')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    records, checks = [], []
    assignments = None
    settings = [(3, 128, 200000, 12000), (6, 256, 400000, 24000), (9, 384, 800000, 48000)]
    if args.heavy:
        settings += [(12, 768, 1600000, 96000), (18, 1024, 2400000, 160000)]
    attempts = 0

    def run(argv, work, expected):
        remaining = args.max_seconds - (time.monotonic() - started)
        if remaining <= 0:
            return False
        command = {'argv': argv, 'cwd': str(work), 'timeout_seconds': min(args.command_timeout, remaining)}
        (work / f'command_{len(records):04d}.json').write_text(json.dumps(command, indent=2))
        try:
            with (work / f'stdout_{len(records):04d}.log').open('wb') as stdout, (work / f'stderr_{len(records):04d}.log').open('wb') as stderr:
                process = subprocess.run(argv, cwd=work, stdout=stdout, stderr=stderr,
                                         timeout=min(args.command_timeout, remaining), check=False)
            good = process.returncode == 0 and all(p.is_file() and p.stat().st_size for p in expected)
            records.append({**command, 'returncode': process.returncode, 'outputs_present': bool(good)})
            return good
        except (OSError, subprocess.TimeoutExpired) as exc:
            records.append({**command, 'error': str(exc)})
            return False

    for round_id, (restarts, centres, pairs, points) in enumerate(settings, 1):
        if time.monotonic() - started >= args.max_seconds or attempts >= args.max_index_attempts:
            break
        work = args.output / f'round_{round_id:02d}'
        work.mkdir()
        candidates = work / 'candidates'
        argv = [python, str(scripts / 'domain_finder.py'), str(args.points.resolve()), str(candidates.resolve()),
                '--method', 'hybrid', '--restarts', str(restarts), '--max-centres', str(centres),
                '--max-pairs', str(pairs), '--max-search-points', str(points), '--max-groups', str(args.max_candidates),
                '--seed', str(args.seed + round_id)]
        if assignments:
            argv += ['--accepted-assignments', str(assignments.resolve())]
        if not run(argv, work, [candidates / 'domains.json']):
            continue
        split = work / 'candidate_refl'
        if not run([python, str(scripts / 'dials_tools.py'), 'split-candidates', str(args.expt.resolve()),
                    str(args.reflections.resolve()), str((candidates / 'domains.json').resolve()), str(split.resolve())],
                   work, [split / 'manifest.json']):
            continue
        manifest = json.loads((split / 'manifest.json').read_text())
        for candidate in manifest['candidates']:
            for method in args.index_methods.split(','):
                if attempts >= args.max_index_attempts or time.monotonic() - started >= args.max_seconds:
                    break
                index_work = work / f'index_{candidate["candidate_id"]:03d}_{method}'
                index_work.mkdir()
                attempts += 1
                argv = [indexer, str(args.expt.resolve()), str((split / candidate['refl']).resolve()),
                        f'method={method}', 'joint_indexing=True', 'max_lattices=1',
                        'refinement.parameterisation.beam.fix=all', 'refinement.parameterisation.detector.fix=all',
                        'refinement.parameterisation.goniometer.fix=all',
                        'output.experiments=indexed.expt', 'output.reflections=indexed.refl']
                if args.max_cell:
                    argv += [f'max_cell={args.max_cell}']
                if args.solution_scorer == 'ariadne':
                    argv += ['indexing.basis_vector_combinations.solution_scorer=ariadne',
                             'indexing.basis_vector_combinations.ariadne.evidence_directory=candidate_evidence',
                             f'indexing.basis_vector_combinations.ariadne.q_sigma={args.candidate_q_sigma}']
                if not run(argv, index_work, [index_work / 'indexed.expt', index_work / 'indexed.refl']):
                    continue
                check_path = index_work / 'index_check.json'
                if run([python, str(scripts / 'dials_tools.py'), 'validate-domain', str((index_work / 'indexed.expt').resolve()),
                        str((index_work / 'indexed.refl').resolve()), str(check_path.resolve()),
                        '--candidate-refl', str((split / candidate['refl']).resolve())], index_work, [check_path]):
                    checks.append(check_path)
        if not checks:
            continue
        accepted = work / 'accepted_models.json'
        if not run([python, str(scripts / 'domain_manager.py'), 'accept-checks', str(accepted.resolve()),
                    *[str(p.resolve()) for p in checks], '--max-p95', str(args.q_tolerance)], work, [accepted]):
            continue
        assignment_dir = work / 'joint_assignment'
        if run([python, str(scripts / 'domain_manager.py'), 'assign', str(args.points.resolve()), str(accepted.resolve()),
                str(assignment_dir.resolve()), '--q-tolerance', str(args.q_tolerance)], work, [assignment_dir / 'assignment.json']):
            assignments = assignment_dir / 'assignment.json'
    report = {'schema': 'scxrd-adaptive-domain-search/1', 'implementation_status': 'experimental_untested',
              'records': records, 'index_attempts': attempts, 'checks': [str(p.resolve()) for p in checks],
              'latest_assignment': str(assignments.resolve()) if assignments else None,
              'elapsed_seconds': time.monotonic() - started, 'heavy_budget_enabled': args.heavy,
              'budgets': {'max_seconds': args.max_seconds, 'max_index_attempts': args.max_index_attempts},
              'acceptance': 'geometry gate only; chemical/phase/twin identification remains evidence-dependent',
              'instrument_geometry': 'held fixed so accepted models share exported q reference frame',
              'unresolved_points': 'retained; exhausting search budget does not establish absence of extra domains'}
    (args.output / 'search.json').write_text(json.dumps(report, indent=2) + '\n')
    print(args.output / 'search.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('points', type=Path)
    p.add_argument('expt', type=Path)
    p.add_argument('reflections', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--heavy', action='store_true')
    p.add_argument('--max-seconds', type=float, default=3600)
    p.add_argument('--command-timeout', type=float, default=600)
    p.add_argument('--max-index-attempts', type=int, default=48)
    p.add_argument('--max-candidates', type=int, default=8)
    p.add_argument('--index-methods', default='fft3d,fft1d')
    p.add_argument('--max-cell', type=float)
    p.add_argument('--solution-scorer', choices=('weighted', 'ariadne'), default='weighted')
    p.add_argument('--candidate-q-sigma', type=float, default=.001)
    p.add_argument('--q-tolerance', type=float, default=.003)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    try:
        if min(args.max_seconds, args.command_timeout, args.max_index_attempts, args.candidate_q_sigma) <= 0:
            raise ValueError('Budgets must be positive')
        if any(m not in ('fft3d', 'fft1d') for m in args.index_methods.split(',')):
            raise ValueError('This unknown-cell search supports fft3d/fft1d; known-cell methods need a separate configured route')
        search(args)
        return 0
    except (ValueError, KeyError, OSError) as exc:
        print('error: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
