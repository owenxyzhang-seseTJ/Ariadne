#!/usr/bin/env python3
"""Check or apply the delivered mathematical-core update to an Ariadne checkout.

The default performs a read-only compatibility check of the cumulative update
from the original Git baseline. --from-first-delivery selects only the round-2
delta. --apply applies the selected patch; incompatible local edits fail safely.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


BASELINE = '797ac22ffa9b81da2dd977e013bea8f114b49eb2'
ROOT = Path(__file__).resolve().parents[1]


def git(target, *arguments):
    return subprocess.run(['git', '-C', str(target), *arguments],
                          text=True, capture_output=True, check=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', type=Path, help='Ariadne Git working-tree root')
    parser.add_argument('--apply', action='store_true', help='apply after compatibility checks')
    parser.add_argument('--allow-different-base', action='store_true',
                        help='allow another HEAD; git apply --check must still pass')
    selected = parser.add_mutually_exclusive_group()
    selected.add_argument('--from-first-delivery', action='store_true',
                          help='apply only round 2 to the first 2026-10-29 delivery')
    selected.add_argument('--patch', type=Path, help='use an explicit patch instead')
    args = parser.parse_args()
    default_patch = ('ariadne-round2.patch' if args.from_first_delivery
                     else 'ariadne-first-principles.patch')
    target = args.target.resolve()
    patch = (args.patch or ROOT / 'patches' / default_patch).resolve()
    if not patch.is_file():
        parser.error(f'Delivered patch not found: {patch}')
    top = git(target, 'rev-parse', '--show-toplevel')
    if top.returncode or Path(top.stdout.strip()).resolve() != target:
        parser.error('target must be the root of an existing Git working tree')
    current = git(target, 'rev-parse', 'HEAD')
    if current.returncode:
        parser.error(current.stderr.strip())
    if current.stdout.strip() != BASELINE and not args.allow_different_base:
        parser.error('HEAD differs from the recorded baseline. Review your branch and '
                     'use --allow-different-base to request a context-based check.')
    checked = git(target, 'apply', '--check', str(patch))
    if checked.returncode:
        reversed_check = git(target, 'apply', '--reverse', '--check', str(patch))
        if reversed_check.returncode == 0:
            print('The complete update is already present; no changes made.')
            return 0
        print(checked.stderr.strip())
        print('Patch is not compatible with this working tree; no changes made.')
        return 1
    summary = git(target, 'apply', '--stat', str(patch))
    if summary.returncode:
        print(summary.stderr.strip())
        return 1
    print(summary.stdout, end='')
    if not args.apply:
        print('Compatibility check passed. Add --apply to apply this update.')
        return 0
    applied = git(target, 'apply', str(patch))
    if applied.returncode:
        print(applied.stderr.strip())
        return 1
    print('Update applied. Review git diff and reinstall python/ in any copied runtime.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
