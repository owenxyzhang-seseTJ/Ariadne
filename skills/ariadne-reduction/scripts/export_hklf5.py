#!/usr/bin/env python3
"""Format genuine corrected observation groups as HKLF 5; does not integrate pixels."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


from hklf5_stream import export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    try:
        export(args.source, args.output)
        return 0
    except (ValueError, KeyError, OSError, StopIteration) as exc:
        print('error: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
