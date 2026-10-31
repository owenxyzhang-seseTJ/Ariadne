"""Command-line interface; run `ariadne-agent --help`."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from . import engine
from .store import Store
from .util import atomic_json, dumps, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description='Ariadne 0.7.0: persistent crystallography agent')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('capabilities')
    init = sub.add_parser('init')
    init.add_argument('request', type=Path)
    init.add_argument('session', type=Path)
    for cmd in ('run', 'status', 'cancel', 'resume'):
        item = sub.add_parser(cmd)
        item.add_argument('session', type=Path)
        if cmd == 'run':
            item.add_argument('--steps', type=int)
    for cmd in ('retry', 'recover', 'result'):
        item = sub.add_parser(cmd)
        item.add_argument('session', type=Path)
        item.add_argument('node')
        if cmd == 'recover':
            item.add_argument('--confirmed-process-stopped', action='store_true')
    export = sub.add_parser('export')
    export.add_argument('session', type=Path)
    export.add_argument('destination', type=Path)
    export.add_argument('--include-inputs', action='store_true')
    bridge = sub.add_parser('bridge')
    bridge.add_argument('integrated', type=Path)
    bridge.add_argument('mapping', type=Path)
    bridge.add_argument('output', type=Path)
    image_request = sub.add_parser('request-from-images')
    image_request.add_argument('directory', type=Path)
    image_request.add_argument('output', type=Path)
    image_request.add_argument('--glob', default='*.cbf')
    image_request.add_argument('--composition', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'capabilities':
            result = engine.capabilities()
        elif args.command == 'init':
            result = engine.initialize(read_json(args.request), args.session, base=args.request.resolve().parent)
        elif args.command == 'run':
            result = engine.run(args.session, max_steps=args.steps)
        elif args.command == 'status':
            result = engine.status(args.session)
        elif args.command == 'cancel':
            result = engine.cancel(args.session)
        elif args.command == 'resume':
            result = engine.resume(args.session)
        elif args.command == 'retry':
            result = engine.retry(args.session, args.node)
        elif args.command == 'recover':
            result = engine.recover(args.session, args.node, confirmed_process_stopped=args.confirmed_process_stopped)
        elif args.command == 'result':
            store = Store(args.session)
            try:
                result = store.node(args.node)
            finally:
                store.close()
        elif args.command == 'export':
            result = engine.export_session(args.session, args.destination, include_inputs=args.include_inputs)
        elif args.command == 'bridge':
            from ..feedback_bridge import convert
            if args.output.exists():
                raise FileExistsError('output must be new')
            result = convert(args.integrated, read_json(args.mapping))
            atomic_json(args.output, result)
            result = {'output': str(args.output), 'blocks': len(result['blocks'])}
        else:
            images = sorted(p.resolve() for p in args.directory.glob(args.glob) if p.is_file())
            if not images or len(images) > 10000 or args.output.exists():
                raise ValueError('require 1..10000 images and a new request path')
            result = dict(schema='ariadne-agent/1', mode='reduce', inputs={f'image_{i:06d}': str(p) for i, p in enumerate(images)},
                          settings={'composition': args.composition})
            atomic_json(args.output, result)
            result = {'output': str(args.output), 'images': len(images)}
        print(dumps(result))
        if args.command == 'run' and any(n['status'] in {'failed', 'blocked', 'interrupted', 'cancelled'} for n in result['nodes']):
            return 1
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(dumps({'error': type(exc).__name__, 'message': str(exc)}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
