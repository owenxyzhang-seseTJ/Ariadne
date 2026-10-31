"""Interruptible native Python actions, using the same process budget as backends."""
from __future__ import annotations

from pathlib import Path
import sys

from .actions import Context, execute
from .backends import invoke
from .process import Cancelled, NeedsReview
from .store import Store
from .util import atomic_json, read_json


ISOLATED_ACTIONS = frozenset({'crystal.audit', 'crystal.predict', 'feedback.condition', 'feedback.loop'})


def execute_bounded(ctx: Context, tool: str, args: dict) -> dict:
    if tool not in ISOLATED_ACTIONS:
        return execute(ctx, tool, args)
    payload = ctx.directory/'action_request.json'
    atomic_json(payload, dict(tool=tool, args=args, started=ctx.started))
    invoke(ctx, [sys.executable, '-m', 'ariadne.agent.action_worker', str(ctx.root),
                 ctx.node_id, str(ctx.directory)], 'native_action')
    outcome = read_json(ctx.directory/'action_outcome.json')
    if outcome['status'] == 'blocked':
        raise NeedsReview(outcome['error'])
    if outcome['status'] == 'cancelled':
        raise Cancelled(outcome['error'])
    if outcome['status'] != 'succeeded':
        raise RuntimeError(outcome['error'])
    return outcome['result']


def main():
    root, node_id, directory = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    payload = read_json(directory/'action_request.json')
    if payload['tool'] not in ISOLATED_ACTIONS:
        raise ValueError('action is not registered for native computation')
    store = Store(root)
    try:
        ctx = Context(store, node_id, directory, payload['started'])
        try:
            if ctx.remaining() <= 0:
                raise TimeoutError('time budget exhausted before native action')
            if store.get('cancel_requested'):
                raise Cancelled('cancellation requested before native action')
            outcome = dict(status='succeeded', result=execute(ctx, payload['tool'], payload['args']))
        except NeedsReview as exc:
            outcome = dict(status='blocked', error=str(exc))
        except Cancelled as exc:
            outcome = dict(status='cancelled', error=str(exc))
        except Exception as exc:
            outcome = dict(status='failed', error=f'{type(exc).__name__}: {exc}')
        atomic_json(directory/'action_outcome.json', outcome)
    finally:
        store.close()


if __name__ == '__main__':
    main()
