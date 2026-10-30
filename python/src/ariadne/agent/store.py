"""Transactional session state. File content, not a success string, is the evidence."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

from .spec import validate_node
from .util import dumps, new_id, now


class Store:
    def __init__(self, root: Path, create: bool = False):
        self.root = root.resolve()
        database = self.root/'agent.sqlite'
        if not create and not database.is_file():
            raise FileNotFoundError('not an initialized agent session')
        self.db = sqlite3.connect(database, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA foreign_keys=ON')
        if create:
            self.db.executescript('''
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE nodes(id TEXT PRIMARY KEY, tool TEXT NOT NULL, args TEXT NOT NULL,
                dependencies TEXT NOT NULL, allow_failed INTEGER NOT NULL, status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0, result TEXT, error TEXT, elapsed REAL NOT NULL DEFAULT 0,
                expanded INTEGER NOT NULL DEFAULT 0, summary TEXT NOT NULL DEFAULT '{}', created TEXT NOT NULL, updated TEXT NOT NULL);
            CREATE TABLE events(seq INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL,
                kind TEXT NOT NULL, payload TEXT NOT NULL, previous_seq INTEGER NOT NULL, event_id TEXT NOT NULL);
            CREATE TABLE attempts(node TEXT NOT NULL, number INTEGER NOT NULL, started TEXT NOT NULL,
                finished TEXT, status TEXT NOT NULL, directory TEXT NOT NULL, pid INTEGER,
                PRIMARY KEY(node,number), FOREIGN KEY(node) REFERENCES nodes(id));
            ''')
            self.set('schema', 'ariadne-agent-state/2')
            self.set('cancel_requested', False)
            self.set('actions_used', 0)
            self.set('decisions_used', 0)
        if self.get('schema') != 'ariadne-agent-state/2':
            raise ValueError('unsupported state schema')

    def close(self):
        self.db.close()

    def get(self, key: str, default=None):
        row = self.db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key: str, value: Any):
        self.db.execute('INSERT INTO metadata(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                        (key, dumps(value)))

    def event(self, kind: str, payload: Any):
        owned = not self.db.in_transaction
        if owned:
            self.db.execute('BEGIN IMMEDIATE')
        try:
            previous = self.db.execute('SELECT seq FROM events ORDER BY seq DESC LIMIT 1').fetchone()
            previous_seq = previous[0] if previous else 0
            at = now()
            self.db.execute('INSERT INTO events(at,kind,payload,previous_seq,event_id) VALUES(?,?,?,?,?)',
                            (at, kind, dumps(payload), previous_seq, new_id()))
            if owned:
                self.db.execute('COMMIT')
        except BaseException:
            if owned:
                self.db.execute('ROLLBACK')
            raise

    def verify_events(self) -> bool:
        required = {
            'session_initialized': {'build', 'inputs'},
            'node_added': {'id', 'tool', 'args', 'depends_on', 'allow_failed_dependencies'},
            'attempt_started': {'node', 'attempt', 'directory'},
            'attempt_finished': {'node', 'status', 'elapsed', 'error'},
            'final_test_exposed': {'dataset', 'frozen'},
        }
        previous_seq = 0
        seen = set()
        for row in self.db.execute('SELECT * FROM events ORDER BY seq'):
            payload = json.loads(row['payload'])
            if not isinstance(payload, dict) or not required.get(row['kind'], set()) <= payload.keys():
                return False
            if (row['seq'] <= previous_seq or row['previous_seq'] != previous_seq
                    or not row['event_id'] or row['event_id'] in seen):
                return False
            seen.add(row['event_id'])
            previous_seq = row['seq']
        return True

    def add_nodes(self, nodes: list[dict[str, Any]]):
        nodes = [validate_node(n) for n in nodes]
        existing = {r['id']: json.loads(r['dependencies']) for r in self.db.execute('SELECT id,dependencies FROM nodes')}
        if len({n['id'] for n in nodes}) != len(nodes) or any(n['id'] in existing for n in nodes):
            raise ValueError('duplicate node id')
        graph = {**existing, **{n['id']: n['depends_on'] for n in nodes}}
        if any(d not in graph for deps in graph.values() for d in deps):
            raise ValueError('unknown dependency')
        done, active = set(), set()
        def visit(key):
            if key in active:
                raise ValueError('cycle in action graph')
            if key in done:
                return
            active.add(key)
            for d in graph[key]:
                visit(d)
            active.remove(key)
            done.add(key)
        for key in graph:
            visit(key)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            for n in nodes:
                self.db.execute('INSERT INTO nodes(id,tool,args,dependencies,allow_failed,status,created,updated) VALUES(?,?,?,?,?,?,?,?)',
                    (n['id'], n['tool'], dumps(n['args']), dumps(n['depends_on']), int(n['allow_failed_dependencies']), 'pending', now(), now()))
                self.event('node_added', n)
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def node(self, key: str) -> dict[str, Any]:
        row = self.db.execute('SELECT * FROM nodes WHERE id=?', (key,)).fetchone()
        if not row:
            raise ValueError('unknown node: '+key)
        out = dict(row)
        for field in ('args', 'dependencies', 'result'):
            if out[field] is not None:
                out[field] = json.loads(out[field])
        return out

    def summaries(self) -> list[dict[str, Any]]:
        # Deliberately do not load large JSON results into every planner observation.
        result = []
        for row in self.db.execute('SELECT id,tool,status,attempts,elapsed,substr(error,1,1000) AS error,summary FROM nodes ORDER BY rowid'):
            item = dict(row)
            item['scientific_summary'] = json.loads(item.pop('summary'))
            result.append(item)
        return result

    def ready(self) -> list[str]:
        statuses = {r['id']: r['status'] for r in self.db.execute('SELECT id,status FROM nodes')}
        ready = []
        for row in self.db.execute('SELECT id,dependencies,allow_failed FROM nodes WHERE status="pending" ORDER BY rowid'):
            deps = json.loads(row['dependencies'])
            wanted = {'succeeded', 'failed', 'cancelled', 'blocked', 'interrupted'} if row['allow_failed'] else {'succeeded'}
            if all(statuses[d] in wanted for d in deps):
                ready.append(row['id'])
        return ready

    def begin(self, key: str) -> Path:
        self.db.execute('BEGIN IMMEDIATE')
        try:
            node = self.node(key)
            if key not in self.ready():
                raise ValueError('node is not ready')
            budget = self.get('request')['budget']
            if node['attempts'] >= budget['max_attempts'] or self.get('actions_used') >= budget['max_actions']:
                raise RuntimeError('action/attempt budget exhausted')
            number = node['attempts']+1
            relative = f'attempts/{key}/{number:03d}'
            directory = self.root/relative
            directory.mkdir(parents=True, exist_ok=False)
            self.db.execute('UPDATE nodes SET status="running",attempts=?,updated=? WHERE id=?', (number, now(), key))
            self.db.execute('INSERT INTO attempts(node,number,started,status,directory) VALUES(?,?,?,?,?)',
                            (key, number, now(), 'running', relative))
            self.set('actions_used', self.get('actions_used')+1)
            self.event('attempt_started', dict(node=key, attempt=number, directory=relative))
            self.db.execute('COMMIT')
            return directory
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def finish(self, key: str, status: str, elapsed: float, result=None, error=None):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            node = self.node(key)
            if node['status'] != 'running':
                raise ValueError('only a running attempt can finish')
            self.db.execute('UPDATE nodes SET status=?,result=?,error=?,elapsed=elapsed+?,updated=?,summary=? WHERE id=?',
                            (status, dumps(result) if result is not None else None, error, elapsed, now(), dumps(scientific_summary(result)), key))
            self.db.execute('UPDATE attempts SET status=?,finished=? WHERE node=? AND number=?',
                            (status, now(), key, node['attempts']))
            self.event('attempt_finished', dict(node=key, status=status, elapsed=elapsed, error=error))
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def elapsed(self) -> float:
        return self.db.execute('SELECT COALESCE(SUM(elapsed),0) FROM nodes').fetchone()[0] + self.get('planning_seconds', 0.)

    def snapshot(self) -> dict[str, Any]:
        return dict(schema='ariadne-agent-observation/1', nodes=self.summaries(), ready=self.ready(),
                    actions_used=self.get('actions_used'), decisions_used=self.get('decisions_used'),
                    elapsed_seconds=self.elapsed(), cancel_requested=self.get('cancel_requested'),
                    scientific_acceptance='not_established')


def scientific_summary(result):
    if not isinstance(result, dict):
        return {}
    summary = {}
    for key in ('model_eligible', 'site_count', 'spacegroup', 'data_rank', 'num_latents', 'num_observed',
                'predictive_nll', 'selected_candidate', 'artifact_count', 'scientific_acceptance', 'review_status'):
        if key in result and isinstance(result[key], (str, int, float, bool, type(None))):
            summary[key] = result[key]
    for key in ('metrics', 'inspection', 'final_test'):
        if isinstance(result.get(key), dict):
            summary[key] = {k: v for k, v in result[key].items()
                            if isinstance(v, (int, float, bool, type(None)))}
    if isinstance(result.get('audit'), dict):
        summary['audit'] = scientific_summary(result['audit'])
    for key in ('candidates', 'eligible_candidates', 'failures', 'errors', 'warnings'):
        if isinstance(result.get(key), list):
            summary[key+'_count'] = len(result[key])
    if 'text' in result:
        summary['text_excerpt'] = str(result['text'])[:1600]
    return summary
