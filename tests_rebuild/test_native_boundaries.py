"""Deployment trust and real interruption of a native calculation worker."""
import json
import sys
import threading
import time

import pytest

from create_demo import write_model
from ariadne.agent import action_worker, engine
from ariadne.agent.backends import invoke
from ariadne.agent.process import deployment_executables
from ariadne.agent.store import Store


def test_task_cannot_select_an_executable(tmp_path):
    root = tmp_path/'session'
    with pytest.raises(ValueError, match='task executable overrides'):
        engine.initialize(dict(mode='audit', executables={'dials.import':'/bin/sh'}), root)
    assert not root.exists()


def test_only_operator_environment_selects_programs(tmp_path, monkeypatch):
    monkeypatch.delenv('ARIADNE_EXECUTABLES_CONFIG', raising=False)
    assert deployment_executables() == {}
    config = tmp_path/'deployment.json'
    config.write_text(json.dumps({'shelxl':sys.executable}))
    monkeypatch.setenv('ARIADNE_EXECUTABLES_CONFIG', str(config))
    assert deployment_executables() == {'shelxl':sys.executable}
    config.write_text(json.dumps({'shell':'/bin/sh'}))
    with pytest.raises(ValueError, match='deployment executable'):
        deployment_executables()


@pytest.mark.parametrize('cancel', [False, True])
def test_native_worker_is_stopped_before_computation_returns(tmp_path, monkeypatch, cancel):
    model = tmp_path/'model.cif'
    write_model(model)
    root = tmp_path/'session'
    engine.initialize(dict(mode='audit', inputs={'cif':str(model)},
                           budget={'action_seconds':10 if cancel else .2}), root)
    marker = tmp_path/'should_not_finish'

    def slow_worker(ctx, argv, label):
        # Exercise the production process supervisor, not a synchronous sleep
        # in the runner. A successful late computation would leave the marker.
        code = f'import time;from pathlib import Path;time.sleep(20);Path({str(marker)!r}).touch()'
        return invoke(ctx, [sys.executable, '-c', code], label)

    monkeypatch.setattr(action_worker, 'invoke', slow_worker)
    timer = threading.Timer(.2, engine.cancel, args=(root,)) if cancel else None
    started = time.monotonic()
    if timer:
        timer.start()
    try:
        state = engine.run(root, max_steps=1)
    finally:
        if timer:
            timer.join()
    assert time.monotonic()-started < 4
    assert state['nodes'][0]['status'] == ('cancelled' if cancel else 'failed')
    assert not marker.exists()
    store = Store(root)
    try:
        pid = store.db.execute('SELECT pid FROM attempts').fetchone()[0]
        assert pid
        from ariadne.server import pid_alive
        assert not pid_alive(pid, group=True)
    finally:
        store.close()
