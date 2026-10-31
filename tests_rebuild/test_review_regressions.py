"""Integration regressions for defects found during the GitHub publication audit."""
import ast
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pytest

from create_demo import create, write_model
from ariadne.agent import engine
from ariadne.agent.backends import ANALYSES, DIALS_OPTIONS, _prepare_shelx, typed_options
from ariadne.agent.loop import measurement_identity
from ariadne.agent.process import run_process
from ariadne.agent.store import Store
from ariadne.agent.util import atomic_json, read_json
from ariadne.feedback import calibrate, ensemble_prior

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('analysis', sorted(ANALYSES))
def test_registered_analysis_matches_its_actual_argument_parser(analysis):
    script, count, _, options = ANALYSES[analysis]
    tree = ast.parse((ROOT/'skills/ariadne-reduction/scripts'/script).read_text())
    declared, positionals = {}, []
    for call in ast.walk(tree):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == 'add_argument':
            flag = ast.literal_eval(call.args[0])
            keywords = {kw.arg: kw.value for kw in call.keywords}
            declared[flag] = keywords
            if not flag.startswith('-'):
                positionals.append(flag)
    assert len(positionals) == count + (0 if analysis == 'overlap_observability' else 1)
    for name, rule in options.items():
        actual = declared['--'+name]
        if isinstance(rule, tuple):
            assert set(rule) <= set(ast.literal_eval(actual['choices']))
        elif rule is bool:
            assert ast.literal_eval(actual['action']) == 'store_true'


def run_analysis(tmp_path, name, files, options=None):
    inputs = {}
    for key, text in files.items():
        path = tmp_path/(key+'.json')
        path.write_text(text)
        inputs[key] = str(path)
    request = dict(mode='custom', inputs=inputs, nodes=[engine.node('analysis', 'analysis.run',
        dict(analysis=name, inputs=[engine.input_ref(k) for k in files], options=options or {}))])
    root = tmp_path/'session'
    engine.initialize(request, root)
    state = engine.run(root)
    assert state['nodes'][0]['status'] == 'succeeded', state
    store = Store(root)
    try:
        return root, store.node('analysis')['result']
    finally:
        store.close()


def test_joint_fit_collects_real_directory_outputs(tmp_path):
    root, result = run_analysis(tmp_path, 'joint_fit', {'manifest': json.dumps({
        'schema': 'scxrd-pixel-patches/1', 'groups': []})})
    paths = {Path(a['path']).name for a in result['artifacts']}
    assert paths == {'integrated_groups.jsonl', 'integration.json'}
    assert all((root/a['path']).is_file() for a in result['artifacts'])


def test_hklf5_action_keeps_observation_to_lines_sidecar(tmp_path):
    metadata = dict(record='metadata', schema='scxrd-observations/1', corrected=True,
        all_contributors_accounted=True, intensity_semantics='measured_group_total',
        phase_relation='same_phase_supported', domain_count=2,
        **{k: 'synthetic protocol case' for k in ('basis','backend','integration_provenance',
            'correction_provenance','excluded_observations_provenance','overlap_model','phase_evidence_provenance')})
    observation = dict(record='observation', observation_id='signed', intensity=-2., sigma=1.,
                       contributors=[dict(domain=1, hkl=[1,2,3]), dict(domain=2, hkl=[2,1,3])])
    root, result = run_analysis(tmp_path, 'export_hklf5', {'observations':
        '\n'.join(json.dumps(x) for x in (metadata, observation))+'\n'})
    assert {Path(a['path']).name for a in result['artifacts']} == {'compound.hkl','compound.export.json'}
    sidecar = next(a for a in result['artifacts'] if a['path'].endswith('.json'))
    assert read_json(root/sidecar['path'])['observation_to_lines'][0]['observation_id'] == 'signed'


def test_reduce_snapshot_preserves_rotation_sequence_paths(tmp_path):
    sources = {}
    for i in range(3):
        path = tmp_path/f'frame_{i+1:04}.cbf'
        path.write_text('layout fixture; not a diffraction image')
        sources[f'image_{i}'] = str(path)
    root = tmp_path/'session'
    engine.initialize(dict(mode='reduce', inputs=sources, settings={'composition':'C2 H6'}), root)
    manifest = read_json(root/'inputs_manifest.json')
    paths = [Path(a['path']) for a in manifest.values()]
    assert len({p.parent for p in paths}) == 1
    assert {p.name for p in paths} == {Path(p).name for p in sources.values()}


@pytest.mark.parametrize('labels', [[0,1,0,1], [1,2,1,2], [2,7,2,7]])
def test_calibration_preserves_external_domain_labels(labels):
    prior = ensemble_prior([[4],[7],[6],[9]])
    blocks = [dict(id=str(i), columns=[i], design=[[1]], values=[v*(2 if i%2 == 0 else 3)],
                   covariance=[[1]]) for i,v in enumerate(prior['mean'])]
    result = calibrate(prior, labels, blocks)
    assert result['domain_labels'] == sorted(set(labels))
    np.testing.assert_allclose(result['coefficients'], [2,3])
    np.testing.assert_allclose(result['mean'], [8,21,12,27])


def test_feedback_metadata_changes_do_not_grant_new_test_exposure(tmp_path):
    root = tmp_path/'demo'
    create(root, groups=16)
    data = read_json(root/'observations.json')
    data['provenance'] = 'new description of the same measurements'
    atomic_json(root/'observations.json', data)
    engine.initialize(read_json(root/'request.json'), root/'second_session', base=root)
    state = engine.run(root/'second_session')
    feedback = next(n for n in state['nodes'] if n['id'] == 'feedback')
    assert feedback['status'] == 'failed' and 'already been exposed' in feedback['error']


def test_feedback_requires_tracked_raw_pixel_identity():
    with pytest.raises(ValueError, match='raw pixel identities'):
        measurement_identity({'blocks':[dict(values=[1], covariance=[[1]])]})


def test_node_definition_tampering_is_detected_before_work(tmp_path):
    model = tmp_path/'model.cif'
    write_model(model)
    root = tmp_path/'session'
    engine.initialize(dict(mode='audit', inputs={'cif':str(model)}), root)
    store = Store(root)
    store.db.execute('UPDATE nodes SET args=? WHERE id=?', ('{}','audit'))
    store.close()
    with pytest.raises(RuntimeError, match='node definition'):
        engine.run(root)
    assert engine.status(root)['actions_used'] == 0


def test_pid_persistence_failure_terminates_child(tmp_path):
    observed = []
    def fail(pid):
        observed.append(pid)
        raise RuntimeError('PID persistence failed')
    with pytest.raises(RuntimeError, match='PID persistence'):
        run_process([sys.executable, '-c', 'import time; time.sleep(30)'], tmp_path, 10,
                    lambda: False, on_pid=fail)
    with pytest.raises(ProcessLookupError):
        os.kill(observed[0], 0)


def test_fast_process_cannot_bypass_log_limit(tmp_path):
    with pytest.raises(RuntimeError, match='logs exceeded'):
        run_process([sys.executable, '-c', "print('x'*10000)"], tmp_path, 10,
                    lambda: False, log_limit_bytes=100)


def test_indented_shelx_include_is_rejected(tmp_path):
    source = tmp_path/'model.ins'
    source.write_text('  +external.ins\nHKLF 4\nEND\n')
    with pytest.raises(ValueError, match='includes'):
        _prepare_shelx(source, tmp_path/'out.ins')


def test_composite_data_are_not_passed_to_shelxt_as_single_domain(tmp_path):
    from ariadne.agent.process import NeedsReview
    source = tmp_path/'model.ins'
    source.write_text('TITL composite case\nHKLF 5\nEND\n')
    with pytest.raises(NeedsReview, match='requires HKLF 4'):
        _prepare_shelx(source, tmp_path/'solve.ins')
    _prepare_shelx(source, tmp_path/'refine.ins', cycles=8)
    assert 'HKLF 5' in (tmp_path/'refine.ins').read_text()


def test_calibrated_detector_options_accept_zero_and_signed_offsets():
    values = {'spotfinder.threshold.ariadne.offset_adu':-12.,
              'spotfinder.threshold.ariadne.readout_variance_adu':0.,
              'spotfinder.threshold.ariadne.gain_adu_per_photon':2.}
    assert typed_options(values, DIALS_OPTIONS['find_spots']) == values
    with pytest.raises(ValueError):
        typed_options({'spotfinder.threshold.ariadne.readout_variance_adu':-1.}, DIALS_OPTIONS['find_spots'])


def test_inprocess_action_cannot_report_success_after_deadline(tmp_path, monkeypatch):
    model = tmp_path/'m.cif'
    write_model(model)
    root = tmp_path/'session'
    engine.initialize(dict(mode='audit', inputs={'cif':str(model)}, budget={'action_seconds':.01}), root)
    original = engine.execute
    def slow(*args):
        time.sleep(.03)
        return original(*args)
    monkeypatch.setattr(engine, 'execute', slow)
    result = engine.run(root)
    assert result['nodes'][0]['status'] == 'failed'
    assert 'time budget' in result['nodes'][0]['error']


def test_planning_time_counts_towards_cumulative_wall_budget(tmp_path, monkeypatch):
    model = tmp_path/'m.cif'
    write_model(model)
    root = tmp_path/'session'
    engine.initialize(dict(mode='audit', inputs={'cif':str(model)}, budget={'wall_seconds':.01}), root)
    def slow(*args):
        time.sleep(.03)
        return {'action':'pause'}
    monkeypatch.setattr(engine, 'choose', slow)
    state = engine.run(root)
    assert state['elapsed_seconds'] >= .03 and state['actions_used'] == 0
    engine.run(root)
    assert engine.status(root)['decisions_used'] == 1
