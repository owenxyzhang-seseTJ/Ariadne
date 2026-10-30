import json
import zipfile
import pytest
from create_demo import write_model
from ariadne.agent import engine
from ariadne.agent.spec import validate_node,validate_request
from ariadne.agent.store import Store
from ariadne.agent.util import contained,pointer


def audit_session(tmp_path,**extra):
    model=tmp_path/'m.cif';write_model(model)
    request={'schema':'ariadne-agent/1','mode':'audit','inputs':{'cif':str(model)},**extra}
    root=tmp_path/'session';engine.initialize(request,root)
    return root


def test_audit_session_runs_and_resumes_without_repeating(tmp_path):
    root=audit_session(tmp_path)
    a=engine.run(root,max_steps=1);assert a['actions_used']==1
    b=engine.run(root);assert b['actions_used']==3
    c=engine.run(root);assert c['actions_used']==3
    assert all(x['status']=='succeeded' for x in c['nodes'])
    store=Store(root);assert store.verify_events();store.close()


def test_input_is_a_real_snapshot_not_a_mutable_source_path(tmp_path):
    root=audit_session(tmp_path);(tmp_path/'m.cif').write_text('changed')
    result=engine.run(root);assert all(x['status']=='succeeded' for x in result['nodes'])


def test_snapshot_tampering_is_refused(tmp_path):
    root=audit_session(tmp_path)
    (root/'inputs/cif/m.cif').write_text('changed')
    with pytest.raises(ValueError,match='changed'):engine.run(root)


def test_output_tampering_is_refused_on_resume(tmp_path):
    root=audit_session(tmp_path);engine.run(root,max_steps=1)
    (root/'attempts/audit/001/result.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):engine.run(root)


def test_sql_result_tampering_is_refused(tmp_path):
    root=audit_session(tmp_path);engine.run(root,max_steps=1)
    s=Store(root);r=s.node('audit')['result'];r['site_count']=5000
    s.db.execute('UPDATE nodes SET result=? WHERE id="audit"',(json.dumps(r),));s.close()
    with pytest.raises(RuntimeError,match='does not match'):engine.run(root)


def test_event_tampering_is_detected(tmp_path):
    root=audit_session(tmp_path);s=Store(root)
    s.db.execute('UPDATE events SET payload=? WHERE seq=1',('{}',));s.close()
    with pytest.raises(RuntimeError,match='integrity'):engine.run(root)


def test_export_is_valid_and_does_not_overwrite(tmp_path):
    root=audit_session(tmp_path);engine.run(root)
    dest=tmp_path/'export.zip';r=engine.export_session(root,dest)
    assert r['size_bytes']>0
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None and 'session.json' in z.namelist()
        assert not any(n.startswith('inputs/') for n in z.namelist())
    with pytest.raises(ValueError):engine.export_session(root,dest)


def test_source_inputs_included_only_on_request(tmp_path):
    root=audit_session(tmp_path);engine.run(root)
    dest=tmp_path/'with_inputs.zip';engine.export_session(root,dest,include_inputs=True)
    with zipfile.ZipFile(dest) as z:assert 'inputs/cif/m.cif' in z.namelist()


def test_action_budget_is_cumulative_across_resume(tmp_path):
    root=audit_session(tmp_path,budget={'max_actions':1})
    assert engine.run(root)['actions_used']==1
    assert engine.run(root)['actions_used']==1


def test_cancel_prevents_work_until_explicit_resume(tmp_path):
    root=audit_session(tmp_path);engine.cancel(root)
    assert engine.run(root)['actions_used']==0
    engine.resume(root);assert engine.run(root)['actions_used']==3


def test_existing_session_is_never_overwritten(tmp_path):
    root=audit_session(tmp_path)
    with pytest.raises(FileExistsError):
        engine.initialize({'mode':'audit','inputs':{'cif':str(tmp_path/'m.cif')}},root)


def test_missing_input_is_rejected_before_session_creation(tmp_path):
    root=tmp_path/'session'
    with pytest.raises(FileNotFoundError):engine.initialize({'mode':'audit','inputs':{'cif':str(tmp_path/'missing')}},root)
    assert not root.exists()


def test_bad_node_reference_cannot_invent_results():
    with pytest.raises(ValueError,match='direct'):
        validate_node(engine.node('x','crystal.audit',{'cif':engine.ref('producer','/cif')}))


def test_review_cannot_accept_a_model_written_success_claim():
    with pytest.raises(ValueError):
        validate_node(engine.node('x','review',{'evidence':[{'scientific_acceptance':'yes'},engine.ref('p')]},['p']))


def test_literal_artifact_claim_cannot_bypass_dependencies():
    with pytest.raises(ValueError,match='literal'):
        validate_node(engine.node('x','crystal.audit',{'cif':{'path':'a','sha256':'fake'}}))


def test_dependency_cycle_rejected(tmp_path):
    model=tmp_path/'m.cif';write_model(model)
    request=dict(mode='custom',inputs={'cif':str(model)},nodes=[
        engine.node('a','crystal.audit',{'cif':engine.input_ref('cif')},['b']),
        engine.node('b','crystal.audit',{'cif':engine.input_ref('cif')},['a'])])
    with pytest.raises(ValueError,match='cycle'):engine.initialize(request,tmp_path/'session')


def test_request_rejects_literal_api_secrets():
    with pytest.raises(ValueError,match='secret'):
        validate_request(dict(mode='audit',inputs={},planner={'kind':'chat-completions','api_key':'secret'}))


@pytest.mark.parametrize('value',['../outside','/etc/passwd','x/../../outside'])
def test_path_escape_is_refused(tmp_path,value):
    with pytest.raises(ValueError):contained(tmp_path,value)


def test_pointer_supports_escaped_dictionary_keys():
    assert pointer({'a/b':{'~x':[5]}},'/a~1b/~0x/0')==5
    with pytest.raises(ValueError):pointer([1,2],'/-1')


def test_unsettled_process_is_not_silently_rerun(tmp_path):
    root=audit_session(tmp_path);store=Store(root);store.begin('audit');store.close()
    with pytest.raises(RuntimeError,match='unsettled'):engine.run(root)
    with pytest.raises(ValueError):engine.recover(root,'audit')
    engine.recover(root,'audit',confirmed_process_stopped=True)
    engine.retry(root,'audit');r=engine.run(root)
    assert r['actions_used']==4


def test_successful_node_cannot_be_retried(tmp_path):
    root=audit_session(tmp_path);engine.run(root,max_steps=1)
    with pytest.raises(ValueError):engine.retry(root,'audit')


def test_concurrent_runner_lock_is_enforced(tmp_path):
    from ariadne.agent.util import exclusive_lock
    root=audit_session(tmp_path)
    with exclusive_lock(root/'.runner.lock'):
        with pytest.raises(RuntimeError,match='another runner'):engine.run(root)


def test_native_raw_route_does_not_guess_composition(tmp_path):
    p=tmp_path/'frame.cbf';p.write_text('not a raw frame')
    with pytest.raises(ValueError,match='composition'):
        engine.initialize(dict(mode='reduce',inputs={'frame':str(p)}),tmp_path/'session')


def test_no_arbitrary_tool_or_shell_action():
    with pytest.raises(ValueError,match='unregistered'):
        validate_node({'id':'bad','tool':'shell','args':{'command':'echo hacked'}})


def test_planner_observations_include_bounded_scientific_diagnostics(tmp_path):
    root=audit_session(tmp_path);r=engine.run(root,max_steps=1)
    assert r['nodes'][0]['scientific_summary']['site_count']==6
    assert r['nodes'][0]['scientific_summary']['model_eligible']


def test_artifact_inspection_requires_a_real_file_reference(tmp_path):
    data=tmp_path/'data.json';data.write_text(json.dumps({'score':[1,2,3]}))
    request=dict(mode='custom',inputs={'data':str(data)},nodes=[
        engine.node('inspect','artifact.inspect',{'artifact':engine.input_ref('data'),'json_pointer':'/score/1','max_chars':20})])
    root=tmp_path/'s';engine.initialize(request,root);engine.run(root)
    store=Store(root);result=store.node('inspect')['result'];store.close()
    assert result['text']=='2' and not result['truncated']
