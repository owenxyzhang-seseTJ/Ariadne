import pytest
from create_demo import create
from ariadne.agent import engine
from ariadne.agent.util import read_json,atomic_json
from ariadne.feedback import closed_split


def test_native_synthetic_demo_and_resume(tmp_path):
    result=create(tmp_path/'demo',groups=16)
    assert result['actions']==5 and result['input_unchanged'] and result['resume_did_not_repeat']
    assert result['selected_candidate']=='good'
    assert result['calibrated_fractions'][0]==pytest.approx(.7,abs=.04)


def test_exposed_test_is_not_reused_in_same_project(tmp_path):
    root=tmp_path/'demo';create(root,groups=16)
    request=read_json(root/'request.json')
    engine.initialize(request,root/'second_session',base=root)
    result=engine.run(root/'second_session')
    feedback=next(x for x in result['nodes'] if x['id']=='feedback')
    assert feedback['status']=='failed' and 'already been exposed' in feedback['error']
    with pytest.raises(RuntimeError,match='exposure'):engine.retry(root/'second_session','feedback')


def test_model_fit_on_holdout_is_rejected_before_exposure(tmp_path):
    root=tmp_path/'demo';create(root,groups=16,run=False)
    data=read_json(root/'observations.json')
    split=closed_split(data['blocks'],len(data['reflections']),str(20261030))
    heldout=next(k for k,v in split.items() if v=='test')
    request=read_json(root/'request.json')
    model=request['nodes'][2]['args']['models'][0]
    model['provenance']='training';model['fit_observation_ids']=[heldout]
    engine.initialize(request,root/'session',base=root)
    result=engine.run(root/'session');f=next(x for x in result['nodes'] if x['id']=='feedback')
    assert f['status']=='failed' and 'leaks' in f['error']
    assert not (root/'.ariadne-feedback-ledger.sqlite').exists()


def test_model_assisted_values_are_not_accepted_as_new_measurements(tmp_path):
    root=tmp_path/'demo';create(root,groups=16,run=False)
    data=read_json(root/'observations.json');data['measurement_semantics']='posterior'
    atomic_json(root/'observations.json',data)
    engine.initialize(read_json(root/'request.json'),root/'session',base=root)
    result=engine.run(root/'session');f=next(x for x in result['nodes'] if x['id']=='feedback')
    assert f['status']=='failed' and 'original signed' in f['error']


def test_preprocessing_leakage_is_rejected(tmp_path):
    root=tmp_path/'demo';create(root,groups=16,run=False)
    data=read_json(root/'observations.json')
    data['preprocessing']={'scope':'training','fit_observation_ids':[b['id'] for b in data['blocks']]}
    atomic_json(root/'observations.json',data)
    engine.initialize(read_json(root/'request.json'),root/'session',base=root)
    result=engine.run(root/'session');f=next(x for x in result['nodes'] if x['id']=='feedback')
    assert f['status']=='failed' and 'preprocessing' in f['error']
