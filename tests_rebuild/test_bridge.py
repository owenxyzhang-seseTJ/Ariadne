import json
import pytest
from ariadne.feedback_bridge import convert


def fixture(tmp_path):
    p=tmp_path/'integrated.jsonl'
    record=dict(observation_id='a',basis='photons',corrected=False,estimator='signed-wls',
                intensity_definition='full-profile amplitude; no further partiality division',status='unresolved',
                contributors=[dict(domain=0,hkl=[1,0,0]),dict(domain=0,hkl=[1,0,0])],
                information_modes=dict(schema='ariadne-information-modes/1',design=[[1,2]],y=[-1],covariance=[[3]]))
    p.write_text(json.dumps(record)+'\n')
    meta=dict(latent_id='h',domain=0,observed_hkl=[1,0,0],model_hkl=[1,0,0],forward_multiplier=2.)
    mapping=dict(schema='ariadne-feedback-map/1',basis='photons',provenance='known physical scale',
                 preprocessing={'scope':'fixed_physical'},groups={'a':dict(raw_ids=['pixels_a'],contributors=[meta,meta])})
    return p,record,mapping


def test_bridge_combines_duplicate_latent_columns(tmp_path):
    p,r,m=fixture(tmp_path);result=convert(p,m)
    assert result['blocks'][0]['design']==[[6.]]
    assert result['blocks'][0]['values']==[-1] and len(result['reflections'])==1


def test_bridge_rejects_model_conditioned_data(tmp_path):
    p,r,m=fixture(tmp_path);r['corrected']=True;p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='original'):convert(p,m)


def test_bridge_rejects_untracked_pixel_identity(tmp_path):
    p,r,m=fixture(tmp_path);m['groups']['a']['raw_ids']=[]
    with pytest.raises(ValueError,match='raw pixel'):convert(p,m)


def test_bridge_requires_known_partiality_convention(tmp_path):
    p,r,m=fixture(tmp_path);r['intensity_definition']='unknown';p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='partiality'):convert(p,m)


def test_bridge_requires_exact_ordered_domain_index_identity(tmp_path):
    p,r,m=fixture(tmp_path);r['contributors'][0]['hkl']=[2,0,0];p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='mismatch'):convert(p,m)
