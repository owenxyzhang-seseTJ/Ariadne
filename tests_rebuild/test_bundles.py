import zipfile
import pytest
from ariadne.agent import engine


def run_bundle(tmp_path,contents,**args):
    archive=tmp_path/'data.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for name,data in contents:z.writestr(name,data)
    request=dict(mode='custom',inputs={'bundle':str(archive)},nodes=[
        engine.node('unpack','files.unpack',dict(archive=engine.input_ref('bundle'),**args))])
    root=tmp_path/'session';engine.initialize(request,root);state=engine.run(root)
    return root,state


def test_sidecar_layout_and_selection_are_preserved(tmp_path):
    root,r=run_bundle(tmp_path,[('run/manifest.json','{}'),('run/pixels.npz','pixels'),('images/0001.cbf','frame')],extensions=['.cbf'])
    assert r['nodes'][0]['status']=='succeeded'
    assert (root/'attempts/unpack/001/bundle/run/pixels.npz').is_file()
    from ariadne.agent.store import Store
    store=Store(root);result=store.node('unpack')['result'];store.close()
    assert result['file_count']==3 and len(result['selected'])==1


@pytest.mark.parametrize('name',['../../escape','/tmp/escape','C:/escape','..\\escape'])
def test_zip_path_traversal_is_blocked(tmp_path,name):
    root,r=run_bundle(tmp_path,[(name,'bad')])
    assert r['nodes'][0]['status']=='failed' and 'unsafe' in r['nodes'][0]['error']


def test_zip_resource_budget_is_enforced(tmp_path):
    root,r=run_bundle(tmp_path,[('big','x'*1000)],max_bytes=10)
    assert r['nodes'][0]['status']=='failed' and 'budget' in r['nodes'][0]['error']


def test_zip_case_collision_is_rejected(tmp_path):
    root,r=run_bundle(tmp_path,[('Data','x'),('data','y')])
    assert r['nodes'][0]['status']=='failed' and 'colliding' in r['nodes'][0]['error']
