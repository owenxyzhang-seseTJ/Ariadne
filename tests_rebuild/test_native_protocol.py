"""Protocol fixtures, NOT executions of installed SHELXT/SHELXL binaries."""
import sys
import time
import pytest
from create_demo import write_model
from ariadne.agent import engine
from ariadne.agent.store import Store
from ariadne.agent.process import run_process,Cancelled
from ariadne.agent.backends import typed_options,DIALS_OPTIONS,_prepare_shelx


INS='TITL protocol fixture\nCELL 0.71073 10 10 10 90 90 90\nLATT -1\nSFAC C\nUNIT 1\nFVAR 1\nC1 1 .1 .2 .3 11 .02\nHKLF 4\nEND\n'


def executable_fixture(path,content):
    path.write_text('#!'+sys.executable+'\n'+content)
    path.chmod(0o755)
    return str(path)


def solve_fixture(tmp_path,missing_pair=False):
    cif=tmp_path/'m.cif';write_model(cif);ciftext=cif.read_text()
    solve=executable_fixture(tmp_path/'shelxt_fixture',
        'from pathlib import Path\n'
        +f"Path('job_a.res').write_text({INS!r})\nPath('job_b.res').write_text({INS!r})\n"
        +"Path('job_a.hkl').write_text('1 0 0 111 3\\n')\n"
        +( "" if missing_pair else "Path('job_b.hkl').write_text('1 0 0 222 4\\n')\n"))
    refine=executable_fixture(tmp_path/'shelxl_fixture',
        'from pathlib import Path\n'
        +"Path('job.res').write_text(Path('job.ins').read_text())\n"
        +f"text={ciftext!r}\n"
        +"r='.04' if '111' in Path('job.hkl').read_text() else '.03'\n"
        +"Path('job.cif').write_text(text+'\\n_refine_ls_R_factor_gt '+r+'\\n')\n")
    ins=tmp_path/'in.ins';ins.write_text(INS)
    hkl=tmp_path/'in.hkl';hkl.write_text('1 0 0 999 9\n')
    request=dict(mode='solve',inputs={'ins':str(ins),'hkl':str(hkl)},executables={'shelxt':solve,'shelxl':refine})
    root=tmp_path/'session';engine.initialize(request,root)
    return root


def test_actual_candidate_pairs_drive_dynamic_refinement(tmp_path):
    root=solve_fixture(tmp_path);result=engine.run(root)
    assert all(x['status']=='succeeded' for x in result['nodes'])
    assert (root/'attempts/refine_00/001/job.hkl').read_text().split()[3]=='111'
    assert (root/'attempts/refine_01/001/job.hkl').read_text().split()[3]=='222'
    store=Store(root);comparison=store.node('compare')['result'];store.close()
    assert comparison['descriptive_lowest_R1']==1
    assert comparison['review_status']=='needs_review'
    assert comparison['common_effective_observation_model_verified'] is False


def test_missing_candidate_hkl_does_not_fallback_to_original(tmp_path):
    root=solve_fixture(tmp_path,missing_pair=True);result=engine.run(root)
    assert 'refine_01' not in [x['id'] for x in result['nodes']]
    store=Store(root);missing=store.node('solve')['result']['candidates'][1];store.close()
    assert not missing['eligible_for_refinement'] and missing['hkl'] is None


def test_zero_exit_without_outputs_is_failed(tmp_path):
    root=solve_fixture(tmp_path)
    exe=tmp_path/'shelxt_fixture';exe.write_text('#!'+sys.executable+'\npass\n');exe.chmod(0o755)
    result=engine.run(root)
    assert result['nodes'][0]['status']=='failed'
    assert 'without' in result['nodes'][0]['error']


def test_native_timeout_is_real_and_bounded(tmp_path):
    started=time.monotonic()
    with pytest.raises(TimeoutError):
        run_process([sys.executable,'-c','import time; time.sleep(20)'],tmp_path,.12,lambda:False)
    assert time.monotonic()-started<4


def test_native_cancellation_is_checked(tmp_path):
    with pytest.raises(Cancelled):
        run_process([sys.executable,'-c','import time; time.sleep(20)'],tmp_path,30,lambda:True)


def test_native_returncode_is_not_scientific_success(tmp_path):
    with pytest.raises(RuntimeError,match='exited 7'):
        run_process([sys.executable,'-c','raise SystemExit(7)'],tmp_path,10,lambda:False)


def test_typed_phil_rejects_output_path_overrides():
    with pytest.raises(ValueError):typed_options({'output.experiments':'outside'},DIALS_OPTIONS['index'])


def test_typed_phil_rejects_bool_as_integer():
    with pytest.raises(ValueError):typed_options({'indexing.multiple_lattice_search.max_lattices':True},DIALS_OPTIONS['index'])


def test_refinement_control_does_not_change_twins_or_restraints(tmp_path):
    source=tmp_path/'m.res';target=tmp_path/'m.ins'
    text=INS.replace('HKLF','TWIN -1 0 0 0 -1 0 0 0 -1\nBASF .25\nDFIX 1.5 C1 C2\nHKLF')
    source.write_text(text);_prepare_shelx(source,target,9);out=target.read_text()
    assert 'TWIN -1' in out and 'BASF .25' in out and 'DFIX 1.5' in out
    assert 'L.S. 9' in out and 'ACTA' in out


def test_external_shelx_include_is_not_followed(tmp_path):
    source=tmp_path/'m.res';source.write_text('+../../secret\n'+INS)
    with pytest.raises(ValueError,match='includes'):_prepare_shelx(source,tmp_path/'m.ins')
