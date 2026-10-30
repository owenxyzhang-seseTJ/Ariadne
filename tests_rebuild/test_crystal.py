import numpy as np
import pytest
import gemmi
from create_demo import write_model
from ariadne.crystal import audit_cif,predict,transform_hkl,load_cif


def test_real_gemmi_model_audit_and_prediction(tmp_path):
    p=tmp_path/'model.cif';write_model(p)
    result=audit_cif(p)
    assert result['model_eligible'] and result['site_count']==6 and result['backend']=='gemmi'
    y=predict([p],[dict(id='a',hkl=[1,2,3]),dict(id='b',hkl=[-1,-2,-3])])
    assert y['predictions'][0][0]>0
    assert y['predictions'][0][0]==pytest.approx(y['predictions'][1][0],rel=1e-12)


def test_wrong_symmetry_operations_rejected(tmp_path):
    p=tmp_path/'m.cif';write_model(p)
    text=p.read_text().replace('_space_group_name_H-M_alt', '_space_group_name_H-M_alt')
    # Change an explicit identity operation without changing the declared P1 group.
    doc=gemmi.cif.read_string(text);block=doc.sole_block()
    loop=block.init_loop('_space_group_symop_', ['id','operation_xyz']);loop.add_row(['1',"'-x,-y,-z'"])
    doc.write_file(str(p))
    assert not audit_cif(p)['model_eligible']
    with pytest.raises(ValueError,match='audit'):predict([p],[dict(id='h',hkl=[1,0,0])])


def test_special_position_occupancy_is_not_double_counted(tmp_path):
    p=tmp_path/'special.cif'
    p.write_text("""data_special
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P -1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_U_iso_or_equiv
C1 C 0 0 0 1 0.02
""")
    value=predict([p],[dict(id='h',hkl=[1,0,0])])['predictions'][0][0]
    f=gemmi.Element('C').it92.calculate_sf(.0025)*np.exp(-8*np.pi**2*.02*.0025)
    assert value==pytest.approx(f**2,rel=2e-7)


def test_multiple_blocks_require_explicit_selection(tmp_path):
    p=tmp_path/'m.cif';write_model(p)
    text=p.read_text();p.write_text(text+text.replace('data_synthetic_model','data_other'))
    with pytest.raises(ValueError,match='block'):audit_cif(p)
    assert audit_cif(p,block='other')['model_eligible']


@pytest.mark.parametrize('bad',[[1.2,0,0],[True,0,0],[1,0],[1,0,0,0]])
def test_integer_hkl_required(bad):
    with pytest.raises(ValueError):transform_hkl(bad,None)


def test_rational_index_mapping_is_exact():
    assert transform_hkl([2,3,4],[['1/2',0,0],[0,1,0],[0,0,1]])==[1,3,4]
    with pytest.raises(ValueError,match='nonintegral'):
        transform_hkl([1,3,4],[['1/2',0,0],[0,1,0],[0,0,1]])


def test_singular_index_mapping_rejected():
    with pytest.raises(ValueError,match='singular'):transform_hkl([1,2,3],[[1,0,0],[0,0,0],[0,0,1]])


def test_zero_reflection_is_not_a_measurement(tmp_path):
    p=tmp_path/'m.cif';write_model(p)
    with pytest.raises(ValueError,match='F000'):predict([p],[dict(id='h',hkl=[0,0,0])])


def test_chemical_elements_are_checked_not_invented(tmp_path):
    p=tmp_path/'m.cif';write_model(p)
    result=audit_cif(p,expected_elements=['Zn','C','O','N'])
    assert not result['model_eligible']
    assert any('unexpected' in v for v in result['errors'])
    assert any('unmodelled' in v for v in result['warnings'])


def test_missing_atoms_is_not_an_eligible_model(tmp_path):
    p=tmp_path/'m.cif';write_model(p)
    text=p.read_text();p.write_text(text.split('loop_')[0])
    assert not audit_cif(p)['model_eligible']


def test_ensemble_cell_setting_must_be_consistent(tmp_path):
    a=tmp_path/'a.cif';b=tmp_path/'b.cif';write_model(a);write_model(b)
    st,block=load_cif(b);st.cell=gemmi.UnitCell(22,18,20,90,96,90);st.make_cif_block().write_file(str(b))
    with pytest.raises(ValueError,match='setting'):
        predict([a,b],[dict(id='a',hkl=[1,2,3])])
