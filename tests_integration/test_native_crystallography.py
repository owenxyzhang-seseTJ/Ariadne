"""Cross-package physical conventions, with independent structure-factor sums."""
import numpy as np
import pytest

pytest.importorskip('ariadne_xtal')
from ariadne.crystal import predict
from ariadne_xtal.io import write_structure
from ariadne_xtal.model import Atom, Cell, Structure
from ariadne_xtal.scattering import calculated_intensities


@pytest.mark.parametrize('spacegroup,special', [('P 1',False), ('P -1',False),
                                              ('P -1',True), ('P 21/c',False)])
def test_independent_native_and_agent_predictions_share_units_and_occupancies(tmp_path, spacegroup, special):
    cell = Cell(8, 9, 10, 90, 107, 90)
    structure = Structure(cell, [
        Atom('C1','C',[0,0,0] if special else [.13,.22,.31],occupancy=.7,u_iso=.025),
        Atom('O1','O',[.37,.41,.55],u_aniso=np.array([[.03,.003,.002],[.003,.04,.001],[.002,.001,.05]]))
    ], spacegroup=spacegroup)
    path = tmp_path/'model.cif'
    write_structure(structure,path)
    hkls = [[1,2,3],[-2,1,1],[0,2,0],[1,2,3],[3,-1,2]]
    reflections = [dict(id=str(i),hkl=h) for i,h in enumerate(hkls)]
    actual = np.asarray(predict([path,path],reflections)['predictions'])
    expected = calculated_intensities(structure,hkls)
    # Gemmi evaluates its IT92 table in finite precision; use the same
    # independent-engine tolerance as Xtal's direct Gemmi comparisons.
    np.testing.assert_allclose(actual,np.column_stack((expected,expected)),rtol=3e-6,atol=2e-5)
    np.testing.assert_array_equal(actual[0],actual[3])


def test_candidate_prediction_keeps_transformed_reflection_order(tmp_path):
    structure = Structure(Cell(8,9,10),[Atom('C1','C',[.1,.2,.3]),Atom('O1','O',[.4,.3,.2])])
    path = tmp_path/'model.cif'
    write_structure(structure,path)
    transform = [[0,1,0],[1,0,0],[0,0,-1]]
    rows = [dict(id='first',hkl=[1,2,3],index_transform=transform),
            dict(id='repeat',hkl=[2,1,-3]),dict(id='last',hkl=[3,2,1])]
    values = predict([path],rows)['predictions']
    expected = calculated_intensities(structure,[[2,1,-3],[2,1,-3],[3,2,1]])
    np.testing.assert_allclose(np.asarray(values)[:,0],expected,rtol=3e-6,atol=2e-5)
