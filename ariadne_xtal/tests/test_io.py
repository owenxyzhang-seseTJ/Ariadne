"""Format/provenance tests. Legacy execution tests use MOCK executables only."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from ariadne_xtal.compat import run_legacy, sha256_file
from ariadne_xtal.io import (SymmetryMismatchWarning, read_cif, read_hkl,
                            read_reflections, read_structure, write_hkl,
                            write_reflections, write_structure)
from ariadne_xtal.model import Atom, Cell, Reflections, Structure
from ariadne_xtal.shelx import (NativeConversionError, ShelxDocument,
                               cartesian_u_to_cif, cif_u_to_cartesian)


def _structure() -> Structure:
    cell = Cell(7.3,8.2,10.4,77,102,89)
    return Structure(cell,[
        Atom("Co1","Co",[0,0,0],u_iso=0.03,u_aniso=np.array([[.021,.002,.001],[.002,.034,-.003],[.001,-.003,.028]]),
             f_prime=-.2,f_double_prime=.3,refine_xyz=False,refine_u=False),
        Atom("O1","O",[.21,.32,.43],occupancy=.7,u_iso=.04,refine_occupancy=True),
    ],spacegroup="P -1")


def test_hkl_fixed_width_and_negative_observations(tmp_path):
    path = tmp_path/"test.hkl"
    refs = Reflections([[-123,-456,123],[1,2,3]],[-1.2345,1234567.0],[.0000012,10.12],batch=[-1,2])
    write_hkl(refs,path)
    read = read_hkl(path)
    np.testing.assert_array_equal(read.hkl,refs.hkl)
    np.testing.assert_allclose(read.intensity,refs.intensity,rtol=1e-5)
    np.testing.assert_allclose(read.sigma,refs.sigma,rtol=1e-8)
    assert read.free_flags is None
    np.testing.assert_array_equal(read.batch,[-1,2])
    np.testing.assert_array_equal(read_hkl(path,negative_batch_is_free=True).free_flags,[True,False])


def test_hkl_fortran_implied_decimal_and_end_record(tmp_path):
    path = tmp_path/"test.hkl"
    path.write_text("   1   2   3    1234     200   1\n   0   0   0    0.00    0.00   0\nTHIS IS A COMMENT AFTER DATA\n")
    refs = read_hkl(path)
    assert len(refs) == 1
    assert refs.intensity[0] == 12.34
    assert refs.sigma[0] == 2.0


def test_hkl_free_format_and_bad_sigma(tmp_path):
    path = tmp_path/"test.hkl"
    path.write_text("1 0 0 -3.1 2.1\n2 0 0 1.2D+4 0.1D+2 -2\n")
    refs = read_hkl(path)
    assert refs.intensity.tolist() == [-3.1,12000]
    path.write_text("1 0 0 100 0\n")
    with pytest.raises(ValueError,match="positive"):
        read_hkl(path)


def test_hkl_refuses_silent_holdout_loss(tmp_path):
    refs = Reflections([[1,0,0],[2,0,0]],[1,2],[1,1],free_flags=[True,False])
    with pytest.raises(ValueError,match="free_flags"):
        write_hkl(refs,tmp_path/"x.hkl")


def test_reflection_json_preserves_precision_and_holdout_flags(tmp_path):
    refs=Reflections([[1,2,3],[3,2,1]],[-.12345678912345678,111.1234567891234],[1.1,2.2],
                     batch=[0,5],free_flags=[True,False])
    path=tmp_path/"refs.json";write_reflections(refs,path)
    restored=read_reflections(path)
    assert restored.to_dict()==refs.to_dict()


def test_document_preserves_original_bytes_and_continuations(tmp_path):
    original = b"TITL raw text\r\nREM comment\r\nC1 1 .2 .3 .4 11 .02 .03 .04 =\r\n  .001 .002 .003 ! rest\r\nEND\r\nQ1 1 0 0 0 11 .01 4.0\r\n"
    source = tmp_path/"x.ins";source.write_bytes(original)
    document = ShelxDocument.from_path(source)
    output = tmp_path/"x.res";document.write(output)
    assert output.read_bytes() == original
    assert len(document.instructions[2].arguments) == 11
    assert document.instructions[-1].active is False


def test_shelx_title_punctuation_is_literal():
    document = ShelxDocument.from_text("TITL a = b ! c\nREM x = y\nEND\n")
    assert document.instructions[0].logical_text == "TITL a = b ! c"
    assert document.instructions[1].logical_text == "REM x = y"


@pytest.mark.parametrize("directive",["AFIX 43","DFIX 1.2 C1 O1","TWIN -1 0 0 0 -1 0 0 0 -1","WGHT .01 .1",
                                       "L.S. 10","CGLS 10","FVAR 1 .5","+restraints.ins","ABIN 1 2","NEUT"])
def test_native_refuses_unsupported_scientific_directives(directive):
    text = f"TITL test\nCELL .71 8 9 10 90 90 90\nLATT -1\nSFAC C\nUNIT 1\n{directive}\nC1 1 .1 .2 .3 11 .02\nHKLF 4\nEND\n"
    with pytest.raises(NativeConversionError):
        ShelxDocument.from_text(text).to_structure()


def test_shelx_special_position_occupancy_and_fixed_parameters():
    text = "TITL test\nCELL .71 8 9 10 90 90 90\nLATT 1\nSFAC C O\nUNIT 1 2\nC1 1 10 10 10 10.5 10.02\nO1 2 .1 .2 .3 11 .03\nHKLF 4\nEND\n"
    with pytest.warns(UserWarning,match="tabulated dispersion"):
        structure = ShelxDocument.from_text(text).to_structure()
    assert len(structure.operations) == 2
    assert structure.atoms[0].occupancy == 1
    assert structure.atoms[1].occupancy == 1
    assert structure.atoms[0].refine_xyz is False
    assert structure.atoms[0].refine_u is False
    assert structure.atoms[0].u_iso == pytest.approx(.02)


def test_shelx_free_variable_atom_parameter_rejected():
    text = "CELL .71 8 9 10 90 90 90\nLATT -1\nSFAC C\nUNIT 1\nC1 1 .1 .2 .3 21 .03\nHKLF 4\nEND\n"
    with pytest.raises(NativeConversionError,match="free-variable"):
        ShelxDocument.from_text(text).to_structure()


def test_shelx_fvar_scale_explicit_metadata():
    text = "CELL .71 8 9 10 90 90 90\nLATT -1\nSFAC C\nUNIT 1\nFVAR 2.3\nC1 1 .1 .2 .3 11 .03\nHKLF 4\nEND\n"
    document = ShelxDocument.from_text(text)
    assert any("re-estimates" in x for x in document.audit()["metadata"])
    with pytest.warns(UserWarning):
        document.to_structure()


def test_cif_to_cartesian_adp_preserves_quadratic_form():
    cell = Cell(7,9,13,71,82,104)
    u_cif = np.array([[.02,.002,.001],[.002,.03,-.002],[.001,-.002,.04]])
    cart = cif_u_to_cartesian(cell,u_cif)
    h = np.array([3,-2,1]); q = cell.reciprocal@h
    norms = np.linalg.norm(cell.reciprocal,axis=0)
    weighted = norms*h
    assert q@cart@q == pytest.approx(weighted@u_cif@weighted,rel=1e-13)
    np.testing.assert_allclose(cartesian_u_to_cif(cell,cart),u_cif,atol=1e-15)


@pytest.mark.parametrize("extension",["json","cif","res"])
def test_model_roundtrips(extension,tmp_path):
    model = _structure();path=tmp_path/f"model.{extension}"
    write_structure(model,path)
    copy = read_structure(path)
    np.testing.assert_allclose(copy.cell.matrix,model.cell.matrix,atol=1e-12)
    assert {x.triplet() for x in copy.operations} == {x.triplet() for x in model.operations}
    for original,restored in zip(model.atoms,copy.atoms):
        assert restored.label.upper() == original.label.upper()
        assert restored.element == original.element
        np.testing.assert_allclose(restored.frac,original.frac,atol=1e-11)
        assert restored.occupancy == pytest.approx(original.occupancy)
        assert restored.f_prime == original.f_prime
        assert restored.f_double_prime == original.f_double_prime
        assert restored.refine_xyz == original.refine_xyz
        assert restored.refine_u == original.refine_u
        assert restored.refine_occupancy == original.refine_occupancy
        if original.u_aniso is not None:
            np.testing.assert_allclose(restored.u_aniso,original.u_aniso,atol=1e-10)


def test_cif_explicit_operations_override_wrong_name_and_export_corrects(tmp_path):
    path=tmp_path/"wrong.cif";write_structure(_structure(),path)
    text=path.read_text().replace("'P -1'","'P 1'")
    path.write_text(text)
    diagnostics=[]
    with pytest.warns(SymmetryMismatchWarning,match="authoritative"):
        model=read_cif(path,diagnostics=diagnostics)
    assert len(model.operations)==2
    assert diagnostics
    out=tmp_path/"corrected.cif";write_structure(model,out)
    assert "'P -1'" in out.read_text()


def test_cif_multiple_models_require_selection(tmp_path):
    path=tmp_path/"model.cif";write_structure(_structure(),path)
    text=path.read_text()
    path.write_text(text+"\n"+text.replace("data_ariadne_xtal","data_second"))
    with pytest.raises(ValueError,match="exactly one"):
        read_structure(path)
    assert len(read_structure(path,block_name="second").atoms)==2


def test_cif_does_not_guess_missing_wavelength(tmp_path):
    path=tmp_path/"missing.cif";write_structure(_structure(),path)
    path.write_text("\n".join(line for line in path.read_text().splitlines() if not line.startswith("_diffrn_radiation_wavelength")))
    with pytest.raises(ValueError,match="explicit X-ray wavelength"):
        read_structure(path)


def test_cif_neutron_is_not_silently_treated_as_xray(tmp_path):
    path=tmp_path/"neutron.cif";write_structure(_structure(),path)
    path.write_text(path.read_text()+"\n_diffrn_radiation_probe neutron\n")
    with pytest.raises(ValueError,match="X-ray form factors"):
        read_structure(path)


@pytest.mark.parametrize("spacegroup",["P 21 21 21","C 1 2/c 1","F m -3 m","I 41/a:2"])
def test_shelx_centering_and_inversion_factored_into_latt(spacegroup,tmp_path):
    cell=Cell(8,8,8)
    model=Structure(cell,[Atom("C1","C",[.123,.234,.345])],spacegroup=spacegroup)
    path=tmp_path/"model.res";write_structure(model,path)
    restored=read_structure(path)
    assert {x.triplet() for x in restored.operations}=={x.triplet() for x in model.operations}
    assert "SYMM -x,-y,-z\n" not in path.read_text()


def _mock_executable(tmp_path: Path,body: str) -> Path:
    executable=tmp_path/"mock_backend.py"
    executable.write_text("#!"+sys.executable+"\n"+body)
    executable.chmod(0o755)
    return executable


def _legacy_inputs(tmp_path: Path) -> Path:
    root=tmp_path/"input";root.mkdir()
    path=root/"sample.ins"
    path.write_bytes(b"TITL MOCK BACKEND FIXTURE\r\n+restraints/one.inc\r\nHKLF 4\r\nEND\r\n")
    (root/"sample.hkl").write_bytes(b"   1   0   0    1.00    1.00   1\n")
    (root/"restraints").mkdir();(root/"nested").mkdir()
    (root/"restraints/one.inc").write_text("+nested/two.inc\nDFIX 1.2 C1 O1\nEND\n")
    (root/"nested/two.inc").write_text("REM nested reference from process cwd\nEND\n")
    return path


def test_legacy_mock_isolated_bytes_hashes_literal_arguments(tmp_path):
    source=_legacy_inputs(tmp_path)
    original=source.read_bytes()
    mock=_mock_executable(tmp_path,"import json, pathlib, sys\np=pathlib.Path(sys.argv[1]+'.ins')\npathlib.Path(sys.argv[1]+'.res').write_bytes(p.read_bytes())\npathlib.Path('args.json').write_text(json.dumps(sys.argv[2:]))\n")
    malicious="$(touch injected) ; echo ignored"
    result=run_legacy("shelxl",mock,source,tmp_path/"output",extra_args=[malicious])
    work=Path(result["working_directory"])
    assert result["status"]=="completed"
    assert result["scientific_validation_performed"] is False
    assert (work/"sample.res").read_bytes()==original
    assert source.read_bytes()==original
    assert (work/"nested/two.inc").exists()
    assert json.loads((work/"args.json").read_text())==[malicious]
    assert not (work/"injected").exists()
    assert next(x for x in result["inputs"] if x["name"]=="sample.ins")["sha256"]==sha256_file(source)
    assert json.loads(Path(result["manifest_path"]).read_text())["status"]=="completed"
    second=run_legacy("shelxl",mock,source,tmp_path/"output")
    assert second["working_directory"]!=result["working_directory"]


def test_legacy_mock_nonzero_and_timeout(tmp_path):
    source=_legacy_inputs(tmp_path)
    mock=_mock_executable(tmp_path,"raise SystemExit(7)\n")
    result=run_legacy("shelxt",mock,source,tmp_path/"output")
    assert result["status"]=="process_failed" and result["returncode"]==7
    mock=_mock_executable(tmp_path,"import time\ntime.sleep(10)\n")
    result=run_legacy("shelxl",mock,source,tmp_path/"output",timeout=.1)
    assert result["status"]=="timed_out"
    assert result["elapsed_seconds"]<3


def test_legacy_input_traversal_and_external_symlink_rejected(tmp_path):
    source=_legacy_inputs(tmp_path)
    mock=_mock_executable(tmp_path,"raise SystemExit(0)\n")
    outside=tmp_path/"outside.inc";outside.write_text("REM private\n")
    source.write_text("+../outside.inc\nHKLF 4\nEND\n")
    with pytest.raises(ValueError,match="inside"):
        run_legacy("shelxl",mock,source,tmp_path/"output")
    (source.parent/"link.inc").symlink_to(outside)
    source.write_text("+link.inc\nHKLF 4\nEND\n")
    with pytest.raises(ValueError,match="leaves"):
        run_legacy("shelxl",mock,source,tmp_path/"output")


def test_legacy_include_cycle_rejected(tmp_path):
    source=_legacy_inputs(tmp_path)
    mock=_mock_executable(tmp_path,"raise SystemExit(0)\n")
    (source.parent/"nested/two.inc").write_text("+restraints/one.inc\nEND\n")
    with pytest.raises(ValueError,match="Cyclic"):
        run_legacy("shelxl",mock,source,tmp_path/"output")


def test_legacy_executable_requires_explicit_path(tmp_path):
    source=_legacy_inputs(tmp_path)
    with pytest.raises(ValueError,match="absolute"):
        run_legacy("shelxl","shelxl",source,tmp_path/"output")
