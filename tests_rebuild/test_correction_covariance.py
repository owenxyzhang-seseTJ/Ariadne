"""Reject unphysical component uncertainty before exporting measured totals."""
import argparse
import json

import numpy as np
import pytest

from correct_overlap_groups import correct


def case(tmp_path, covariance):
    record=dict(observation_id='g',status='joint_components',quality_flags=[],basis='synthetic',
                contributors=[dict(domain=1,hkl=[1,0,0]),dict(domain=2,hkl=[0,1,0])],
                components=[dict(intensity=-2.),dict(intensity=10.)],
                covariance_identifiable_subspace=covariance)
    correction=dict(schema='scxrd-group-corrections/1',basis='synthetic',domain_count=2,
                    correction_provenance='synthetic',integration_provenance='synthetic',
                    complete_contributor_audit=True,contains_lp_scaling_absorption=True,
                    scale_convention='common_across_domains_keep_twin_fraction',
                    groups={'g':dict(factors=[2.,3.],factor_uncertainty='treated_as_fixed')})
    integrated,config=tmp_path/'groups.jsonl',tmp_path/'corrections.json'
    integrated.write_text(json.dumps(record)+'\n');config.write_text(json.dumps(correction))
    return argparse.Namespace(integrated=integrated,corrections=config,output=tmp_path/'out')


@pytest.mark.parametrize('covariance', [[[1.,2.],[2.,1.]],[[1.,.1],[.3,1.]],[[1.]]])
def test_invalid_component_covariance_cannot_become_an_exportable_total(tmp_path,covariance):
    with pytest.raises(ValueError,match='covariance'):
        correct(case(tmp_path,covariance))


def test_signed_components_and_correlated_covariance_propagate_exactly(tmp_path):
    covariance=np.array([[4.,-.5],[-.5,2.]])
    args=case(tmp_path,covariance.tolist());correct(args)
    result=json.loads((args.output/'corrected_observations.jsonl').read_text().splitlines()[1])
    assert result['intensity']==26.
    assert result['sigma']**2==pytest.approx(np.array([2.,3.])@covariance@np.array([2.,3.]))


@pytest.mark.parametrize('chunk_rows',[0,17])
def test_joint_fit_accepts_independently_known_background(chunk_rows):
    from overlap_integrator import solve_joint
    rng=np.random.default_rng(817)
    profiles=rng.uniform(.1,1,(150,2))
    variance=rng.uniform(1,3,150)
    data=profiles@np.array([-2.,7.])+rng.normal(0,.1,150)
    weighted=profiles/np.sqrt(variance[:,None])
    expected=np.linalg.lstsq(weighted,data/np.sqrt(variance),rcond=None)[0]
    result=solve_joint(data,variance,profiles,np.ones(150,bool),np.empty((150,0)),1e-10,
                       chunk_rows=chunk_rows)
    np.testing.assert_allclose(result['intensities_internal'],expected,atol=1e-12)
    np.testing.assert_allclose(result['covariance_identifiable_subspace'],
                               np.linalg.inv(weighted.T@weighted),rtol=1e-12)
    assert result['dof']==148 and result['background_coefficients'].size==0
