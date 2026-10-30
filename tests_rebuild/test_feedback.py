import numpy as np
import pytest
from scipy.linalg import block_diag
from ariadne.feedback import (condition, dense_covariance, ensemble_prior, calibrate,
                              closed_split, observations)


def dense_reference(m,d,U,blocks):
    H=np.zeros((sum(len(b['values']) for b in blocks),len(m)))
    offset=0
    for b in blocks:
        size=len(b['values']);H[offset:offset+size,b['columns']]=b['design'];offset+=size
    y=np.concatenate([b['values'] for b in blocks]) if blocks else np.zeros(0)
    R=block_diag(*[b['covariance'] for b in blocks]) if blocks else np.zeros((0,0))
    K=np.diag(d)+U@U.T
    S=R+H@K@H.T
    r=y-H@m
    if not len(y):
        return m,K,0.
    mu=m+K@H.T@np.linalg.solve(S,r)
    C=K-K@H.T@np.linalg.solve(S,H@K)
    nll=.5*(r@np.linalg.solve(S,r)+np.linalg.slogdet(S)[1]+len(y)*np.log(2*np.pi))
    return mu,C,nll


def problem(seed=2,rank=3):
    rng=np.random.default_rng(seed);n=9
    m=rng.normal(size=n);d=rng.uniform(.1,2.,n);U=rng.normal(size=(n,rank))*.3
    blocks=[]
    for i, cols in enumerate(([0,1,2],[1,3],[4,5],[5,6])):
        p=2;H=rng.normal(size=(p,len(cols)));B=rng.normal(size=(p,p));R=B@B.T+np.eye(p)*.1
        blocks.append(dict(id=str(i),columns=list(cols),design=H.tolist(),values=rng.normal(size=p).tolist(),
                           covariance=R.tolist(),raw_ids=[str(i)]))
    return m,d,U,blocks


@pytest.mark.parametrize('rank',[0,1,4])
@pytest.mark.parametrize('seed',[1,9,104])
def test_matches_dense_reference(seed,rank):
    m,d,U,b=problem(seed,rank);actual=condition(m,d,U,b);mu,C,nll=dense_reference(m,d,U,b)
    np.testing.assert_allclose(actual['mean'],mu,rtol=2e-12,atol=1e-12)
    np.testing.assert_allclose(dense_covariance(actual),C,rtol=5e-12,atol=1e-12)
    assert actual['predictive_nll']==pytest.approx(nll,rel=1e-12)
    assert np.linalg.eigvalsh(dense_covariance(actual))[0]>0


def test_exact_overlap_retains_unmeasured_difference():
    b=[dict(id='sum',columns=[0,1],design=[[1,1]],values=[13],covariance=[[1]])]
    r=condition([4,6],[2,2],np.zeros((2,0)),b);C=dense_covariance(r)
    v=np.array([1.,-1.]);w=np.ones(2)
    assert r['data_rank']==1 and r['data_identifiable']==[False,False]
    assert v@C@v==pytest.approx(4.) and w@C@w<4.


def test_signed_observation_is_not_truncated():
    b=[dict(id='weak',columns=[0],design=[[1]],values=[-3.],covariance=[[.01]])]
    r=condition([0.],[100.],np.zeros((1,0)),b)
    assert r['mean'][0]<-2.9 and b[0]['values']==[-3.]


@pytest.mark.parametrize('scale',[1e-10,1e-5,1e5,1e10])
def test_intensity_units_do_not_change_inference(scale):
    m,d,U,b=problem();target=condition(m,d,U,b)
    changed=[{**x,'design':(np.asarray(x['design'])/scale).tolist()} for x in b]
    result=condition(m*scale,d*scale**2,U*scale,changed)
    np.testing.assert_allclose(np.asarray(result['mean'])/scale,target['mean'],rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(np.asarray(result['variance'])/scale**2,target['variance'],rtol=1e-11)
    assert result['predictive_nll']==pytest.approx(target['predictive_nll'],rel=1e-12)


def test_repeated_exposures_apply_prior_once():
    blocks=[dict(id=str(i),columns=[0],design=[[1]],values=[4],covariance=[[1]]) for i in range(5)]
    result=condition([0],[1],np.zeros((1,0)),blocks)
    assert result['mean'][0]==pytest.approx(20/6)
    assert result['variance'][0]==pytest.approx(1/6)


def test_unobserved_variables_keep_prior_and_correlations():
    m,d,U,b=problem();r=condition(m,d,U,b)
    assert r['data_identifiable'][-1] is False
    mu,C,_=dense_reference(m,d,U,b)
    assert r['mean'][-1]==pytest.approx(mu[-1])
    assert C[-1,0]!=0


def test_all_unobserved_equals_prior():
    m=np.array([1.,2.]);d=np.array([.2,.3]);U=np.array([[.4],[.7]])
    r=condition(m,d,U,[])
    np.testing.assert_allclose(dense_covariance(r),np.diag(d)+U@U.T)
    assert r['mean']==[1.,2.] and r['predictive_nll']==0


def test_raw_pixel_reuse_is_rejected():
    b=[dict(id=str(i),columns=[i],design=[[1]],values=[0],covariance=[[1]],raw_ids=['same']) for i in range(2)]
    with pytest.raises(ValueError,match='counted more'):
        condition([0,0],[1,1],np.zeros((2,0)),b)


@pytest.mark.parametrize('bad',[0.,-1.,float('nan'),float('inf')])
def test_invalid_prior_variance_rejected(bad):
    with pytest.raises(ValueError):condition([0],[bad],np.zeros((1,0)),[])


def test_singular_noise_not_regularized():
    with pytest.raises(ValueError,match='positive definite'):
        condition([0],[1],np.zeros((1,0)),[dict(id='a',columns=[0],design=[[1]],values=[1],covariance=[[0]])])


def test_memory_budget_is_enforced():
    m,d,U,b=problem()
    with pytest.raises(MemoryError):condition(m,d,U,b,max_local_elements=1)


def test_dense_export_is_opt_in_and_bounded():
    r=condition([1,2],[1,1],np.zeros((2,0)),[])
    with pytest.raises(MemoryError):dense_covariance(r,max_size=1)


def test_ensemble_is_epistemic_mixture_not_amplitude_average():
    J=np.array([[0.,10.],[2.,4.]])
    r=ensemble_prior(J,[.25,.75],absolute_sd=1,relative_sd=0)
    m=np.asarray(r['mean']);U=np.asarray(r['factors'])
    np.testing.assert_allclose(m,[7.5,3.5]);assert (U@U.T)[0,0]==pytest.approx(18.75)


def test_ensemble_requires_nonzero_model_error_floor():
    with pytest.raises(ValueError):ensemble_prior([[1]],absolute_sd=0)


def test_calibration_propagates_scale_and_covariance():
    prior=ensemble_prior([[4],[7],[6],[9]],absolute_sd=.5,relative_sd=.1)
    coeff=np.array([2.,3.]);dom=[0,1,0,1];blocks=[]
    for j,m in enumerate(prior['mean']):
        blocks.append(dict(id=str(j),columns=[j],design=[[1]],values=[coeff[dom[j]]*m],covariance=[[.2]]))
    r=calibrate(prior,dom,blocks)
    np.testing.assert_allclose(r['coefficients'],coeff)
    np.testing.assert_allclose(r['fractions'],[.4,.6])
    np.testing.assert_allclose(r['independent_variance'],np.array(prior['independent_variance'])*np.array([4,9,4,9]))
    assert np.linalg.eigvalsh(r['coefficient_covariance'])[0]>0


def test_calibration_does_not_separate_identical_domain_columns():
    p=ensemble_prior([[4],[4]])
    b=[dict(id='x',columns=[0,1],design=[[1,1]],values=[8],covariance=[[1]])]
    with pytest.raises(ValueError,match='identifiable'):calibrate(p,[0,1],b)


def test_calibration_boundary_is_not_given_interior_errorbar():
    p=ensemble_prior([[4]])
    b=[dict(id='x',columns=[0],design=[[1]],values=[-2],covariance=[[1]])]
    with pytest.raises(ValueError,match='boundary'):calibrate(p,[0],b)


def test_split_preserves_shared_latents_and_declared_orbits():
    b=[dict(id=str(i),columns=[i],design=[[1]],values=[1],covariance=[[1]],
            equivalence_ids=['shared'] if i<2 else []) for i in range(10)]
    result=closed_split(b,10,'a')
    assert result['0']==result['1'] and set(result.values())=={'train','validation','test'}
    assert result==closed_split(b,10,'a')


def test_no_fake_holdout_for_one_connected_group():
    b=[dict(id=str(i),columns=[0],design=[[1]],values=[1],covariance=[[1]]) for i in range(4)]
    with pytest.raises(ValueError,match='independent'):closed_split(b,1)


def test_inflating_model_variance_is_penalized_by_logdet():
    b=[dict(id='x',columns=[0],design=[[1]],values=[0],covariance=[[1]])]
    modest=condition([0],[1],np.zeros((1,0)),b)['predictive_nll']
    inflated=condition([0],[1e12],np.zeros((1,0)),b)['predictive_nll']
    assert inflated>modest


def test_duplicate_latent_columns_rejected():
    b=[dict(id='x',columns=[0,0],design=[[1,1]],values=[0],covariance=[[1]])]
    with pytest.raises(ValueError):observations(b,1)


def test_tiny_nonsymmetric_covariance_is_not_silently_accepted():
    b=[dict(id='x',columns=[0],design=[[1],[1]],values=[0,0],
            covariance=[[1e-20,1e-21],[0,1e-20]])]
    with pytest.raises(ValueError,match='symmetric'):condition([0],[1],np.zeros((1,0)),b)
