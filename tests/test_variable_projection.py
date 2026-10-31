"""Exact reduced residual derivatives and complete-model covariance checks."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.linalg import solve_triangular

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'skills/ariadne-reduction/scripts'))
from variable_projection import ProfiledScaleModel, RankDeficientProfile
from composite_scaling import fit_global
from test_composite_scaling import composite_truth, block


def profiled_model(blocks, rcond=1e-10, **kwargs):
    ids = list(dict.fromkeys(label for b in blocks for label in b['equivalence_ids']))
    prepared, designs, targets = [], [], []
    for b in blocks:
        lower=np.linalg.cholesky(b['covariance'])
        prepared.append((b['y'],lower,b['design'],b['features'],b['known_factors'],
                         np.array([ids.index(label) for label in b['equivalence_ids']])))
        designs.append(solve_triangular(lower,b['design'],lower=True))
        targets.append(solve_triangular(lower,b['y'],lower=True))
    return ProfiledScaleModel(prepared,designs,np.concatenate(targets),len(ids),rcond,lambda:None,
                              **kwargs)


class VariableProjectionTests(unittest.TestCase):
    def test_exact_profile_jacobian_away_from_solution(self):
        blocks,_,_=composite_truth(noisy=True)
        model=profiled_model(blocks)
        beta=np.array([-.31,.27])
        state=model.state(beta)
        columns=[]
        for axis in range(2):
            direction=np.eye(2)[axis]*1e-5
            columns.append((model.state(beta+direction)['residual']-model.state(beta-direction)['residual'])/(2e-5))
        reference=np.column_stack(columns)
        np.testing.assert_allclose(state['jacobian'],reference,rtol=2e-8,atol=2e-8)
        a=state['joint_jacobian'][:,:3]
        jb=state['joint_jacobian'][:,3:]
        incomplete=jb-a @ np.linalg.lstsq(a,jb,rcond=None)[0]
        self.assertGreater(np.linalg.norm(incomplete-reference),1.)

    def test_signed_fit_and_covariance_match_joint_solution(self):
        for noisy in (False,True):
            with self.subTest(noisy=noisy):
                blocks,_,_=composite_truth(noisy)
                joint=fit_global(blocks)
                actual=fit_global(blocks,optimizer='profiled')
                self.assertEqual(actual['status'],'converged',actual['reason'])
                self.assertLess(actual['I'][-1],0.)
                np.testing.assert_allclose(actual['I'],joint['I'],rtol=2e-8,atol=2e-8)
                np.testing.assert_allclose(actual['beta'],joint['beta'],atol=2e-9)
                np.testing.assert_allclose(actual['joint_covariance'],joint['joint_covariance'],rtol=2e-7,atol=1e-9)
                self.assertAlmostEqual(actual['chi2'],joint['chi2'],delta=1e-10)
                self.assertIn('variable_projection',[r['phase'] for r in actual['trace']])

    def test_repeated_contributor_ids_and_cache_copy(self):
        blocks,_,_=composite_truth(noisy=True)
        for b in blocks:
            b['design']=np.column_stack((b['design'][:,0]/2,b['design'][:,0]/2,b['design'][:,1:]))
            b['features']=np.vstack((b['features'][0],b['features']))
            b['known_factors']=np.r_[b['known_factors'][0],b['known_factors']]
            b['equivalence_ids']=[b['equivalence_ids'][0],*b['equivalence_ids']]
        model=profiled_model(blocks); beta=np.array([.1,-.2])
        before=model.state(beta); evaluated=model.evaluations
        self.assertIs(model.state(beta.copy()),before)
        self.assertEqual(model.evaluations,evaluated)
        beta[0]+=1e-6
        self.assertIsNot(model.state(beta),before)
        self.assertEqual(model.evaluations,evaluated+1)

    def test_rank_deficient_profile_falls_back_without_ridge(self):
        blocks=[]
        for x in (-1.,1.):
            blocks.append(block(['a','b'],[[1,1]],[[x],[x]],np.array([.1]),[3.,2.]))
        actual=fit_global(blocks,optimizer='profiled')
        self.assertEqual(actual['status'],'unresolved')
        self.assertIn('profiled_fallback',actual)
        self.assertLess(actual['rank'],3)

    def test_support_partition_retains_derivatives_at_cancellation(self):
        # At beta=0 the first predicted row is exactly zero, but its derivative
        # is nonzero.  Splitting on the current A would lose that information.
        b=block(['a','a'],[[1,-1],[1,0]],[[1],[-1]],np.zeros(1),[1.,1.])
        b['y']=np.array([3.,1.])
        state=profiled_model([b]).state(np.zeros(1))
        np.testing.assert_allclose(state['intensity'],[1.],atol=1e-14)
        np.testing.assert_allclose(state['residual'],[-3.,0.],atol=1e-14)
        np.testing.assert_allclose(state['jacobian'],[[2.],[6.]],atol=1e-14)

    def test_partition_uses_global_singular_cutoff(self):
        # Five columns have singular values (2,.5,.5,.5,.5); a separate pair
        # has (.sqrt(2-.36**2),.36).  A per-component rcond=.2 would accept
        # both, whereas the full design correctly rejects .36 < .2*2.
        first=np.linalg.cholesky(.75*np.ones((5,5))+.25*np.eye(5)).T
        rho=1-.36**2
        second=np.linalg.cholesky([[1.,rho],[rho,1.]]).T
        blocks=[block([f'a{i}' for i in range(5)],first,np.zeros((5,1)),[0.],np.ones(5)),
                block(['b0','b1'],second,np.zeros((2,1)),[0.],np.ones(2))]
        with self.assertRaises(RankDeficientProfile):
            profiled_model(blocks,rcond=.2).state(np.zeros(1))
        np.testing.assert_allclose(profiled_model(blocks,rcond=.17).state(np.zeros(1))['intensity'],
                                   np.ones(7),atol=1e-14)

    def test_disconnected_components_and_constant_rows_match_dense_fit(self):
        blocks,_,_=composite_truth(noisy=True)
        independent=block(['separate'],[[1.],[0.],[2.]],[[.4,-.8]],
                          [.24,-.16],[-2.],covariance=np.diag([.8,1.3,.9]))
        independent['y']+=np.array([.2,3.,-.1])
        blocks.append(independent)
        beta=np.array([-.23,.19])
        state=profiled_model(blocks).state(beta)
        a=state['joint_jacobian'][:,:4]
        y=np.concatenate([solve_triangular(np.linalg.cholesky(b['covariance']),b['y'],lower=True)
                          for b in blocks])
        reference=np.linalg.lstsq(a,y,rcond=1e-10)[0]
        np.testing.assert_allclose(state['intensity'],reference,rtol=2e-13,atol=2e-13)
        np.testing.assert_allclose(state['residual'],a@reference-y,rtol=2e-13,atol=2e-13)
        np.testing.assert_array_equal(state['jacobian'][-2],np.zeros(2))
        without=profiled_model(blocks,include_joint_jacobian=False).state(beta)
        self.assertIsNone(without['joint_jacobian'])
        for key in ('intensity','residual','jacobian'):
            np.testing.assert_array_equal(without[key],state[key])

    def test_saddle_rejection_and_scale_units_are_retained(self):
        blocks=[block(['a'],[[1]],[[x]],np.zeros(1),[1.]) for x in (-1.,0.,1.)]
        for b,y in zip(blocks,[1.,-1.,1.]):
            b['y'][:]=y
        actual=fit_global(blocks,optimizer='profiled')
        self.assertEqual(actual['reason'],'negative_curvature_stationary_point')
        ordinary,_,_=composite_truth(noisy=True)
        scaled=[]
        for b in ordinary:
            b={key:value.copy() if hasattr(value,'copy') else value for key,value in b.items()}
            b['known_factors']*=np.array([1e-12,1.,1e12]); scaled.append(b)
        first=fit_global(ordinary,optimizer='profiled'); second=fit_global(scaled,optimizer='profiled')
        self.assertEqual(second['status'],'converged',second['reason'])
        np.testing.assert_allclose(second['I']*np.array([1e-12,1.,1e12]),first['I'],rtol=1e-8)

    def test_linear_full_rank_does_not_claim_unobservable_scale(self):
        blocks=[block(['a'],[[1]],[[x]],[.2],[0.]) for x in (-1.,1.)]
        actual=fit_global(blocks,optimizer='profiled')
        self.assertEqual(actual['status'],'unresolved')
        self.assertEqual(actual['rank'],1)
        self.assertFalse(actual['identifiable_parameters'][-1])
        self.assertIsNone(actual.get('profiled_fallback'))


if __name__=='__main__':
    unittest.main()
