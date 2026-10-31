"""Exact factorization and stable profile-ratio regressions with independent oracles."""
from decimal import Decimal, localcontext
from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gammaln
from scipy.stats import chi2

SCRIPTS = Path(__file__).resolve().parents[1] / 'skills/ariadne-reduction/scripts'
sys.path.insert(0, str(SCRIPTS))
from constrained_pixels import (fit_constrained, _poisson_likelihood_difference,
                                _poisson_predictor_reduction, _independent_poisson_profile)
from test_constrained_pixels import fit, noise


class LikelihoodRound2Tests(unittest.TestCase):
    def test_poisson_sufficient_statistics_preserve_arbitrary_likelihood_ratios_and_gradient(self):
        # First two rows measure I1, next two I2+I3; final affine row remains.
        P = np.array([[.2,0,0],[.7,0,0],[0,.1,.1],[0,.3,.3],[.2,.3,.1]])
        units, gain = np.array([4.,.5,2.]), 1.7
        design = P * units / gain
        counts = np.array([2.,1.,0.,3.,5.])
        bg = np.array([0.,0.,0.,0.,.4])
        n, b, D, removed = _poisson_predictor_reduction(counts, bg, design, P)
        self.assertEqual(removed, 2)
        def likelihood(x, n, b, D):
            mu = b + D @ x
            return float(np.sum(mu-n*np.log(mu)+gammaln(n+1))), D.T @ (1-n/mu)
        base = np.array([1.,2.,3.])
        for trial in (np.array([2.,1.,4.]), np.array([.1,5.,.2])):
            f, g = likelihood(trial, counts, bg, design)
            fc, gc = likelihood(trial, n, b, D)
            assert_allclose(f-likelihood(base, counts, bg, design)[0],
                            fc-likelihood(base, n, b, D)[0], atol=2e-14, rtol=0)
            assert_allclose(g, gc, atol=2e-15, rtol=2e-15)

    def test_near_equal_profiles_are_not_aggregated(self):
        P = np.array([[.1,.1+1e-14],[.2,.2+1e-14]])
        counts, bg = np.array([1.,2.]), np.zeros(2)
        n, b, D, removed = _poisson_predictor_reduction(counts, bg, P, P)
        self.assertEqual(removed, 0)
        assert_allclose(D, P, atol=0, rtol=0)
        self.assertIs(n, counts)

    def test_aggregation_does_not_cross_integer_resolution(self):
        P = np.ones((2,1))
        counts = np.array([2.**52, 1.])
        result = _poisson_predictor_reduction(counts, np.zeros(2), P, P)
        self.assertEqual(result[3], 0)

    def test_centered_poisson_difference_matches_80_digit_algebra(self):
        for n, base, change in ((1e12,1e12,.01), (1e12,1e12,1e6),
                                (3.,4.,-3.9), (1e6,1e6+.1,-.02)):
            with localcontext() as context:
                context.prec = 80
                dn, db, dc = (Decimal.from_float(v) for v in (n,base,change))
                expected = float(dc-dn*((db+dc)/db).ln())
            actual = _poisson_likelihood_difference(np.array([n]),np.array([base]),np.array([change]))
            assert_allclose(actual, expected, atol=3e-15, rtol=3e-13)

    def test_poisson_support_boundary_is_exact(self):
        self.assertTrue(np.isinf(_poisson_likelihood_difference(np.array([2.]),np.array([2.]),np.array([-2.]))))
        self.assertEqual(_poisson_likelihood_difference(np.array([0.]),np.array([2.]),np.array([-2.])), -2.)
        self.assertEqual(_poisson_likelihood_difference(np.array([0.]),np.array([0.]),np.array([2.])), 2.)

    def test_constant_chi_square_cannot_destroy_signal_profile_interval(self):
        # Background-only observations create chi2~1e17. Their constant cannot
        # be added before subtracting two profile objectives separated by O(1).
        y = np.r_[1e8+3.6, np.tile([0.,2e8],8)]
        P = np.r_[1.,np.zeros(16)][:,None]
        result = fit_constrained(y,np.ones(len(y)),P,np.ones(len(y),bool),
                                 np.ones((len(y),1)),None,'nnls',1e-8)
        expected = y[0]-result['background_coefficients'][0]
        interval = result['total_interval']
        self.assertEqual(interval['status'],'ok')
        assert_allclose([interval['lower'],interval['upper']],
                        [expected-np.sqrt(chi2.ppf(.95,1)),expected+np.sqrt(chi2.ppf(.95,1))],
                        atol=2e-9,rtol=0)
        self.assertGreater(result['chi2'],1e17)
        self.assertEqual(result['likelihood_diagnostics']['likelihood_factorization']['parameter_independent_pixels'],16)

    def test_poisson_disjoint_profiles_reduce_to_independent_total_counts(self):
        P = np.zeros((15,3))
        P[:5,0] = .2
        P[5:10,1] = .2
        P[10:,2] = .2
        y = np.array([0,1,3,1,2, 2,1,0,0,1, 0,0,0,0,0],float)
        result = fit(y,P,method='poisson',calibration=noise(1.,0.))
        self.assertTrue(result['converged'])
        assert_allclose(result['intensities_internal'],[7.,4.,0.],atol=2e-7)
        stats=result['likelihood_diagnostics']['likelihood_factorization']
        self.assertEqual(stats['optimization_observations'],3)
        self.assertEqual(stats['poisson_sufficient_statistics_rows_removed'],12)
        cutoff=chi2.ppf(.95,1)
        for expected,interval in zip([7.,4.,0.],result['profile_intervals']):
            self.assertEqual(interval['status'],'ok')
            for key in ['lower','upper']:
                t=interval[key]
                if t==0:
                    continue
                direct=2*(t-expected-expected*np.log(t/expected)) if expected else 2*t
                assert_allclose(direct,cutoff,atol=3e-7)

    def test_poisson_complete_overlap_retains_only_total_information(self):
        p=np.array([.1,.2,.3,.4])
        result=fit([0,3,4,1],np.column_stack([p,p,p]),method='poisson',calibration=noise(1.,0.))
        self.assertEqual(result['rank'],1)
        self.assertTrue(all(x['status']=='unidentifiable' for x in result['profile_intervals']))
        self.assertEqual(result['likelihood_diagnostics']['likelihood_factorization']['optimization_observations'],1)
        assert_allclose(result['intensities_internal'].sum(),8.,atol=1e-7)
        endpoint=brentq(lambda t:2*(t-8-8*np.log(t/8))-chi2.ppf(.95,1),8.,30.)
        assert_allclose(result['total_interval']['upper'],endpoint,atol=2e-7)

    def test_scalar_dual_total_profile_handles_zero_count_active_set_change(self):
        counts=np.array([3.,0.])
        exposure=np.array([2.,1.])
        solver=_independent_poisson_profile(counts,np.zeros(2),np.diag(exposure),np.ones(2),np.array([1.5,0.]))
        def objective(I):
            if I[0]<=0:
                return np.inf
            return 2*I[0]+I[1]-3*np.log(2*I[0])
        for total in [.2,1.5,2.9,3.,4.,10.]:
            answer=solver(np.ones(2),total)
            assert_allclose(answer,[min(total,3.),max(total-3.,0.)],atol=3e-14)
            result=minimize_scalar(lambda v:objective([v,total-v]),bounds=(1e-15,total),method='bounded',
                                   options={'xatol':1e-12})
            optimum=min(result.fun,objective([total,0.]))
            assert_allclose(objective(answer),optimum,atol=2e-10)

    def test_scalar_dual_unequal_positive_exposures_matches_independent_primal_search(self):
        n,e=np.array([3.,2.]),np.array([.3,1.7])
        units=np.array([.4,5.])
        solver=_independent_poisson_profile(n,np.zeros(2),np.diag(e*units),units,n/e/units)
        for total in [.2,3.,20.]:
            result=solver(np.ones(2),total)*units
            assert_allclose(result.sum(),total,rtol=2e-15)
            def objective(t):
                mu=e*np.array([t,total-t])
                return float(np.sum(mu-n*np.log(mu)))
            reference=minimize_scalar(objective,bounds=(1e-15,total-1e-15),method='bounded',options={'xatol':1e-12})
            assert_allclose(objective(result[0]),reference.fun,atol=2e-10)

    def test_all_zero_counts_total_bound_uses_least_exposed_direction(self):
        P=np.array([[.3,0],[0,.8]])
        result=fit([0,0],P,method='poisson',calibration=noise(1.,0.))
        self.assertTrue(result['converged'])
        self.assertGreater(result['likelihood_diagnostics']['analytic_poisson_nuisance_solutions'],0)
        assert_allclose(result['total_interval']['upper'],chi2.ppf(.95,1)/(2*.3),atol=1e-8)

    def test_profile_roots_are_invariant_under_extreme_adu_unit_changes(self):
        P=np.array([[.6],[.3],[.1]])
        signal=np.array([-.6,1.1,2.4])
        base=fit(signal,P,method='poisson-gaussian',background=.5,
                 calibration=noise(1.4,1.1),variance=2.)
        for scale in [1e-12,1e8]:
            result=fit(signal*scale,P,method='poisson-gaussian',background=.5*scale,
                       calibration=noise(1.4*scale,1.1*scale**2),variance=2.*scale**2)
            self.assertEqual(result['total_interval']['status'],'ok')
            for key in ['lower','upper']:
                assert_allclose(result['total_interval'][key]/scale,base['total_interval'][key],rtol=2e-8,atol=2e-8)
            assert_allclose(result['total_interval']['upper_likelihood_ratio'],chi2.ppf(.95,1),atol=1e-6)

    def test_pg_constant_pixel_factor_preserves_normalized_diagnostic(self):
        base=fit([-.6,1.1,2.4],np.array([[.6],[.3],[.1]]),method='poisson-gaussian',
                 background=.5,calibration=noise(1.4,1.1))
        # Add many exact background-only rows without changing the WLS estimate.
        halo=2048
        y=np.r_[[-.6,1.1,2.4],np.full(halo,.5)]
        P=np.vstack(([[.6],[.3],[.1]],np.zeros((halo,1))))
        result=fit_constrained(y,np.ones(len(y)),P,np.ones(len(y),bool),np.ones((len(y),1)),
                               noise(1.4,1.1),'poisson-gaussian',1e-8)
        assert_allclose(result['intensities_internal'],base['intensities_internal'],atol=2e-7)
        assert_allclose(result['total_interval']['upper'],base['total_interval']['upper'],atol=2e-7)
        self.assertEqual(result['likelihood_diagnostics']['likelihood_factorization']['parameter_dependent_pixels'],3)
        self.assertIsNotNone(result['likelihood_diagnostics']['constant_pixel_convolution'])
        from test_constrained_pixels import reference_pg
        I=result['intensities_internal'][0]
        reference=-2*(sum(reference_pg(value,(.5+prob*I)/1.4,1.4,1.1,300)
                         for value,prob in zip(y[:3],P[:3,0]))+halo*reference_pg(.5,.5/1.4,1.4,1.1,300))
        assert_allclose(result['chi2'],reference,atol=2e-9,rtol=0)


if __name__=='__main__':
    unittest.main(verbosity=2)
