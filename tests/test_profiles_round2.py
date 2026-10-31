"""Independent box-integral score, face conservation and streaming statistics."""
from __future__ import annotations

from itertools import permutations
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import tracemalloc
import unittest
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills/ariadne-reduction/scripts'))
from profile_geometry import gaussian_voxels, gaussian_voxels_with_derivatives
from profile_learning import signal_moments, align_reference, stream_templates


def score_reference(covariance, centre, lower, order=80):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    coordinates = [lower[d] + (nodes + 1) / 2 - centre[d] for d in range(3)]
    x, y, z = coordinates[0][:, None, None], coordinates[1][None, :, None], coordinates[2][None, None, :]
    p = np.linalg.inv(covariance)
    exponent = p[0, 0]*x*x + p[1, 1]*y*y + p[2, 2]*z*z + 2*p[0, 1]*x*y + 2*p[0, 2]*x*z + 2*p[1, 2]*y*z
    density = np.exp(-exponent / 2) * weights[:, None, None] * weights[None, :, None] * weights[None, None, :]
    density /= 8 * np.sqrt((2*np.pi)**3 * np.linalg.det(covariance))
    return density.sum(), np.array([(density * (p[d, 0]*x + p[d, 1]*y + p[d, 2]*z)).sum() for d in range(3)])


class GaussianFluxTests(unittest.TestCase):
    def test_translation_derivatives_against_original_density_score(self):
        rotation = np.linalg.qr(np.array([[.7, -.4, .8], [.6, .8, -.1], [-.3, .5, .7]]))[0]
        for widths in ([.35, .8, 1.3], [.11, .45, 1.2]):
            covariance = rotation @ np.diag(np.square(widths)) @ rotation.T
            for lower in [(-1, 0, 0), (0, 0, 0), (1, 0, 0)]:
                with self.subTest(widths=widths, lower=lower):
                    reference = score_reference(covariance, [.37, .81, .22], lower)
                    converged = score_reference(covariance, [.37, .81, .22], lower, 104)
                    np.testing.assert_allclose(reference[1], converged[1], rtol=1e-10, atol=2e-12)
                    bounds = tuple(edge for lo in lower for edge in (lo, lo+1))
                    mass, derivative = gaussian_voxels_with_derivatives(covariance, [.37, .81, .22], bounds)
                    np.testing.assert_allclose(mass.item(), converged[0], rtol=2e-9, atol=2e-11)
                    np.testing.assert_allclose(derivative.ravel(), converged[1], rtol=3e-8, atol=2e-10)

    def test_tail_flux_and_marginal_mass(self):
        covariance = np.array([[1., .4, 0], [.4, 1.5, 0], [0, 0, 2.]])
        mass, derivative = gaussian_voxels_with_derivatives(covariance, [0, 0, 0], (9, 10, -14, 15, -12, 13))
        expected_mass = ndtr(-9.) - ndtr(-10.)
        expected_flux = (np.exp(-81/2) - np.exp(-100/2)) / np.sqrt(2*np.pi)
        np.testing.assert_allclose(mass.sum(), expected_mass, rtol=2e-9, atol=0)
        np.testing.assert_allclose(derivative[0].sum(), expected_flux, rtol=2e-9, atol=0)

    def test_subpixel_derivative_has_no_difference_step(self):
        sigma = 1e-4
        centre = [1 + .2*sigma, .5, .5]
        covariance = np.diag([sigma*sigma, .4, .7])
        bounds = (0, 1, 0, 1, 0, 1)
        value, derivative = gaussian_voxels_with_derivatives(covariance, centre, bounds)
        factor = (ndtr(.5/np.sqrt(.4)) - ndtr(-.5/np.sqrt(.4))) * (ndtr(.5/np.sqrt(.7)) - ndtr(-.5/np.sqrt(.7)))
        expected = -np.exp(-.2**2/2) / (np.sqrt(2*np.pi)*sigma) * factor
        np.testing.assert_allclose(derivative[0].item(), expected, rtol=3e-12)
        old_step = .001
        direction = np.array([old_step, 0, 0])
        difference = (gaussian_voxels(covariance, np.array(centre)+direction, bounds) - gaussian_voxels(covariance, np.array(centre)-direction, bounds)) / (2*old_step)
        self.assertGreater(abs(difference.item()/expected - 1), .8)
        self.assertGreater(value.item(), 0)

    def test_wide_diagonal_does_not_cancel_central_probability_or_flux(self):
        sigma = 1e10
        values, derivative = gaussian_voxels_with_derivatives(np.diag([sigma*sigma, 1, 1]), [0, .5, .5], (0, 1, 0, 1, 0, 1))
        transverse = (ndtr(.5)-ndtr(-.5))**2
        np.testing.assert_allclose(values.item(), transverse/(np.sqrt(2*np.pi)*sigma), rtol=3e-15)
        np.testing.assert_allclose(derivative[0].item(), transverse/(2*np.sqrt(2*np.pi)*sigma**3), rtol=3e-15)

    def test_shared_faces_add_up_across_partition(self):
        covariance = np.array([[1.4, .3, .1], [.3, 1.9, .25], [.1, .25, .8]])
        centre = [1.17, -.13, 2.29]
        whole, dw = gaussian_voxels_with_derivatives(covariance, centre, (-3, 5, -4, 3, -2, 6))
        first, df = gaussian_voxels_with_derivatives(covariance, centre, (-3, 1, -4, 3, -2, 6))
        last, dl = gaussian_voxels_with_derivatives(covariance, centre, (1, 5, -4, 3, -2, 6))
        np.testing.assert_allclose(whole, np.concatenate([first,last], axis=2), rtol=3e-13, atol=1e-15)
        np.testing.assert_allclose(dw, np.concatenate([df,dl], axis=3), rtol=3e-12, atol=3e-15)

    def test_axis_permutation_for_vector_derivative(self):
        covariance = np.array([[1.4, .3, .1], [.3, 1.9, .25], [.1, .25, .8]])
        centre, bounds = np.array([.17, .23, -.12]), (-1, 3, -2, 1, 0, 2)
        _, derivative = gaussian_voxels_with_derivatives(covariance, centre, bounds)
        for permutation in permutations(range(3)):
            indices = list(permutation)
            reordered = tuple(edge for axis in indices for edge in bounds[2*axis:2*axis+2])
            _, actual = gaussian_voxels_with_derivatives(covariance[np.ix_(indices,indices)], centre[indices], reordered)
            expected = np.array([derivative[axis].transpose(2,1,0).transpose(permutation).transpose(2,1,0) for axis in indices])
            np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=2e-14)

    def test_bounded_scratch_and_caller_owned_outputs(self):
        covariance = np.array([[1.4, .3, .1], [.3, 1.9, .25], [.1, .25, .8]])
        values, derivative = np.empty((17,29,31)), np.empty((3,17,29,31))
        budget = 256 * 1024
        tracemalloc.start()
        got, dg = gaussian_voxels_with_derivatives(covariance, [15.17,14.23,8.12], (0,31,0,29,0,17),
                                                 out=values, derivative_out=derivative, max_working_bytes=budget)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertIs(got, values)
        self.assertIs(dg, derivative)
        self.assertLess(peak, 1.5*budget + 128*1024)
        expected = gaussian_voxels(covariance,[15.17,14.23,8.12],(0,31,0,29,0,17))
        np.testing.assert_allclose(values,expected,rtol=3e-14,atol=1e-15)


class StreamingProfileTests(unittest.TestCase):
    @staticmethod
    def references():
        rng = np.random.default_rng(31029)
        signals = [rng.uniform(.1,2,(5,7,6)) for _ in range(12)]
        profiles = []
        for index, signal in enumerate(signals):
            centre, covariance = signal_moments(signal)
            profiles.append({'source_row':index,'shape':signal.shape,'centre':centre-.5,
                             'covariance_zyx':covariance,'position_xyz':[index%3,index//3,.3*index]})
        return signals, profiles

    def test_moments_against_dense_coordinate_definition(self):
        signals,_ = self.references()
        for signal in signals[:3]:
            weight = signal/signal.sum()
            grid = np.asarray(np.meshgrid(*[np.arange(n)+.5 for n in signal.shape],indexing='ij'))
            centre = (grid*weight).reshape(3,-1).sum(1)
            delta = grid.reshape(3,-1).T-centre
            covariance = (delta.T*weight.ravel())@delta+np.eye(3)/12
            actual_centre,actual_covariance = signal_moments(signal)
            np.testing.assert_allclose(actual_centre,centre,rtol=1e-15,atol=1e-15)
            np.testing.assert_allclose(actual_covariance,covariance,rtol=3e-14,atol=3e-15)

    def test_alignment_matches_original_dense_trilinear_interpolation(self):
        signals,profiles = self.references()
        for shape in ((9,11,13),(8,10,12)):
            grid = np.asarray(np.meshgrid(*[np.arange(n)-(n-1)/2 for n in shape],indexing='ij')).reshape(3,-1)
            for centre in (profiles[0]['centre'],np.array([2.,3.,2.5]),np.array([2.5,3.5,2.])):
                expected = map_coordinates(signals[0]/signals[0].sum(),grid+centre[:,None],order=1,mode='constant',cval=0).reshape(shape)
                expected /= expected.sum()
                for budget in (4096,8*1024**2):
                    actual = align_reference(signals[0],centre,shape,max_working_bytes=budget)
                    np.testing.assert_allclose(actual,expected,rtol=2e-15,atol=2e-18)

    def test_streamed_pooled_and_shrunk_statistics_match_full_stack(self):
        signals,profiles = self.references()
        shape,grid,shrinkage = (9,11,13),(2,2,1),4.
        aligned = [align_reference(signal,p['centre'],shape) for signal,p in zip(signals,profiles)]
        expected = np.sum(aligned,axis=0);expected/=expected.sum()
        coordinates = np.array([p['position_xyz'] for p in profiles]);low=coordinates.min(0);span=np.maximum(coordinates.max(0)-low,1)
        bins = np.minimum(np.floor((coordinates-low)/span*np.array(grid)).astype(int),np.array(grid)-1)
        with TemporaryDirectory() as temporary:
            calls=[]
            def load(profile):
                calls.append(profile['source_row']);return signals[profile['source_row']]
            name,model = stream_templates(profiles,load,shape,temporary,'profile',grid_xyz=grid,
                                         min_local_references=2,shrinkage_references=shrinkage,max_learning_bytes=200000)
            np.testing.assert_allclose(np.load(Path(temporary)/name),expected,rtol=3e-15,atol=2e-18)
            self.assertEqual(sorted(calls),list(range(len(signals))))
            self.assertEqual(len(calls),len(signals))
            json.dumps(model, allow_nan=False)
            for node in model['nodes']:
                selected = np.flatnonzero(np.all(bins==node['grid_index'],axis=1))
                alpha=len(selected)/(len(selected)+shrinkage)
                local=alpha*np.mean([aligned[i] for i in selected],axis=0)+(1-alpha)*expected
                local/=local.sum()
                np.testing.assert_allclose(np.load(Path(temporary)/node['template']),local,rtol=3e-15,atol=2e-18)
                self.assertEqual(node['source_rows'],selected.tolist())
            self.assertFalse(any(Path(temporary).glob('.profile-sums-*')))


class NativeProfilePrepareTests(unittest.TestCase):
    def test_native_raw_frames_gaussian_flux_and_empirical_derivative_provenance(self):
        try:
            from dials.array_family import flex
            from dxtbx.model import Experiment, ExperimentList
            from dxtbx.model.experiment_list import ExperimentListFactory
        except ImportError:
            self.skipTest('Requires native DIALS/dxtbx raw-image readers')
        from overlap_prepare import prepare
        with TemporaryDirectory() as temporary:
            directory=Path(temporary)
            filenames=[]
            for number in range(1,9):
                path=directory/f'frame_{number:04d}.img'
                header=('{\nHEADER_BYTES= 512;\nDIM= 2;\nBYTE_ORDER= little_endian;\n'
                        'TYPE= unsigned_short;\nSIZE1= 32;\nSIZE2= 32;\nPIXEL_SIZE= 0.1;\n'
                        'DISTANCE= 150;\nWAVELENGTH= 0.7;\nBEAM_CENTER_X= 1.6;\nBEAM_CENTER_Y= 1.6;\n'
                        f'OSC_START= {(number-1)*.1};\nOSC_RANGE= 0.1;\nTIME= 1;\n'+'}\n')
                path.write_bytes(header.encode().ljust(512,b' ')+np.full((32,32),10,dtype='<u2').tobytes())
                filenames.append(str(path))
            first=ExperimentListFactory.from_filenames(filenames)[0]
            second=Experiment(imageset=first.imageset,beam=first.beam,detector=first.detector,
                              scan=first.scan,goniometer=first.goniometer)
            expts=ExperimentList([first,second]);expt_path=directory/'models.expt'
            expts.as_file(str(expt_path))
            predictions=flex.reflection_table()
            predictions['id']=flex.int([0,1]);predictions['panel']=flex.size_t([0,0])
            centres=np.array([[16.3,16.1,3.4],[17.7,16.3,3.4]])
            predictions['xyzcal.px']=flex.vec3_double([tuple(c) for c in centres])
            predictions['miller_index']=flex.miller_index([(1,2,3),(3,2,1)])
            prediction_path=directory/'predictions.refl';predictions.as_file(str(prediction_path))
            covariance=np.array([[1.1,.2,.1],[.2,1.4,.1],[.1,.1,.5]])
            noise={'gain':1.,'readout_variance':4.,'source':'synthetic calibration'}
            entries=[{'experiment_id':0,'panel':0,'covariance_px_xyz':covariance.tolist(),
                      'noise_model':noise,'provenance':{'source_rows':[1]}},
                     {'experiment_id':1,'panel':0,'template':'empirical.npy','template_shape_zyx':[3,3,3],
                      'covariance_px_xyz':np.eye(3).tolist(),'noise_model':noise,'provenance':{'source_rows':[2]}}]
            np.save(directory/'empirical.npy',np.ones((3,3,3))/27,allow_pickle=False)
            library_path=directory/'library.json'
            library_path.write_text(json.dumps({'schema':'scxrd-profile-library/1','entries':entries,
                'complete_prediction_provenance':'synthetic native interface fixture','basis':{'scope':'fixture'},
                'experiment_to_domain':{'0':0,'1':1}}))
            output=directory/'patches'
            args=SimpleNamespace(expt=expt_path,predictions=prediction_path,library=library_path,output=output,
                                 frame_cache=3,template_cache=2,background_margin=1,support_sigma=3.,
                                 max_voxels=100000,max_components=8,position_derivatives=True)
            with redirect_stdout(io.StringIO()):
                prepare(args)
            manifest=json.loads((output/'patches.json').read_text())
            self.assertEqual(len(manifest['groups']),1)
            self.assertEqual(manifest['retained_unprocessed_predictions'],[])
            group=manifest['groups'][0]
            methods=group['position_derivatives']['per_contributor']
            self.assertEqual(methods[0]['method'],'gaussian_boundary_flux')
            self.assertIsNone(methods[0]['finite_difference_step_px'])
            self.assertEqual(methods[1]['method'],'empirical_template_central_difference')
            self.assertEqual(methods[1]['finite_difference_step_px'],.001)
            expected,derivative=gaussian_voxels_with_derivatives(covariance,centres[0],group['bounds_xyz'])
            with np.load(output/group['patch']) as patch:
                np.testing.assert_array_equal(patch['data'],10.)
                np.testing.assert_array_equal(patch['profiles'][0],expected)
                np.testing.assert_array_equal(patch['position_derivatives_xyz'][0],derivative)
                self.assertEqual(patch['position_derivatives_xyz'].shape[:2],(2,3))


if __name__ == '__main__':
    unittest.main()
