"""复合缩放适配、同源误差传播与原生几何输出检查。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'skills/ariadne-reduction/scripts'
sys.path.insert(0, str(SCRIPTS))
from global_scale import equivalent_hkl, prepare, scale, make_corrections


def inputs():
    rng = np.random.default_rng(80)
    config = {'schema': 'ariadne-global-scaling/1', 'basis': 'synthetic-P1',
              'geometry_provenance': 'known synthetic vectors', 'lp_provenance': 'known forward multipliers',
              'symmetry_provenance': 'known P1 anomalous=True', 'integration_provenance': 'known synthetic observations',
              'complete_contributor_audit': True, 'direction_frame': 'crystal_cartesian_common_setting',
              'scan_order': 1, 'absorption_degree': 2, 'phase_relation': 'same_phase_supported',
              'phase_evidence_provenance': 'synthetic known same phase',
              'domains': {str(d): {'space_group': 'P 1', 'anomalous': True} for d in (1,2)}, 'groups': {}}
    records = []
    for j in range(90):
        name = f'g{j}'
        h = j % 6 + 1
        contributors = [{'domain': d, 'hkl': [h, 1, 1]} for d in (1,2)]
        meta = []
        for component in contributors:
            a, b = rng.normal(size=(2,3))
            a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
            meta.append(dict(component, scan_id='scan', phi_rad=(j // 6)/14,
                             lp_forward=1. + .02 * (j % 4), incident_unit_xyz=a.tolist(), outgoing_unit_xyz=b.tolist()))
        config['groups'][name] = meta
        total_only = j % 4 == 0
        records.append({'schema': 'scxrd-pixel-group/1', 'observation_id': name,
                        'status': 'total_only' if total_only else 'joint_components',
                        'quality_flags': [], 'basis': config['basis'], 'corrected': False,
                        'contributors': contributors, 'components': [{'intensity': 1.},{'intensity': 1.}],
                        'total': {'intensity': 2., 'variance': 8.},
                        'covariance_identifiable_subspace': [[4., .5],[.5,4.]],
                        'intensity_definition': 'full-profile amplitude; no further partiality division'})
    blocks, labels, _, _, _ = prepare(records, config)
    beta = np.array([.13, .08, -.06, .03, .04, -.02])
    truth = {}
    for block, record in zip(blocks, records):
        for key in block['equivalence_ids']:
            domain, h, _, _ = json.loads(key)
            truth[key] = (200 if domain == 1 else 90) + h * 8
        amplitudes = np.array([truth[key] for key in block['equivalence_ids']])
        forward = block['known_factors'] * np.exp(block['features'] @ beta)
        values = forward * amplitudes
        for c,v in zip(record['components'],values): c['intensity'] = float(v)
        record['total']['intensity'] = float(values.sum())
    return records, config, beta, truth


class GlobalScaleTests(unittest.TestCase):
    def test_symmetry_keeps_anomalous_pairs_when_declared(self):
        self.assertNotEqual(equivalent_hkl([1,2,3], 'P 1', True), equivalent_hkl([-1,-2,-3], 'P 1', True))
        self.assertEqual(equivalent_hkl([1,2,3], 'P 1', False), equivalent_hkl([-1,-2,-3], 'P 1', False))

    def test_provenance_and_identity_validation(self):
        records, config, _, _ = inputs()
        changed = copy.deepcopy(config)
        changed['groups']['g0'][0]['hkl'] = [99,1,1]
        with self.assertRaises(ValueError): prepare(records, changed)
        changed = copy.deepcopy(config)
        changed['groups']['g0'][0]['incident_unit_xyz'] = [2.,0,0]
        with self.assertRaises(ValueError): prepare(records, changed)
        changed = copy.deepcopy(config)
        changed['complete_contributor_audit'] = False
        with self.assertRaises(ValueError): prepare(records, changed)

    def test_same_data_cross_covariance_and_export(self):
        records, config, beta, truth = inputs()
        with tempfile.TemporaryDirectory(prefix='ariadne_global_') as t:
            root = Path(t)
            integrated, geometry = root/'groups.jsonl',root/'geometry.json'
            integrated.write_text(''.join(json.dumps(r)+'\n' for r in records))
            geometry.write_text(json.dumps(config))
            args = argparse.Namespace(integrated=integrated,geometry=geometry,output=root/'result',
                                      max_iterations=300,tolerance=1e-9,rcond=1e-9,max_parameters=500,max_seconds=60)
            scale(args)
            fit = json.loads((args.output/'scaling.json').read_text())
            self.assertEqual(fit['status'], 'converged', fit.get('reason'))
            np.testing.assert_allclose(fit['beta'], beta, atol=2e-5)
            corrections = json.loads((args.output/'corrections.json').read_text())
            self.assertGreater(len(corrections['fit_only_observations']), 0)
            self.assertNotIn('g0', corrections['groups'])
            group = corrections['groups']['g1']
            f, C = np.array(group['factors']), np.array(records[1]['covariance_identifiable_subspace'])
            Cf, X = np.array(group['factor_covariance']), np.array(group['intensity_factor_cross_covariance'])
            I = np.array([c['intensity'] for c in records[1]['components']])
            expected_var = f @ C @ f + I @ Cf @ I + 2*f @ X @ I
            output = args.output/'corrected/corrected_observations.jsonl'
            corrected = [json.loads(line) for line in output.read_text().splitlines()]
            selected = next(row for row in corrected[1:] if row['observation_id']=='g1')
            self.assertAlmostEqual(selected['sigma']**2, expected_var, places=8)
            self.assertAlmostEqual(selected['intensity'], 200+90+2*2*8, delta=.01)
            self.assertEqual(corrected[0]['intensity_semantics'], 'measured_group_total')
            from export_hklf5 import export
            export(output, root/'scaled.hkl')
            self.assertTrue((root/'scaled.hkl').is_file())
            print('GLOBAL_SCALE_TRUTH', json.dumps({'status':fit['status'], 'beta':fit['beta'],
                  'chi2':fit['chi2'], 'corrected_groups':len(corrected)-1,
                  'fit_only_groups':len(corrections['fit_only_observations'])}))

    def test_native_geometry_identity_and_lp(self):
        from test_centroid_em import DialsAdapterTests
        from dials.array_family import flex
        from scaling_geometry import build
        models = DialsAdapterTests.experiments()
        predictions = flex.reflection_table()
        for eid,e in enumerate(models):
            table = flex.reflection_table.from_predictions(e, margin=0)
            table['id'] = flex.int(len(table),eid)
            predictions.extend(table)
        selected = [0, len(predictions)-1]
        records = []
        for i in selected:
            records.append({'observation_id':f'obs{i}', 'status':'joint_components','quality_flags':[],
                            'contributors':[{'domain':int(predictions['id'][i])+1,'experiment_id':int(predictions['id'][i]),
                                             'prediction_row':i,'hkl':list(predictions['miller_index'][i])}]})
        with tempfile.TemporaryDirectory(prefix='ariadne_geometry_') as t:
            root=Path(t)
            models.as_file(str(root/'models.expt'))
            predictions.as_file(str(root/'pred.refl'))
            (root/'groups.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            evidence={'basis':'P1','geometry_provenance':'synthetic','symmetry_provenance':'synthetic',
                      'complete_contributor_audit':True,'domains':{str(d):{'space_group':'P 1', 'anomalous':True,'experiment_ids':[d-1]} for d in (1,2)}}
            (root/'evidence.json').write_text(json.dumps(evidence))
            build(argparse.Namespace(integrated=root/'groups.jsonl',models=root/'models.expt',predictions=root/'pred.refl',
                                     evidence=root/'evidence.json',output=root/'geometry'))
            config=json.loads((root/'geometry/scaling_geometry.json').read_text())
            predictions.compute_corrections(models)
            for i in selected:
                item=config['groups'][f'obs{i}'][0]
                qe=float(predictions['qe'][i]) if 'qe' in predictions else 1.
                self.assertAlmostEqual(item['lp_forward'],qe/float(predictions['lp'][i]))
                self.assertAlmostEqual(np.linalg.norm(item['incident_unit_xyz']),1.)
                self.assertAlmostEqual(np.linalg.norm(item['outgoing_unit_xyz']),1.)


if __name__ == '__main__':
    unittest.main()
