#!/usr/bin/env python3
"""由原生 DIALS 预测生成复合缩放几何侧表，保持贡献者行顺序。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def build(args):
    from dials.array_family import flex
    from dxtbx.model.experiment_list import ExperimentListFactory
    from soft_multi_indexer import raw_key
    from global_scale import write_json
    records = [json.loads(line) for line in args.integrated.read_text().splitlines() if line.strip()]
    models = ExperimentListFactory.from_json_file(str(args.models), check_format=False)
    pred = flex.reflection_table.from_file(str(args.predictions))
    evidence = json.loads(args.evidence.read_text())
    if evidence.get('complete_contributor_audit') is not True:
        raise ValueError('Require explicit complete contributor audit in evidence')
    for key in ('basis', 'domains', 'symmetry_provenance', 'geometry_provenance'):
        if not evidence.get(key):
            raise ValueError('Missing evidence: ' + key)
    if any(not e.crystal or not e.scan or not e.goniometer or e.crystal.num_scan_points or e.beam.num_scan_points or e.goniometer.num_scan_points for e in models):
        raise ValueError('Geometry requires static rotation crystal models')
    pred.compute_corrections(models)
    keys = sorted(set(raw_key(e) for e in models))
    groups = {}
    for record in records:
        if record.get('status') not in ('joint_components', 'total_only') or record.get('quality_flags'):
            continue
        entries = []
        for contributor in record['contributors']:
            eid, row = contributor['experiment_id'], contributor['prediction_row']
            if (type(eid) is not int or not 0 <= eid < len(models) or type(row) is not int or not 0 <= row < len(pred)
                    or pred['id'][row] != eid or list(pred['miller_index'][row]) != contributor['hkl']):
                raise ValueError('Prediction provenance/identity mismatch')
            e = models[eid]
            declared = evidence['domains'][str(contributor['domain'])]
            if 'experiment_ids' not in declared or eid not in declared['experiment_ids']:
                raise ValueError('Explicit domain/experiment mapping required')
            from cctbx import sgtbx
            if sgtbx.space_group_info(declared['space_group']).group() != e.crystal.get_space_group():
                raise ValueError('Declared symmetry disagrees with model')
            lp, qe = float(pred['lp'][row]), float(pred['qe'][row]) if 'qe' in pred else 1.
            if not np.isfinite(lp) or not np.isfinite(qe) or min(lp, qe) <= 0:
                raise ValueError('Invalid native LP/QE correction')
            phi = float(pred['xyzcal.mm'][row][2])
            axis = np.array(e.goniometer.get_rotation_axis_datum())
            R = (np.array(e.goniometer.get_setting_rotation()).reshape(3,3)
                 @ Rotation.from_rotvec(phi * axis).as_matrix()
                 @ np.array(e.goniometer.get_fixed_rotation()).reshape(3,3)
                 @ np.array(e.crystal.get_U()).reshape(3,3))
            incident = R.T @ np.array(e.beam.get_sample_to_source_direction())
            outgoing = R.T @ np.array(pred['s1'][row])
            outgoing /= np.linalg.norm(outgoing)
            entries.append({'domain': contributor['domain'], 'hkl': contributor['hkl'],
                            'scan_id': f'raw-scan-{keys.index(raw_key(e))}', 'phi_rad': phi,
                            'lp_forward': qe/lp, 'incident_unit_xyz': incident.tolist(),
                            'outgoing_unit_xyz': outgoing.tolist()})
        if record['observation_id'] in groups:
            raise ValueError('Duplicate observation id')
        groups[record['observation_id']] = entries
    config = dict(evidence, schema='ariadne-global-scaling/1', groups=groups,
                  direction_frame='crystal_cartesian_common_setting',
                  integration_provenance=str(args.integrated.resolve()),
                  lp_provenance={'method': 'DIALS compute_corrections; forward=qe/lp',
                                 'models': str(args.models.resolve()), 'predictions': str(args.predictions.resolve())})
    if args.output.exists():
        raise ValueError('Output directory already exists')
    args.output.mkdir(parents=True)
    write_json(args.output / 'scaling_geometry.json', config)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('integrated', type=Path)
    p.add_argument('models', type=Path)
    p.add_argument('predictions', type=Path)
    p.add_argument('evidence', type=Path)
    p.add_argument('output', type=Path)
    build(p.parse_args())


if __name__ == '__main__':
    main()
