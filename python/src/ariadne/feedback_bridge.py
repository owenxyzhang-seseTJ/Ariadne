"""Convert R2 measured information modes with an explicit physical identity map.

Forward multipliers must include the declared fixed LP/scale/exposure convention.
This adapter never guesses symmetry identities or substitutes model-derived FCFs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from .feedback import observations


def convert(integrated: str | Path, mapping: dict[str, Any]) -> dict[str, Any]:
    if mapping.get('schema') != 'ariadne-feedback-map/1':
        raise ValueError('unsupported feedback map')
    if not mapping.get('basis') or not mapping.get('provenance'):
        raise ValueError('mapping requires basis and physical-correction provenance')
    names, identities, blocks, exclusions = [], {}, [], []
    with Path(integrated).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            key = record['observation_id']
            if record.get('quality_flags'):
                exclusions.append({'id': key, 'reason': 'integration_quality_flags'})
                continue
            if record.get('basis') != mapping['basis'] or record.get('corrected') is not False:
                raise ValueError('adapter requires matching-basis original uncorrected measured amplitudes')
            if record.get('estimator') not in (None, 'signed-wls'):
                raise ValueError('do not reinterpret constrained count estimates as signed Gaussian measurements')
            if record.get('intensity_definition') != 'full-profile amplitude; no further partiality division':
                raise ValueError('unknown intensity/partiality convention')
            meta = mapping.get('groups', {}).get(key)
            if not meta or not meta.get('raw_ids'):
                raise ValueError('declare ordered contributor map and unique raw pixel group IDs')
            contributors = record['contributors']
            if len(meta.get('contributors', [])) != len(contributors):
                raise ValueError('contributor map size mismatch')
            mode = record.get('information_modes')
            if mode:
                if mode.get('schema') != 'ariadne-information-modes/1':
                    raise ValueError('unsupported mode representation')
                H, y, R = np.asarray(mode['design'], float), mode['y'], mode['covariance']
            elif record.get('status') == 'joint_components':
                H = np.eye(len(contributors))
                y = [c['intensity'] for c in record['components']]
                R = record['covariance_identifiable_subspace']
            elif record.get('status') == 'total_only':
                H = np.ones((1, len(contributors)))
                y, R = [record['total']['intensity']], [[record['total']['variance']]]
            else:
                exclusions.append({'id': key, 'reason': 'no_measured_information_modes'})
                continue
            if H.shape != (len(y), len(contributors)):
                raise ValueError('mode/contributor dimensions disagree')
            local_names = []
            for contributor, item in zip(contributors, meta['contributors']):
                if contributor['domain'] != item['domain'] or contributor['hkl'] != item['observed_hkl']:
                    raise ValueError('ordered domain/HKL map mismatch')
                name = item['latent_id']
                reflection = dict(id=name, hkl=item['model_hkl'], domain=item['domain'])
                if 'index_transform' in item:
                    reflection['index_transform'] = item['index_transform']
                if name in identities and identities[name] != reflection:
                    raise ValueError('same latent ID has conflicting physical identity')
                if name not in identities:
                    names.append(name)
                    identities[name] = reflection
                if name not in local_names:
                    local_names.append(name)
            D = np.zeros((len(y), len(local_names)))
            for j, item in enumerate(meta['contributors']):
                factor = item['forward_multiplier']
                if isinstance(factor, bool) or not np.isfinite(factor) or factor <= 0:
                    raise ValueError('positive physical forward multiplier required')
                D[:, local_names.index(item['latent_id'])] += factor*H[:, j]
            blocks.append(dict(id=key, latent_names=local_names, design=D.tolist(), values=y,
                               covariance=R, raw_ids=meta['raw_ids'],
                               equivalence_ids=meta.get('equivalence_ids', [])))
    lookup = {name: j for j, name in enumerate(names)}
    for block in blocks:
        block['columns'] = [lookup[name] for name in block.pop('latent_names')]
    observations(blocks, len(names))
    if not blocks:
        raise ValueError('no usable measured information')
    return dict(schema='ariadne-feedback-data/1', measurement_semantics='experimental_signed_information_modes',
                reflections=[identities[name] for name in names], blocks=blocks,
                preprocessing=mapping.get('preprocessing', {}), exclusions=exclusions,
                provenance=mapping['provenance'],
                comparison_scope='fixed experimental mode representation; not cross-geometry pixel evidence')
