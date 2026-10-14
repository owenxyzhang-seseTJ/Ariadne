"""Fixed trilinear interpolation of shrunk profiles; no per-reflection fit."""
from __future__ import annotations

from itertools import product
import numpy as np


def select_profile(entry, position):
    model = entry.get('local_model')
    if not model or not model['nodes']:
        return entry
    low = np.asarray(model['low_xyz'])
    span = np.asarray(model['span_xyz'])
    dimensions = np.asarray(model['grid_xyz'], dtype=int)
    relative = (np.asarray(position) - low) / span
    if np.any(relative < 0) or np.any(relative > 1):
        return entry  # Return the explicitly supplied pooled profile outside calibration.
    coordinate = np.clip(relative * dimensions - .5, 0, dimensions - 1)
    lower = np.floor(coordinate).astype(int)
    upper = np.minimum(lower + 1, dimensions - 1)
    fraction = coordinate - lower
    nodes = {tuple(n['grid_index']): n for n in model['nodes']}
    parts = {}
    covariance = np.zeros((3, 3))
    provenance_nodes = []
    for bits in product((0, 1), repeat=3):
        cell = tuple(upper[d] if bits[d] else lower[d] for d in range(3))
        weight = float(np.prod([fraction[d] if bits[d] else 1 - fraction[d] for d in range(3)]))
        if weight <= 0:
            continue
        node = nodes.get(cell, entry)
        parts[node['template']] = parts.get(node['template'], 0.) + weight
        covariance += weight * np.asarray(node['covariance_px_xyz'])
        provenance_nodes.append({'grid_index': list(cell), 'weight': weight,
                                 'reference_rows': node.get('source_rows', entry['provenance']['source_rows'])})
    return {**entry, 'template_mixture': [{'template': name, 'weight': w} for name, w in parts.items()],
            'covariance_px_xyz': covariance.tolist(),
            'provenance': {**entry['provenance'], 'local_interpolation': provenance_nodes}}
