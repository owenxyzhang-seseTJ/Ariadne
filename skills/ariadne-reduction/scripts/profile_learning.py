"""Bounded-array empirical profile learning with unchanged linear statistics."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.ndimage import affine_transform, map_coordinates


def signal_moments(signal):
    """Centroid and within-box covariance without a 3-by-voxel coordinate grid."""
    signal = np.asarray(signal, dtype=float)
    total = float(signal.sum())
    if signal.ndim != 3 or not np.all(np.isfinite(signal)) or np.any(signal < 0) or not np.isfinite(total) or total <= 0:
        raise ValueError('Require a finite nonnegative 3D reference with positive mass')
    positions = [np.arange(n) + .5 for n in signal.shape]
    marginal = [signal.sum(axis=tuple(d for d in range(3) if d != axis)) / total for axis in range(3)]
    centre = np.array([weights @ coordinate for weights, coordinate in zip(marginal, positions)])
    delta = [positions[axis] - centre[axis] for axis in range(3)]
    covariance = np.diag([marginal[axis] @ (delta[axis]**2) for axis in range(3)])
    for first in range(3):
        for second in range(first + 1, 3):
            other = 3 - first - second
            pair = signal.sum(axis=other) / total
            covariance[first, second] = covariance[second, first] = delta[first] @ pair @ delta[second]
    covariance += np.eye(3) / 12
    return centre, covariance


def align_reference(signal, centre, shape, *, out=None, max_working_bytes=8 * 1024**2):
    """Trilinear translation, exploiting its separable coordinate map.

    Interpolation is linear, so normalization can be deferred until after
    resampling. A diagonal affine pullback needs only per-axis interpolation
    tables, not a 3-by-voxel coordinate grid. A bounded coordinate-tile path is
    retained for exceptionally long axes or small workspace allowances. The
    workspace allowance excludes the supplied signal and returned output;
    compiled SciPy bookkeeping is covered by a conservative per-axis estimate,
    not a process-RSS guarantee.
    """
    signal = np.asarray(signal, dtype=float)
    if max_working_bytes < 4096:
        raise ValueError('Alignment needs at least 4096 working bytes')
    output = np.empty(shape, dtype=float) if out is None else out
    if output.shape != tuple(shape) or output.dtype != np.float64 or not output.flags.c_contiguous:
        raise ValueError('Alignment output must be a contiguous float64 array')
    nz, ny, nx = shape
    centre = np.asarray(centre)
    if 4096 + 128 * sum(shape) <= max_working_bytes:
        # Passing a diagonal vector selects SciPy's separable interpolation
        # path; passing an identity matrix would use its generic slower path.
        affine_transform(signal, np.ones(3), offset=centre - (np.asarray(shape) - 1) / 2,
                         output=output, order=1, mode='constant', cval=0., prefilter=False)
    else:
        target = output.ravel()
        chunk = max(1, min(len(target), int(max_working_bytes) // 80))
        for start in range(0, len(target), chunk):
            stop = min(start + chunk, len(target))
            linear = np.arange(start, stop)
            coordinates = np.empty((3, stop - start))
            coordinates[0] = linear // (ny * nx) - (nz - 1) / 2 + centre[0]
            coordinates[1] = (linear // nx) % ny - (ny - 1) / 2 + centre[1]
            coordinates[2] = linear % nx - (nx - 1) / 2 + centre[2]
            map_coordinates(signal, coordinates, output=target[start:stop], order=1,
                            mode='constant', cval=0., prefilter=False)
    total = float(output.sum())
    if not np.isfinite(total) or total <= 0:
        return None
    output /= total
    return output


def learning_array_bytes(profiles, shape):
    """Conservative live-array estimate, excluding the loaded reflection table."""
    voxels = int(np.prod(shape))
    reference = max(int(np.prod(profile['shape'])) for profile in profiles)
    return 3 * voxels * 8 + 6 * reference * 8


def stream_templates(profiles, signal_loader, shape, directory, prefix, *, grid_xyz=(1, 1, 1),
                     min_local_references=5, shrinkage_references=20., max_learning_bytes=536870912):
    """Learn the same pooled/shrunk means with O(V) resident template arrays.

    Each reference is aligned once. Profiles are grouped by their fixed local
    calibration cell; local sums are spooled to temporary .npy files until the
    pooled mean is known. Final shrinkage uses only a local sum and the pooled
    mean. Temporary storage is O(G V), RAM for aligned templates is O(V), and
    no reference-by-template tensor is retained.
    """
    directory = Path(directory)
    dimensions = np.asarray(grid_xyz, dtype=int)
    covariance = np.mean([profile['covariance_zyx'] for profile in profiles], axis=0)
    coordinates = np.asarray([profile['position_xyz'] for profile in profiles])
    low, high = coordinates.min(axis=0), coordinates.max(axis=0)
    span = np.maximum(high - low, 1.)
    bins = np.minimum(np.floor((coordinates - low) / span * dimensions).astype(int), dimensions - 1)
    groups = defaultdict(list)
    for profile, cell in zip(profiles, bins):
        groups[tuple(cell)].append(profile)
    workspace_bytes = min(8 * 1024**2, max_learning_bytes - learning_array_bytes(profiles, shape))
    if workspace_bytes < 4096:
        raise ValueError('Template learning array budget exceeded')
    total = np.zeros(shape)
    aligned = np.empty(shape)
    local = np.empty(shape) if np.prod(dimensions) > 1 else None
    nodes = []
    with TemporaryDirectory(prefix='.profile-sums-', dir=directory) as temporary:
        for cell, references in sorted(groups.items()):
            if local is not None:
                local.fill(0)
            rows, covariance_sum = [], np.zeros((3, 3))
            for profile in references:
                result = align_reference(signal_loader(profile), profile['centre'], shape,
                                         out=aligned, max_working_bytes=workspace_bytes)
                if result is None:
                    continue
                total += aligned
                if local is not None:
                    local += aligned
                rows.append(profile['source_row'])
                covariance_sum += profile['covariance_zyx']
            if local is not None and len(rows) >= min_local_references:
                path = Path(temporary) / ('_'.join(map(str, cell)) + '.npy')
                np.save(path, local, allow_pickle=False)
                nodes.append({'grid_index': [int(index) for index in cell], 'n_references': len(rows),
                              'source_rows': rows, 'sum_path': path, 'covariance_sum': covariance_sum})
        if not np.isfinite(total.sum()) or total.sum() <= 0:
            raise ValueError('Failed profile alignment')
        total /= total.sum()
        name = prefix + '.npy'
        np.save(directory / name, total, allow_pickle=False)
        # Free alignment buffers before loading one local sum at a time.
        del aligned, local
        if 'result' in locals():
            del result
        for node in nodes:
            count = node['n_references']
            alpha = count / (count + shrinkage_references)
            shrunk = np.load(node.pop('sum_path'), allow_pickle=False)
            shrunk *= alpha / count
            flat, pooled = shrunk.ravel(), total.ravel()
            chunk = max(1, workspace_bytes // 16)
            for start in range(0, len(flat), chunk):
                stop = min(start + chunk, len(flat))
                flat[start:stop] += (1 - alpha) * pooled[start:stop]
            shrunk /= shrunk.sum()
            local_name = prefix + '_node_' + '_'.join(map(str, node['grid_index'])) + '.npy'
            np.save(directory / local_name, shrunk, allow_pickle=False)
            local_covariance = alpha * node.pop('covariance_sum') / count + (1 - alpha) * covariance
            node.update(template=local_name, covariance_px_xyz=local_covariance[::-1, ::-1].tolist(),
                        shrinkage_weight=alpha)
        model = None
        if np.prod(dimensions) > 1:
            model = {'grid_xyz': dimensions.tolist(), 'low_xyz': low.tolist(), 'span_xyz': span.tolist(),
                     'nodes': nodes, 'prior_reference_count': shrinkage_references,
                     'outside_support': 'pooled reference; no local extrapolation',
                     'selection': 'trilinear nonnegative weights; missing nodes use pooled reference'}
    return name, model
