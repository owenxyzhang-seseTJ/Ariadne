"""Convert DIALS threshold input back to detector ADU without changing its data.

DIALS supplies (raw - pedestal) / correction_gain to threshold extensions. The
Ariadne noise parameters describe raw ADU. This temporary conversion is only for
threshold selection; PixelList and integration keep DIALS corrected units.
"""
from __future__ import annotations

import numpy as np


def corrected_to_adu(image, imageset, i_panel=0, region_of_interest=None):
    """Return a separate ADU array and the panel's nominal ADU/photon gain.

    External lookup gain/pedestal maps take precedence over panel constants, as
    in dxtbx ImageSet::get_corrected_data. Threshold callbacks have no frame
    index: sequences have a shared detector, while still images with different
    panel calibrations are rejected instead of silently using frame zero.
    """
    data = np.asarray(image, dtype=np.float64)
    if imageset is None:
        return data.copy(), None
    from dxtbx.imageset import ImageSequence

    detector = imageset.get_detector()
    panel = detector[i_panel]
    nominal_gain = panel.get_gain()
    pedestal = panel.get_pedestal()
    if not isinstance(imageset, ImageSequence):
        signature = [(p.get_gain(), p.get_pedestal()) for p in detector]
        for index in range(1, len(imageset)):
            other = imageset.get_detector(index)
            if [(p.get_gain(), p.get_pedestal()) for p in other] != signature:
                raise ValueError('ADU threshold conversion requires frame-independent detector calibration')

    # dxtbx enables panel gain correction only if *all* gains are positive.
    gains = [p.get_gain() for p in detector]
    correction_gain = nominal_gain if all(g > 0 for g in gains) and any(abs(g-1) > 1e-7 for g in gains) else 1.
    correction_pedestal = pedestal if any(abs(p.get_pedestal()) > 1e-7 for p in detector) else 0.
    for name, fallback in (('gain', correction_gain), ('pedestal', correction_pedestal)):
        lookup = getattr(imageset.external_lookup, name).data
        value = fallback
        if not lookup.empty():
            value = np.asarray(lookup.tile(i_panel).data().as_numpy_array(), dtype=np.float64)
            if region_of_interest is not None:
                x0, x1, y0, y1 = region_of_interest
                value = value[y0:y1, x0:x1]
            if value.shape != data.shape:
                raise ValueError('Detector lookup shape differs from threshold image')
        if not np.all(np.isfinite(value)) or (name == 'gain' and np.any(np.asarray(value) <= 0)):
            raise ValueError('Invalid detector correction '+name)
        if name == 'gain':
            correction_gain = value
        else:
            correction_pedestal = value
    return data * correction_gain + correction_pedestal, nominal_gain
