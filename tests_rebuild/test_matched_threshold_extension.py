"""用完整补丁内扩展检验 DIALS corrected 输入的单位衔接。"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ariadne import spots


def extension_class():
    pytest.importorskip('dials')
    patch = Path(__file__).resolve().parents[1]/'patches/dials-3.30.1-ariadne.patch'
    text = patch.read_text()
    start = text.index('+++ b/src/dials/extensions/ariadne_spotfinder_threshold_ext.py')
    body = text.index('\n', text.index('@@ ', start))+1
    namespace = {}
    exec(compile('\n'.join(line[1:] for line in text[body:].splitlines()),
                 str(patch), 'exec'), namespace)
    return namespace['AriadneThresholdExt']


@pytest.mark.parametrize('calibrate', [False, True])
def test_extension_matches_raw_adu_and_preserves_corrected_input(calibrate):
    from dials.array_family import flex
    from dxtbx.imageset import ImageSetFactory
    from dxtbx.model import DetectorFactory
    from ariadne.threshold_adapter import corrected_to_adu
    cls = extension_class()
    detector = DetectorFactory.simple('PAD', 100., (6., 6.), '+x', '-y', (.1, .1),
                                      (128, 128), (-100., 100000.))
    detector[0].set_gain(5.)
    detector[0].set_pedestal(17.)
    imageset = ImageSetFactory.make_imageset(['unit-frame.cbf'], check_format=False)
    imageset.set_detector(detector)
    rng = np.random.default_rng(42)
    raw = 64.+5.*rng.poisson(12., (128, 128))
    yy, xx = np.indices(raw.shape)
    raw += 2000.*np.exp(-.5*((yy-56.3)**2+(xx-68.6)**2))
    corrected = (raw-17.)/5.
    before = corrected.copy()
    params = cls.phil().extract()
    params.matched.calibrate = calibrate
    params.matched.radial_background = False
    params.offset_adu = 64.
    params.readout_variance_adu = 9.
    ext = cls(SimpleNamespace(spotfinder=SimpleNamespace(threshold=SimpleNamespace(ariadne=params))))
    valid = np.ones(raw.shape, bool)
    actual = ext.compute_threshold(flex.double(corrected), flex.bool(valid), imageset=imageset).as_numpy_array()
    restored, gain = corrected_to_adu(corrected, imageset)
    expected = spots.propose_matched(raw, valid, gain=5., offset=64., readout_variance=9.,
                                     calibrate=calibrate)
    np.testing.assert_allclose(restored, raw, rtol=3e-16, atol=0.)
    assert gain == 5.
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(corrected, before)
