"""Targeted ADU decoding/unit contracts and production command wiring."""
from pathlib import Path
from types import SimpleNamespace
import struct
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'skills/ariadne-reduction/scripts'))
from dials_runtime import read_bruker100_adu
from ariadne.agent import backends


def test_signed_underflows_are_absolute_and_overflows_keep_precision(tmp_path):
    header = dict(FORMAT='100', NPIXELB='1 1', NROWS='1', NCOLS='6',
                  NOVERFL='2 2 1', NEXP='1 0 64 0 0', HDRBLKS='5')
    def padded(data):
        return data+b'\0'*((-len(data)) % 16)
    path = tmp_path/'frame.sfrm'
    path.write_bytes(b' '*2560 + bytes([0, 1, 255, 255, 0, 10]) +
                     padded(struct.pack('bb', -5, 47)) +
                     padded(struct.pack('<HH', 500, 65535)) +
                     padded(struct.pack('<I', 4_000_000_000)))
    actual = read_bruker100_adu(path, header)
    np.testing.assert_array_equal(actual, [[-5, 65, 564, 4_000_000_064, 47, 74]])
    assert actual.dtype == np.float64
    with pytest.raises(ValueError, match='Truncated'):
        path.write_bytes(path.read_bytes()[:-1])
        read_bruker100_adu(path, header)


def test_no_underflow_baseline_and_no_baseline_modes(tmp_path):
    path = tmp_path/'frame.sfrm'
    path.write_bytes(b' '*2560+struct.pack('<HH', 0, 500))
    header = dict(FORMAT='100', NPIXELB='2 1', NROWS='1', NCOLS='2',
                  NOVERFL='0 0 0', NEXP='1 0 64 0 0', HDRBLKS='5')
    np.testing.assert_array_equal(read_bruker100_adu(path, header), [[64, 564]])
    header['NOVERFL'] = '-1 0 0'
    np.testing.assert_array_equal(read_bruker100_adu(path, header), [[0, 500]])


def test_matched_options_keep_adu_calibration_and_valid_zero_controls():
    prefix = 'spotfinder.threshold.ariadne.'
    options = {prefix+'algorithm': 'matched', prefix+'offset_adu': -7.,
               prefix+'gain_adu_per_photon': None, prefix+'readout_variance_adu': 4.,
               prefix+'matched.calibrate': False, prefix+'matched.local_cells': 0,
               prefix+'matched.zinger_sigma': 0., prefix+'matched.peak_threshold': None}
    assert backends.typed_options(options, backends.DIALS_OPTIONS['find_spots']) == options
    with pytest.raises(ValueError):
        backends.typed_options({prefix+'matched.local_cells': True}, backends.DIALS_OPTIONS['find_spots'])


def test_production_spots_defaults_and_raw_reader_are_in_same_dispatch(tmp_path, monkeypatch):
    from ariadne.agent.util import artifact
    source = tmp_path/'input.expt'
    source.write_text('{}')
    work = tmp_path/'attempt'
    work.mkdir()
    ctx = SimpleNamespace(root=tmp_path, directory=work, executables={})
    commands = []
    monkeypatch.setattr(backends, 'executable', lambda name, mapping: '/bin/'+name)
    def invoke(ctx, argv):
        commands.append(argv)
        (work/'strong.refl').write_bytes(b'output')
    monkeypatch.setattr(backends, 'invoke', invoke)
    backends.run_dials(ctx, dict(operation='find_spots', inputs=[artifact(tmp_path, source, 'experiments')]))
    argv = commands[0]
    assert argv[:3] == ['/bin/scientific_python',
                        str(ROOT/'skills/ariadne-reduction/scripts/dials_runtime.py'), '/bin/dials.find_spots']
    assert 'spotfinder.threshold.algorithm=ariadne' in argv
    assert 'spotfinder.threshold.ariadne.algorithm=matched' in argv
    assert not any('gain=' in arg for arg in argv)


def test_non_python_dispatcher_fails_before_loading_dxtbx(tmp_path, monkeypatch):
    import dials_runtime
    shell = tmp_path/'dials.import'
    shell.write_text('#!/bin/sh\nexec echo shell-dispatcher\n')
    def forbidden():
        raise AssertionError('must reject dispatcher before installing adapter')
    monkeypatch.setattr(dials_runtime, 'install_bruker_adapter', forbidden)
    with pytest.raises(ValueError, match='requires a Python dispatcher'):
        dials_runtime.main([str(shell)])


def test_mcp_uses_shared_matched_calibration_and_decoder(tmp_path, monkeypatch):
    pytest.importorskip('mcp')
    from ariadne import server
    source = tmp_path/'input.expt'
    source.write_text('{}')
    monkeypatch.setattr(server, 'job_path', lambda job: tmp_path)
    monkeypatch.setattr(backends, 'executable', lambda name, mapping: '/bin/'+name)
    monkeypatch.setattr(server, 'deployment_executables', lambda: {})
    commands = []
    def submit(path, stage, operation, argv, outputs, timeout):
        commands.append(argv)
        return {'status': 'submitted'}
    monkeypatch.setattr(server, 'submit_operation', submit)
    p = 'spotfinder.threshold.ariadne.'
    server.start_operation(str(tmp_path), 'find_spots', ['input.expt'],
                           {p+'matched.calibrate': False, p+'offset_adu': -4.,
                            p+'readout_variance_adu': 0., p+'matched.local_cells': 0,
                            p+'matched.peak_threshold': None})
    assert commands[0][:3] == ['/bin/scientific_python',
        str(ROOT/'skills/ariadne-reduction/scripts/dials_runtime.py'), '/bin/dials.find_spots']
    assert 'spotfinder.threshold.algorithm=ariadne' in commands[0]
    assert p+'algorithm=matched' in commands[0]
    assert p+'matched.peak_threshold=None' in commands[0]
    with pytest.raises(ValueError):
        server.start_operation(str(tmp_path), 'find_spots', ['input.expt'], {p+'matched.local_cells': True})
    # A later pixel-reading stage uses the identical process-local reader.
    (tmp_path/'input.refl').write_bytes(b'reflections')
    server.start_operation(str(tmp_path), 'integrate', ['input.expt', 'input.refl'])
    assert commands[-1][1:3] == [str(ROOT/'skills/ariadne-reduction/scripts/dials_runtime.py'), '/bin/dials.integrate']


def test_mcp_pixel_reading_analyses_share_raw_decoder(tmp_path, monkeypatch):
    pytest.importorskip('mcp')
    from ariadne import server
    monkeypatch.setattr(server, 'job_path', lambda job: tmp_path)
    commands = []
    monkeypatch.setattr(server, 'submit_operation',
                        lambda path, stage, analysis, argv, outputs, timeout: commands.append(argv) or {'status': 'submitted'})
    for name, suffixes in (('learn_profiles', ('.refl', '.json')),
                           ('build_overlaps', ('.expt', '.refl', '.json'))):
        inputs = []
        for index, suffix in enumerate(suffixes):
            file = tmp_path/f'{name}{index}{suffix}'
            file.write_bytes(b'input')
            inputs.append(file.name)
        server.start_analysis(str(tmp_path), name, inputs)
        assert commands[-1][1] == str(ROOT/'skills/ariadne-reduction/scripts/dials_runtime.py')
        assert commands[-1][2].endswith({'learn_profiles': 'learn_profile_library.py',
                                         'build_overlaps': 'overlap_prepare.py'}[name])


def test_corrected_threshold_input_roundtrip_with_gain_pedestal_maps():
    pytest.importorskip('dxtbx')
    from dxtbx.format.image import ImageDouble
    from dxtbx import flumpy
    from ariadne.threshold_adapter import corrected_to_adu
    class Images:
        def __len__(self):
            return 1
        def get_detector(self):
            return [SimpleNamespace(get_gain=lambda: 4., get_pedestal=lambda: 10.)]
    images = Images()
    gain = np.array([[2., 4., 8.], [4., 2., 8.]])
    pedestal = np.array([[3., 5., 7.], [7., 3., 5.]])
    raw = np.array([[-12., 10., 42.], [60., -3., 500.]])
    images.external_lookup = SimpleNamespace(
        gain=SimpleNamespace(data=ImageDouble(flumpy.from_numpy(gain))),
        pedestal=SimpleNamespace(data=ImageDouble(flumpy.from_numpy(pedestal))))
    corrected = (raw-pedestal)/gain
    before = corrected.copy()
    adu, nominal_gain = corrected_to_adu(corrected, images)
    np.testing.assert_array_equal(adu, raw)
    np.testing.assert_array_equal(corrected, before)
    assert nominal_gain == 4.
    adu_roi, _ = corrected_to_adu(corrected[:, 1:], images, region_of_interest=(1, 3, 0, 2))
    np.testing.assert_array_equal(adu_roi, raw[:, 1:])
    # For explicit detector ADU noise calibration, score inputs stay identical.
    physical_gain, offset, readout = 5., -2., 9.
    np.testing.assert_array_equal((adu-offset)/physical_gain, (raw-offset)/physical_gain)
    np.testing.assert_array_equal(np.full_like(adu, readout/physical_gain**2),
                                  np.full_like(raw, readout/physical_gain**2))
