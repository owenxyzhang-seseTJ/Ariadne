#!/usr/bin/env python3
"""Run a DIALS Python dispatcher with process-local Bruker100 ADU decoding.

Use the same runner for import, spotfinding and integration, including expert
workflow commands: dials.python dials_runtime.py /path/to/dials.import ARGS...
No installed dxtbx source, detector geometry or panel calibration is changed.
"""
from __future__ import annotations

from pathlib import Path
import runpy
import sys


def read_bruker100_adu(path, header):
    """Decode FORMAT 100 as signed absolute ADU, retaining overflow precision."""
    import numpy as np

    if header['FORMAT'].strip() != '100':
        raise ValueError('Only Bruker FORMAT 100 is supported by the ADU adapter')
    width, under_width = map(int, header['NPIXELB'].split())
    if width not in (1, 2) or under_width != 1:
        raise ValueError('Unsupported Bruker100 storage width')
    rows, cols = (int(header[key].split()[0]) for key in ('NROWS', 'NCOLS'))
    nu, n2, n4 = map(int, header['NOVERFL'].split())
    if rows <= 0 or cols <= 0 or nu < -1 or n2 < 0 or n4 < 0:
        raise ValueError('Invalid Bruker100 dimensions or table counts')
    with open(path, 'rb') as stream:
        stream.seek(int(header['HDRBLKS'])*512)

        def read(count, dtype, padded=False):
            size = count*np.dtype(dtype).itemsize
            payload = stream.read((size+15) & ~15 if padded else size)
            if len(payload) != ((size+15) & ~15 if padded else size):
                raise ValueError('Truncated Bruker100 pixel/table data')
            return np.frombuffer(payload, dtype, count).astype(np.int64)

        image = read(rows*cols, {1: 'u1', 2: '<u2'}[width])
        zero = image == 0
        under = read(nu, 'i1', True) if nu > 0 else None
        for count, sentinel, dtype in ((n2, 255, '<u2'), (n4, 65535, '<u4')):
            if count:
                values = read(count, dtype, True)
                mask = image == sentinel
                if np.count_nonzero(mask) != count:
                    raise ValueError('Bruker100 overflow count differs from sentinel pixels')
                image[mask] = values
        if nu != -1:
            image += int(header['NEXP'].split()[2])
            if under is not None:
                if np.count_nonzero(zero) != nu:
                    raise ValueError('Bruker100 underflow count differs from zero pixels')
                image[zero] = under
    # Double retains negative i1 and every uint32 ADU exactly in dxtbx.
    return image.reshape(rows, cols).astype(np.float64)


def install_bruker_adapter():
    """Patch only the current process; EXPT keeps the normal format identity."""
    from dxtbx.format.FormatBrukerModern import FormatBrukerModern
    from dxtbx import flumpy

    def get_raw_data(self):
        return flumpy.from_numpy(read_bruker100_adu(self._image_file, self.header_dict))

    FormatBrukerModern.get_raw_data = get_raw_data
    original_detector = FormatBrukerModern._detector

    def detector(self):
        result = original_detector(self)
        if self.header_dict['FORMAT'].strip() == '100':
            # Signed i1 underflows are valid readout samples, not bad-pixel
            # sentinels. dxtbx's trusted-range comparison is strictly > lower.
            for panel in result:
                lower, upper = panel.get_trusted_range()
                panel.set_trusted_range((min(lower, -129.), upper))
        return result

    if not getattr(FormatBrukerModern._detector, '_ariadne_signed_adu', False):
        detector._ariadne_signed_adu = True
        FormatBrukerModern._detector = detector


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise ValueError('usage: dials_runtime.py DIALS_PYTHON_DISPATCHER [ARGS...]')
    dispatcher = Path(argv[0]).resolve(strict=True)
    # dials conda dispatchers have a bash/pythonw shebang and a Python body.
    # Compile before loading scientific code for a clear deployment failure.
    try:
        compile(dispatcher.read_bytes(), str(dispatcher), 'exec')
    except (SyntaxError, ValueError) as exc:
        raise ValueError('The process-local DIALS reader requires a Python dispatcher; '
                         'configure scientific_python and Python DIALS console scripts: '+str(dispatcher)) from exc
    install_bruker_adapter()
    sys.argv = [str(dispatcher), *argv[1:]]
    # DIALS console scripts import their registered command_line module. For
    # companion scripts, retain normal sibling imports (the source is trusted).
    sys.path.insert(0, str(dispatcher.parent))
    runpy.run_path(str(dispatcher), run_name='__main__')


if __name__ == '__main__':
    main()
