"""Synthetic suites with ground truth and oracle matched-filter SNR."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment


def grid_case(seed, size=512):
    """Original Ariadne dials_deep generator (photon units, gain 1)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((size, size))
    base = float(rng.choice([1., 4., 10.]))
    readout = float(rng.choice([0., 3., 8.])) if rng.random() < .6 else 0.
    gy, gx = rng.normal(0, .004, 2)
    bg = np.maximum(base + gy*(yy-size/2) + gx*(xx-size/2), .2)
    expectation = bg.copy()
    truth = []
    for y0 in np.linspace(34, size-34, 7):
        for x0 in np.linspace(34, size-34, 7):
            flux = 10*np.exp(rng.uniform(np.log(2), np.log(160)))
            sigma = rng.uniform(.55, 2.5)
            cy, cx = y0+rng.normal(0, 4), x0+rng.normal(0, 4)
            g = np.exp(-.5*((yy-cy)**2+(xx-cx)**2)/sigma**2); g /= g.sum()
            expectation += flux*g
            truth.append(dict(y=cy, x=cx, flux=flux, g=g))
    image = rng.poisson(expectation).astype(float)
    if readout:
        image += rng.normal(0, np.sqrt(readout), image.shape)
    mask = rng.random(image.shape) > .008
    return finish(dict(image=image, mask=mask, gain=1., offset=0., readout_adu=readout,
                       background=bg, truth=truth))


def detector_case(seed, size=512):
    """Crowded, structured case: rings, gaps, beamstop, zingers, anisotropic spots, ADU offset."""
    rng = np.random.default_rng(10_000+seed)
    yy, xx = np.indices((size, size)).astype(float)
    gain = float(rng.choice([1., 25.]))
    offset = 64. if gain > 1 else 0.
    readout_ph = float(rng.choice([0., .05, 1., 4.]))
    readout_adu = readout_ph*gain*gain
    b0 = float(np.exp(rng.uniform(np.log(.3), np.log(12.))))
    cy, cx = size/2+rng.normal(0, 10), size/2+rng.normal(0, 10)
    rad = np.hypot(yy-cy, xx-cx)
    bg = b0*(.6+.8*np.exp(-rad/300.))
    rings = []
    for _ in range(int(rng.integers(0, 3))):
        r0 = rng.uniform(60, 240); a = rng.uniform(.1, .8); w = rng.uniform(1.5, 4.)
        bg += a*b0*np.exp(-.5*((rad-r0)/w)**2); rings.append((r0, w))
    bg *= np.where(rad < 26, .3, 1.)
    expectation = bg.copy()
    mask = np.ones((size, size), bool)
    gy = int(rng.integers(150, 360)); mask[gy:gy+7, :] = False
    gx = int(rng.integers(150, 360)); mask[:, gx:gx+4] = False
    mask[rad < 20] = False
    arm = (np.abs(yy-cy) < 4) & (xx > cx)
    mask[arm] = False
    mask &= rng.random((size, size)) > .002
    truth = []
    positions = []
    tries = 0
    while len(positions) < 150 and tries < 20000:
        tries += 1
        p = rng.uniform(12, size-12, 2)
        if any((p[0]-q[0])**2+(p[1]-q[1])**2 < 36 for q in positions):
            continue
        positions.append(p)
    for py, px in positions:
        flux = float(np.exp(rng.uniform(np.log(8.), np.log(5000.))))
        s1 = rng.uniform(.6, 2.2); s2 = s1*rng.uniform(1., 1.8); th = rng.uniform(0, np.pi)
        c, s = np.cos(th), np.sin(th)
        y0, x0 = int(py), int(px)
        sl = (slice(max(0, y0-12), min(size, y0+13)), slice(max(0, x0-12), min(size, x0+13)))
        dy = yy[sl]-py; dx = xx[sl]-px
        u = c*dx+s*dy; v = -s*dx+c*dy
        g = np.exp(-.5*((u/s2)**2+(v/s1)**2)); g /= g.sum()
        expectation[sl] += flux*g
        full = np.zeros((size, size)); full[sl] = g
        truth.append(dict(y=py, x=px, flux=flux, g=full, sl=sl))
    photons = rng.poisson(expectation).astype(float)
    # zingers and short streaks (not Bragg truth)
    zingers = []
    for _ in range(15):
        y, x = rng.integers(0, size, 2); photons[y, x] += rng.uniform(30, 500); zingers.append((y, x))
    for _ in range(5):
        y, x = rng.integers(0, size-3, 2); v = rng.uniform(30, 200)
        photons[y, x] += v; photons[y, x+1] += v; zingers.append((y, x+.5))
    adu = offset+gain*photons
    if readout_adu:
        adu += rng.normal(0, np.sqrt(readout_adu), adu.shape)
    adu = np.round(adu)  # integer ADU like real detectors
    # drop truth spots largely hidden by mask
    kept = []; ignore = []
    for t in truth:
        sl = t['sl']
        frac = (t['g'][sl]*mask[sl]).sum()
        if frac >= .6:
            kept.append(t)
        else:
            ignore.append((t['y'], t['x']))
    return finish(dict(image=adu, mask=mask, gain=gain, offset=offset, readout_adu=readout_adu,
                       background=bg, truth=kept, ignore=ignore, zingers=zingers, rings=rings, centre=(cy, cx)))


def finish(case):
    b = case['background']; r = case['readout_adu']/case['gain']**2
    m = case['mask']
    for t in case['truth']:
        g = t.pop('g')
        sl = t.pop('sl', None)
        if sl is not None:
            gs, bs, ms = g[sl], b[sl], m[sl]
        else:
            nz = g > 1e-12; gs, bs, ms = g[nz], b[nz], m[nz]
        t['snr'] = float(t['flux']*np.sqrt(np.sum(ms*gs*gs/(bs+r))))
    return case


def components(mask, minimum=1, maximum=1000):
    groups, count = label(mask)
    flat = groups.ravel()
    sizes = np.bincount(flat, minlength=count+1)
    yy, xx = np.indices(mask.shape)
    sy = np.bincount(flat, weights=yy.ravel(), minlength=count+1)
    sx = np.bincount(flat, weights=xx.ravel(), minlength=count+1)
    keep = np.flatnonzero((sizes >= minimum) & (sizes <= maximum)); keep = keep[keep != 0]
    return np.column_stack((sy[keep]/sizes[keep], sx[keep]/sizes[keep]))


SNR_BINS = [0, 3, 5, 7, 10, 20, np.inf]


def score(found, truth, gate=3., ignore=()):
    pts = np.asarray([[t['y'], t['x']] for t in truth])
    pairs = []
    if len(ignore) and len(found):
        # detections of spots mostly hidden by the mask are neither TP nor FP
        ig = np.asarray(ignore, float)
        dmin = np.linalg.norm(found[:, None]-ig[None], axis=2).min(1)
        dtru = np.linalg.norm(found[:, None]-pts[None], axis=2).min(1) if len(pts) else np.full(len(found), np.inf)
        found = found[~((dmin <= 8.) & (dmin < dtru))]
    if len(found) and len(pts):
        d = np.linalg.norm(pts[:, None]-found[None], axis=2)
        r, c = linear_sum_assignment(np.where(d <= gate, d, 1e6))
        pairs = [(i, j, d[i, j]) for i, j in zip(r, c) if d[i, j] <= gate]
    matched = np.zeros(len(truth), bool)
    for i, _, _ in pairs:
        matched[i] = True
    snr = np.asarray([t['snr'] for t in truth])
    return dict(tp=len(pairs), fp=len(found)-len(pairs), fn=len(truth)-len(pairs),
                matched=matched, snr=snr, err=[e for _, _, e in pairs])


def aggregate(rows, pixels):
    tp = sum(r['tp'] for r in rows); fp = sum(r['fp'] for r in rows); fn = sum(r['fn'] for r in rows)
    p = tp/max(1, tp+fp); rc = tp/max(1, tp+fn)
    matched = np.concatenate([r['matched'] for r in rows]); snr = np.concatenate([r['snr'] for r in rows])
    bins = {}
    for lo, hi in zip(SNR_BINS[:-1], SNR_BINS[1:]):
        sel = (snr >= lo) & (snr < hi)
        bins[f'{lo}-{hi}'] = [float(matched[sel].mean()) if sel.any() else None, int(sel.sum())]
    errs = [e for r in rows for e in r['err']]
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=rc, f1=2*p*rc/max(1e-12, p+rc),
                fp_per_mpx=fp/pixels*1e6, recall_by_oracle_snr=bins,
                localization_median_px=float(np.median(errs)) if errs else None)


GENERATORS = dict(grid=grid_case, detector=detector_case)
