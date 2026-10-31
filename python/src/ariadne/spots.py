"""Spot-pixel proposals; not a Bragg classifier.

propose(): mask-aware multi-scale standardized excess (annulus background).
propose_matched(): radial x tile background, calibrated matched filters and a watershed; the
default detector (see the section comment above propose_matched).

Box sums use separable running filters: O(scales * pixels), not window-area
loops. The standardized statistic is conditional on the estimated background;
threshold scores are not calibrated p-values after adaptive
estimation/multiple scales.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import correlate1d, label, maximum_filter, uniform_filter
from scipy.special import xlogy

try:
    from ._native import spot_proposals as _native_proposals
except ImportError:
    _native_proposals = None
try:
    from ._native import matched_spot_proposals as _native_matched
except ImportError:
    _native_matched = None


def box_sum(array, width):
    return uniform_filter(array, size=width, mode="constant", cval=0.) * width**2


def propose(image, mask, *, gain, offset, readout_variance, noise_model,
            background_width, exclusion_width, scales, score_threshold,
            support_sigma, min_valid_fraction, background_iterations,
            statistic="standardized", component_min_size=4):
    if gain is None or not np.isfinite(gain) or gain <= 0:
        raise ValueError("Supply calibrated gain_adu_per_photon; gain is never inferred")
    if noise_model not in ("poisson", "poisson_gaussian"):
        raise ValueError("Unsupported pixel noise model")
    if readout_variance < 0 or (noise_model == "poisson" and readout_variance != 0):
        raise ValueError("Readout noise requires poisson_gaussian approximation")
    if not scales or any(w < 1 or w % 2 != 1 for w in scales):
        raise ValueError("Use positive odd-pixel scales")
    if (background_width % 2 != 1 or exclusion_width % 2 != 1
            or background_width <= exclusion_width or exclusion_width < max(scales)):
        raise ValueError("Background annulus must exclude all candidate kernel supports")
    if not 0 < min_valid_fraction <= 1 or background_iterations < 0 or score_threshold <= 0 or support_sigma < 0:
        raise ValueError("Invalid threshold settings")
    if statistic not in ("standardized", "deviance"):
        raise ValueError("statistic must be standardized or deviance")
    if type(component_min_size) is not int or component_min_size < 1:
        raise ValueError("component_min_size must be a positive integer")
    data = (np.asarray(image, dtype=float) - offset) / gain
    valid = np.asarray(mask, dtype=bool) & np.isfinite(data)
    if data.ndim != 2 or valid.shape != data.shape:
        raise ValueError("A two-dimensional panel and matching mask are required")
    if noise_model == "poisson" and np.any(data[valid] < 0):
        raise ValueError("Negative corrected pixels contradict the selected Poisson model")
    readout = readout_variance / gain**2
    if _native_proposals is not None:
        return _native_proposals(data,valid,readout,noise_model=='poisson',background_width,
                                 exclusion_width,scales,score_threshold,support_sigma,
                                 min_valid_fraction,background_iterations,
                                 statistic=='standardized',component_min_size)
    retained = valid.copy()
    annulus_area = background_width**2 - exclusion_width**2

    def background(selection):
        weights = selection.astype(float)
        counts = box_sum(weights, background_width) - box_sum(weights, exclusion_width)
        selected_data = np.where(selection, data, 0.)
        sums = box_sum(selected_data, background_width) - box_sum(selected_data, exclusion_width)
        usable = counts >= min_valid_fraction * annulus_area
        mean = np.maximum(np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0), 0.)
        return mean, counts, usable

    for _ in range(background_iterations):
        mean, count, usable = background(retained)
        # Remove positive contaminants only; do not winsorise the measured pixels.
        clipping_variance = mean + readout + (mean + readout) / np.maximum(count, 1.)
        retained &= ~(usable & (data > mean + 4. * np.sqrt(clipping_variance)))
    mean, count, usable = background(retained)
    support = data > mean + support_sigma * np.sqrt(mean + readout)
    if statistic == "standardized" or noise_model == "poisson_gaussian":
        background_variance = (mean + readout) / np.maximum(count, 1.)
        np.copyto(background_variance, 0., where=~valid)
    del count, retained
    # data 与 mean 为本次调用拥有的数组，完成原值计算后直接掩膜。
    np.copyto(data, 0., where=~valid)
    np.copyto(mean, 0., where=~valid)
    best = np.zeros_like(data)
    valid_weights = valid.astype(float)
    for width in scales:
        n_valid = box_sum(valid_weights, width)
        observed = box_sum(data, width)
        expected = box_sum(mean, width)
        enough = n_valid >= width**2 * min_valid_fraction
        if statistic == "standardized":
            variance = expected + readout * n_valid
            variance += box_sum(background_variance, width) * width**2
            statistic_map = np.maximum(observed - expected, 0.) / np.sqrt(np.maximum(variance, .5))
        elif noise_model == "poisson":
            # A finite expected-count floor avoids infinity for zero sampled background.
            # This regularization is an algorithm parameter, not a measured variance.
            expected = np.maximum(expected, .5)
            excess = np.maximum(observed, expected)
            statistic_map = np.sqrt(np.maximum(2. * (xlogy(excess, excess / expected) - excess + expected), 0.))
        else:
            # Quasi-Gaussian excess with annulus background-estimation uncertainty.
            variance = expected + readout * n_valid
            variance += box_sum(background_variance, width) * width**2
            statistic_map = np.maximum(observed - expected, 0.) / np.sqrt(np.maximum(variance, .5))
        np.maximum(best, statistic_map, out=best, where=enough & usable)
    selected = valid & usable & support & (best >= score_threshold)
    if component_min_size > 1 and np.any(selected):
        groups, count = label(selected)
        sizes = np.bincount(groups.ravel(), minlength=count + 1)
        selected &= sizes[groups] >= component_min_size
    return selected


# ---------------------------------------------------------------------------
# Matched-filter proposals
#
# Noise model. In photon units x = (ADU - offset)/gain a background pixel has var(x) = b + r for a
# Poisson expectation b and readout variance r. Real panels deviate: the effective gain is not
# the nominal one, the pedestal may be left in the data and charge sharing correlates neighbours.
# With calibrate=True each frame measures this from its own background pixels: var(x - b) =
# alpha*b + beta is fitted over geometric bins of b (4-sigma clipped second moments, least
# squares weighted by sqrt(count)/variance) and x becomes effective photons (x - o)/alpha with
# o = max(-beta/alpha, 0) and r = max(beta, 0)/alpha^2; a second fit on the refitted background
# refines the first. The correlation rho(d) of standardized residuals at the 12 lags with
# |dy|, |dx| <= 2 gives, for a separable kernel h with autocorrelation a, the variance factor
# kappa_h = 1 + sum_d 2 rho(d) a(dy) a(dx) / a(0)^2 of N = sum h (x - b), and
# kappa_inf = 1 + 2 sum_d rho(d) for sums over many pixels (the background estimate).
#
# Background b(y,x) = P(k) * S(y,x): P is a profile over radial (2-theta) bins k and S a bilinear
# tile modulation, both Poisson maximum-likelihood ratios (sum x / sum expected) over 2x2 cells
# within clip*sigma of the current model, alternated `background_rounds` times. S starts from the
# per-tile median of x/(n P), which strong spots cannot drag; the refit after calibration starts
# from the first solution. A local factor c = sum x / sum b over (2k+1)^2 blocks of 2x2 pixels
# (k = local_cells, two-sided clipped) absorbs structure finer than a tile, e.g. shadow edges and
# diffuse streaks, without letting b go negative; the clipped mean square of the block residuals
# gives an excess-variance factor phi >= 1.
#
# Statistic. N = sum h (x - b) has V = (b + r) phi (kappa_h Q + kappa_inf A^2/n_eff) with
# A = sum_valid h, Q = sum_valid h^2 and n_eff the pixels behind b; z = N/sqrt(V) minus the
# Cornish-Fisher shift of the Poisson skewness at the threshold t_h. The thresholds follow from
# the expected number of local maxima of a smooth Gaussian field above u (Euler characteristic,
# 2-D term), area sqrt(lx ly) u exp(-u^2/2)/(2 pi)^1.5 with lx, ly = 2 (1 - correlation of z at a
# unit step), set to false_alarms/2 per scale. This is a Gaussian-field design target;
# fitted backgrounds and low-count tails require empirical false-alarm validation.
# Pixels with z >= region_threshold on the fine
# scale are split into basins by a union-find watershed (basins merge unless both peaks rise
# `contrast` above the saddle). A basin is kept when it owns a valid 3x3 local maximum with
# z >= t_h on either scale and its flux is not confined to <= 2 adjacent pixels (zinger test).

MATCHED_KERNELS = (np.array([1., 4., 6., 4., 1.])/16,
                   np.array([4., 16., 32., 48., 56., 48., 32., 16., 4.])/256)
NOISE_LAGS = ((0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0), (1, 2), (2, 1), (1, -2), (2, -1),
              (2, 2), (2, -2))
_LOCAL_BLOCK = 2          # 局部统计分块边长，细分块避免强信号污染裁剪
_CORRELATION_STEP = 16    # rows between correlation samples
_NOISE_BINS = 24          # geometric b bins of the variance fit
_NOISE_MIN_COUNT = 200    # pixels for a bin to enter the fit


def radial_bins(coordinate, mask, width=None):
    """Integer bins of a radial coordinate (pixels from the beam centre or 2-theta).

    The default width is the median pixel-to-pixel step of the coordinate over the mask, i.e.
    about one pixel along the radius, so every bin holds a ring of pixels."""
    coordinate = np.asarray(coordinate, dtype=float)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(coordinate)
    if coordinate.ndim != 2 or valid.shape != coordinate.shape:
        raise ValueError("A two-dimensional coordinate map and matching mask are required")
    if not valid.any():
        return np.zeros(coordinate.shape, np.int32)
    if width is None:
        gy, gx = np.gradient(coordinate)
        step = np.hypot(gy, gx)[valid]
        step = step[np.isfinite(step) & (step > 0)]
        width = float(np.median(step)) if step.size else 1.
    if not width > 0:
        raise ValueError("Bin width must be positive")
    bins = np.floor((np.where(np.isfinite(coordinate), coordinate, 0.)-coordinate[valid].min())/width)
    return np.clip(bins, 0, None).astype(np.int32)


def _correlate2(array, h):
    return correlate1d(correlate1d(array, h, axis=0, mode="constant"), h, axis=1, mode="constant")


def _block_sums(array, block):
    H, W = array.shape
    p = np.zeros((-(-H//block)*block, -(-W//block)*block))
    p[:H, :W] = array
    return p.reshape(p.shape[0]//block, block, p.shape[1]//block, block).sum((1, 3))


def _upsample(array, block, shape):
    return np.repeat(np.repeat(array, block, 0), block, 1)[:shape[0], :shape[1]]


def _window_sum(array, k):
    """Zero-padded (2k+1)^2 window sums, added in a fixed order (rows, then columns)."""
    H, W = array.shape
    p = np.pad(array, k)
    rows = p[0:H, :].copy()
    for j in range(1, 2*k+1):
        rows += p[j:j+H, :]
    out = rows[:, 0:W].copy()
    for j in range(1, 2*k+1):
        out += rows[:, j:j+W]
    return out


def _fill_tiles(sn, sd, count, need):
    ny, nx = sn.shape
    ok = (count >= need) & (sd > 0)
    if not ok.any():
        return None, None
    m = np.where(ok, sn/np.where(ok, sd, 1.), 0.)
    c0 = np.where(ok, count, 0.)
    filled = ok.copy()
    while not filled.all():
        f = filled.astype(float)
        pm, pw, pc = np.pad(m*f, 1), np.pad(f, 1), np.pad(c0*f, 1)
        s = sum(pm[1+dy:1+dy+ny, 1+dx:1+dx+nx] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        c = sum(pw[1+dy:1+dy+ny, 1+dx:1+dx+nx] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        sc = sum(pc[1+dy:1+dy+ny, 1+dx:1+dx+nx] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        new = ~filled & (c > 0)
        m = np.where(new, s/np.maximum(c, 1.), m)
        c0 = np.where(new, sc/np.maximum(c, 1.), c0)
        filled |= new
    return m, np.maximum(c0, 1.)


def _interp_tiles(grid, rows, cols, tile):
    """Bilinear interpolation between tile centres at pixel-centre coordinates."""
    ny, nx = grid.shape

    def axis(u, n):
        c = np.clip((u+.5)/tile-.5, 0, n-1)
        i0 = np.minimum(np.floor(c).astype(int), max(n-2, 0))
        return i0, np.minimum(i0+1, n-1), c-i0
    y0, y1, fy = axis(rows, ny)
    x0, x1, fx = axis(cols, nx)
    top = grid[y0][:, x0]*(1-fx)+grid[y0][:, x1]*fx
    bottom = grid[y1][:, x0]*(1-fx)+grid[y1][:, x1]*fx
    return top*(1-fy)[:, None]+bottom*fy[:, None]


def matched_background(x, valid, readout, bins=None, tile=32, clip=3., rounds=2, min_bin=16,
                       warm=None):
    """Background b = P(bin) * S(y,x) (NumPy reference).

    Returns b, the inverse effective pixel count behind b, the kept (unclipped) pixels and the
    cell state (S, P*S per pixel) of the solution, or None when no radial bin or tile holds
    enough pixels. warm=(S, e) starts from a previous solution instead of the tile medians."""
    H, W = x.shape
    xs = _block_sums(np.where(valid, x, 0.), 2).ravel()
    ns = _block_sums(valid.astype(float), 2).ravel()
    ch, cw = -(-H//2), -(-W//2)
    pixel_bins = np.zeros((H, W), np.int64) if bins is None else np.asarray(bins, np.int64)
    flat = pixel_bins[::2, ::2].ravel()
    nb = int(pixel_bins.max())+1
    rows, cols = 2*np.arange(ch)+.5, 2*np.arange(cw)+.5
    tc = tile//2
    ty, tx = -(-ch//tc), -(-cw//tc)

    def tiles(a, fill=0.):
        p = np.full((ty*tc, tx*tc), fill)
        p[:ch, :cw] = a.reshape(ch, cw)
        return p.reshape(ty, tc, tx, tc)

    def profile(keep, S):
        num = np.bincount(flat, np.where(keep, xs, 0.), nb)
        den = np.bincount(flat, np.where(keep, ns*S, 0.), nb)
        cnt = np.bincount(flat, np.where(keep, ns, 0.), nb)
        good = (cnt >= min(min_bin, max(1., cnt.max()))) & (den > 0)
        if not good.any():
            return None, None
        P = np.where(good, num/np.where(good, den, 1.), 0.)
        if not good.all():
            gi, bi = np.flatnonzero(good), np.flatnonzero(~good)
            P[bi] = np.interp(bi, gi, P[gi])
            cnt[bi] = np.interp(bi, gi, cnt[gi])
        return P, cnt

    def clipped(e):
        e = np.maximum(e, 0.)
        return (ns > 0) & (xs <= e+clip*np.sqrt(e+ns*readout)+1.)
    if warm is None:
        S = np.ones(xs.size)
        P, cnt = profile(ns > 0, S)
        if P is None:
            return None
        ok = (ns > 0) & (P[flat] > 0)
        ratio = tiles(np.where(ok, xs/np.where(ok, ns*P[flat], 1.), np.nan), np.nan)
        ratio = ratio.transpose(0, 2, 1, 3).reshape(ty, tx, tc*tc)
        enough = np.isfinite(ratio).sum(-1) >= .25*tc*tc
        median = np.zeros((ty, tx))
        if enough.any():
            median[enough] = np.nanmedian(ratio[enough], axis=-1)
        m, _ = _fill_tiles(median, np.ones((ty, tx)), enough.astype(float), 1.)
        if m is not None:
            S = _interp_tiles(m, rows, cols, tile).ravel()
        keep = clipped(ns*P[flat]*S)
    else:
        S, e = warm[0].copy(), warm[1]
        keep = clipped(ns*e)
    for _ in range(rounds):
        P, cnt = profile(keep, S)
        if P is None:
            return None
        keep = clipped(ns*P[flat]*S)
        m, c0 = _fill_tiles(tiles(np.where(keep, xs, 0.)).sum((1, 3)),
                            tiles(np.where(keep, ns*P[flat], 0.)).sum((1, 3)),
                            tiles(np.where(keep, ns, 0.)).sum((1, 3)), .25*tile*tile)
        if m is None:
            return None
        S = _interp_tiles(m, rows, cols, tile).ravel()
        keep = clipped(ns*P[flat]*S)
    pixel_rows, pixel_cols = np.arange(H), np.arange(W)
    b = P[pixel_bins]*_interp_tiles(m, pixel_rows, pixel_cols, tile)
    inv_neff = 1./np.maximum(cnt[pixel_bins], 1.)+_interp_tiles(1./c0, pixel_rows, pixel_cols, tile)
    keep_pixels = _upsample(keep.reshape(ch, cw), 2, (H, W)) & valid
    return b, inv_neff, keep_pixels, (S, P[flat]*S)


def _noise_fit(x, b, keep, readout):
    """(alpha, beta) of var(x - b) = alpha*b + beta from one pixel per kept 2x2 cell on every
    fourth cell row (pixels (8i, 2j)) in geometric bins of b; None when fewer than two bins hold
    _NOISE_MIN_COUNT pixels or the bins do not span enough of b to separate alpha from beta."""
    s = keep[::8, ::2]
    bs = b[::8, ::2][s]
    R = (x[::8, ::2]-b[::8, ::2])[s]
    positive = bs > 0
    bs, R = bs[positive], R[positive]
    if bs.size < 2*_NOISE_MIN_COUNT:
        return None
    lo, hi = float(bs.min()), float(bs.max())
    q = (hi/lo)**(1./_NOISE_BINS)*(1+1e-12)
    edges = [lo]
    for _ in range(_NOISE_BINS-1):
        edges.append(edges[-1]*q)
    g = np.searchsorted(np.array(edges[1:]), bs, side="right")
    cnt = np.bincount(g, minlength=_NOISE_BINS).astype(float)
    sb = np.bincount(g, bs, minlength=_NOISE_BINS)
    s2 = np.maximum(sb/np.maximum(cnt, 1.)+readout, 1e-6)
    for _ in range(4):
        ok = R*R < 16*s2[g]
        n = np.bincount(g, ok, minlength=_NOISE_BINS)
        s2 = np.maximum(np.bincount(g, np.where(ok, R*R, 0.), minlength=_NOISE_BINS)/np.maximum(n, 1.), 1e-12)
    use = cnt >= _NOISE_MIN_COUNT
    if use.sum() < 2:
        return None
    bm, v = sb[use]/cnt[use], s2[use]
    w2 = cnt[use]/(v*v)
    s00, s01, s11 = np.sum(w2*bm*bm), np.sum(w2*bm), np.sum(w2)
    t0, t1 = np.sum(w2*bm*v), np.sum(w2*v)
    det = s00*s11-s01*s01
    if not det > 1e-2*s00*s11:
        return None
    alpha, beta = (t0*s11-t1*s01)/det, (s00*t1-s01*t0)/det
    return (float(alpha), float(beta)) if alpha > 0 else None


def _noise_correlation(x, valid, b, keep, readout):
    """Correlation of standardized residuals at NOISE_LAGS from kept pixels on every
    _CORRELATION_STEP-th row (0 for a lag with fewer than 1000 pairs)."""
    u = np.where(valid, (x-b)/np.sqrt(np.maximum(np.maximum(b, 0.)+readout, 1e-6)), 0.)
    q = keep & (np.abs(u) < 4)
    H, W = u.shape
    step = _CORRELATION_STEP
    rho = np.zeros(len(NOISE_LAGS))
    for i, (dy, dx) in enumerate(NOISE_LAGS):
        x0, x1 = max(0, -dx), W-max(0, dx)
        a, m = u[0:H-dy:step, x0:x1], q[0:H-dy:step, x0:x1]
        c = u[dy:H:step, x0+dx:x1+dx][:a.shape[0]]
        m = m & q[dy:H:step, x0+dx:x1+dx][:a.shape[0]]
        saa, scc = np.sum(a[m]**2), np.sum(c[m]**2)
        if m.sum() >= 1000 and saa > 0 and scc > 0:
            rho[i] = np.sum(a[m]*c[m])/np.sqrt(saa*scc)
    return rho


def _kernel_noise(h, rho):
    """kappa = var(sum h (x - b))/(var(x) sum h^2) and the unit-step lambdas (y, x) of the
    normalized field for noise correlations rho at NOISE_LAGS."""
    a = np.correlate(h, h, "full")
    c = len(h)-1

    def ac(d):
        return a[c+d] if abs(d) <= c else 0.
    terms = [(0, 0, 1.)]
    for (dy, dx), r in zip(NOISE_LAGS, rho):
        terms += [(dy, dx, r), (-dy, -dx, r)]

    def cov(Dy, Dx):
        total = 0.
        for ey, ex, r in terms:
            total += r*ac(Dy-ey)*ac(Dx-ex)
        return total
    v0 = cov(0, 0)
    return v0/(a[c]*a[c]), (2*(1-cov(1, 0)/v0), 2*(1-cov(0, 1)/v0))


def _rft_threshold(area, lam, target):
    """u with area sqrt(ly lx) u exp(-u^2/2)/(2 pi)^1.5 = target (bisection on [1, 12])."""
    rate = area*np.sqrt(max(lam[0]*lam[1], 0.))/(2*np.pi)**1.5
    lo, hi = 1., 12.
    for _ in range(60):
        u = .5*(lo+hi)
        if rate*u*np.exp(-u*u/2) > target:
            lo = u
        else:
            hi = u
    return .5*(lo+hi)


def _local_scale(xs, ns, es, rho, readout, k, clip=3., iterations=3):
    """Block factor c (sum x / sum b over kept blocks in the window), excess-variance factor phi
    and the inverse kept pixel count, from block sums of x, valid pixels and b."""
    B = _LOCAL_BLOCK
    kc = 1.
    for (dy, dx), r in zip(NOISE_LAGS, rho):
        kc += 2*r*max(B-abs(dy), 0)*max(B-abs(dx), 0)/(B*B)
    valid = ns > 0
    c, phi = np.ones(xs.shape), np.ones(xs.shape)
    for _ in range(iterations):
        e = c*es
        d = xs-e
        keep = valid & (d*d < clip*clip*phi*np.maximum(e+ns*readout, 1e-6)*kc)
        sx, se = _window_sum(np.where(keep, xs, 0.), k), _window_sum(np.where(keep, es, 0.), k)
        c = np.where(se > 1e-9, sx/np.where(se > 1e-9, se, 1.), c)
        e = c*es
        d = xs-e
        nk = _window_sum(keep.astype(float), k)
        su = _window_sum(np.where(keep, d*d/(np.maximum(e+ns*readout, 1e-6)*kc), 0.), k)
        n = np.maximum(nk, 1.)
        phi = np.where(nk >= 8, np.maximum(1., (su/n-np.sqrt(2./n))/.9733), phi)
    return c, phi, 1./np.maximum(_window_sum(np.where(keep, ns, 0.), k), 1.)


def _matched_scores(x, valid, b, inv_neff, readout, phi, kappa, kappa_inf, thresholds):
    """Calibrated z maps for the two kernels (-inf where invalid or less than half a kernel is
    valid)."""
    R = np.where(valid, x-b, 0.)
    vf = valid.astype(float)
    bp = np.maximum(b, 0.)
    s2 = np.maximum(bp+readout, 1e-6)*phi
    out = []
    for h, k, t in zip(MATCHED_KERNELS, kappa, thresholds):
        N, A, Q = _correlate2(R, h), _correlate2(vf, h), _correlate2(vf, h*h)
        V = np.maximum(s2*(k*Q+kappa_inf*A*A*inv_neff), 1e-12)
        g1 = bp*np.sum(h**3)**2/V**1.5
        g2 = bp*np.sum(h**4)**2/V**2
        Z = N/np.sqrt(V)-(g1*(t*t-1)/6+g2*(t**3-3*t)/24-g1*g1*(2*t**3-5*t)/36)
        out.append(np.where(valid & (A > .5), Z, -np.inf))
    return out


def _watershed(score, candidates, contrast):
    H, W = score.shape
    s = score.ravel()
    index = np.flatnonzero(candidates.ravel())
    order = index[np.argsort(-s[index], kind="stable")]
    label_of = np.full(H*W, -1, np.int64)
    parent = {}
    boundary = np.zeros(H*W, bool)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for p in order:
        p = int(p)
        y, x = divmod(p, W)
        roots = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y+dy, x+dx
                if (dy or dx) and 0 <= yy < H and 0 <= xx < W and label_of[yy*W+xx] >= 0:
                    q = find(int(label_of[yy*W+xx]))
                    if q not in roots:
                        roots.append(q)
        if not roots:
            parent[p] = p
            label_of[p] = p
            continue
        main = max(roots, key=lambda q: (s[q], -q))
        for other in roots:
            if other != main:
                if min(s[main], s[other])-s[p] < contrast:
                    parent[other] = main
                else:
                    boundary[p] = True
        label_of[p] = main
    roots = np.full(H*W, -1, np.int64)
    selected = label_of >= 0
    roots[selected] = [find(int(v)) for v in label_of[selected]]
    return roots.reshape(H, W), boundary.reshape(H, W)


def _sharp(x, b, readout, valid, root, fraction, sigma):
    """True when the flux near root sits in <= 2 adjacent pixels: fraction*core - halo exceeds
    sigma standard deviations, halo being the rest of the 3x3 around the residual maximum."""
    H, W = x.shape
    y, xx = divmod(int(root), W)
    best, py, px = -np.inf, y, xx
    for yy in range(max(0, y-1), min(H, y+2)):
        for xq in range(max(0, xx-1), min(W, xx+2)):
            v = x[yy, xq]-b[yy, xq] if valid[yy, xq] else 0.
            if v > best:
                best, py, px = v, yy, xq
    Y0, Y1, X0, X1 = max(0, py-1), min(H, py+2), max(0, px-1), min(W, px+2)
    window = valid[Y0:Y1, X0:X1]
    R = np.where(window, x[Y0:Y1, X0:X1]-b[Y0:Y1, X0:X1], 0.)
    V = np.where(window, np.maximum(x[Y0:Y1, X0:X1], b[Y0:Y1, X0:X1])+readout, 0.)
    cy, cx = py-Y0, px-X0
    second, value = None, -np.inf
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        qy, qx = cy+dy, cx+dx
        if 0 <= qy < R.shape[0] and 0 <= qx < R.shape[1] and R[qy, qx] > value:
            second, value = (qy, qx), R[qy, qx]
    core, vcore = R[cy, cx]+R[second], V[cy, cx]+V[second]
    halo, vhalo = R.sum()-core, V.sum()-vcore
    return fraction*core-halo > sigma*np.sqrt(fraction*fraction*vcore+vhalo)


def _noise_info(gain, offset, G, O, r, rho, kappa, kappa_inf, thresholds):
    g = gain*G
    return dict(gain=g, offset=offset+gain*O, readout_variance=r*g*g,
                correlation=dict(zip(NOISE_LAGS, [float(v) for v in rho])),
                kappa=[float(k) for k in kappa], kappa_inf=float(kappa_inf),
                thresholds=[float(t) for t in thresholds])


def propose_matched(image, mask, *, gain, offset=0., readout_variance=0., bins=None, tile=32,
                    background_rounds=2, clip=3., min_bin=16, calibrate=True, false_alarms=.02,
                    peak_threshold=None, region_threshold=3., contrast=3., zinger_fraction=.5,
                    zinger_sigma=4., local_cells=4, component_min_size=1, info=None):
    """Matched-filter spot pixels (see the section comment above).

    image/offset/readout_variance are in ADU and gain in ADU per photon. They set the starting
    photon scale; with calibrate=True every frame refits the effective gain, offset and readout
    variance and measures the pixel correlations from its background, otherwise they are used
    as given with uncorrelated noise. bins (optional, non-negative integers) assigns each pixel
    to a radial or 2-theta bin, e.g. radial_bins(two_theta, mask); without it the background
    profile is a single bin. false_alarms is the expected number of noise peaks per frame that
    sets the peak thresholds; peak_threshold (z) overrides it. local_cells=0 disables the local
    background factor and zinger_sigma=None or 0 the zinger test. A dict passed as info receives
    the noise model (effective gain, offset, readout_variance, correlation, kappa, kappa_inf)
    and the thresholds used."""
    if gain is None or not np.isfinite(gain) or gain <= 0:
        raise ValueError("Supply gain_adu_per_photon (positive)")
    if not np.isfinite(offset) or not (np.isfinite(readout_variance) and readout_variance >= 0):
        raise ValueError("offset must be finite and readout_variance non-negative")
    if type(tile) is not int or tile < 4 or tile % 2:
        raise ValueError("tile must be an even integer >= 4")
    if type(background_rounds) is not int or background_rounds < 1 or type(min_bin) is not int or min_bin < 1:
        raise ValueError("background_rounds and min_bin must be positive integers")
    if not (np.isfinite(clip) and clip > 0) or not np.isfinite(region_threshold) or not (np.isfinite(contrast) and contrast >= 0):
        raise ValueError("Invalid threshold settings")
    if peak_threshold is not None and not (np.isfinite(peak_threshold) and peak_threshold > 0):
        raise ValueError("peak_threshold must be positive")
    if not (np.isfinite(false_alarms) and false_alarms > 0):
        raise ValueError("false_alarms must be positive")
    zinger_sigma = 0. if zinger_sigma is None else float(zinger_sigma)
    if not (np.isfinite(zinger_fraction) and zinger_fraction > 0) or not (np.isfinite(zinger_sigma) and zinger_sigma >= 0):
        raise ValueError("Invalid zinger settings")
    if type(local_cells) is not int or local_cells < 0:
        raise ValueError("local_cells must be a non-negative integer")
    if type(component_min_size) is not int or component_min_size < 1:
        raise ValueError("component_min_size must be a positive integer")
    data = np.asarray(image, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if data.ndim != 2 or mask.shape != data.shape or min(data.shape) < 3:
        raise ValueError("A two-dimensional panel (>= 3x3) and matching mask are required")
    if bins is not None:
        bins = np.asarray(bins)
        if bins.shape != data.shape or not np.issubdtype(bins.dtype, np.integer) or (bins.size and bins.min() < 0):
            raise ValueError("bins must be non-negative integers with the panel shape")
        if bins.size and bins.max() >= np.iinfo(np.int32).max:
            raise ValueError("bins exceed the supported int32 range")
        bins = bins.astype(np.int32, copy=False)
    readout = readout_variance/gain**2
    if _native_matched is not None:
        out, stats = _native_matched(data, mask, bins, float(gain), float(offset), float(readout), tile,
                                     background_rounds, float(clip), min_bin, int(bool(calibrate)),
                                     float(false_alarms), float(peak_threshold or 0.),
                                     float(region_threshold), float(contrast), float(zinger_fraction),
                                     zinger_sigma, local_cells, component_min_size)
        if info is not None:
            G, O, r, kinf, k0, k1, t0, t1 = stats[:8]
            info.update(_noise_info(gain, offset, G, O, r, stats[8:20], (k0, k1), kinf, (t0, t1)),
                        stage_ms=dict(zip(("load", "background", "calibration", "noise", "scores",
                                           "basins"), stats[20:])))
        return out
    x = (data-offset)/gain
    valid = mask & np.isfinite(x)
    x = np.where(valid, x, 0.)
    r, G, O = readout, 1., 0.
    rho = np.zeros(len(NOISE_LAGS))
    nothing = np.zeros(x.shape, bool)
    fit = matched_background(x, valid, r, bins, tile, clip, background_rounds, min_bin)
    if fit is None:
        if info is not None:
            info.update(_noise_info(gain, offset, G, O, r, rho, (1., 1.), 1., (np.inf, np.inf)))
        return nothing
    b, inv_neff, keep, (S, e) = fit
    xe = x
    if calibrate:
        for it in range(2):
            if it:
                fit = matched_background(xe, valid, r, bins, tile, clip, background_rounds, min_bin,
                                         warm=(S, (e-o)/alpha))
                if fit is None:
                    return nothing
                b, inv_neff, keep, (S, e) = fit
            alpha, beta = _noise_fit(xe, b, keep, r) or (1., r)
            o = max(-beta/alpha, 0.)
            r = max(beta, 0.)/alpha**2
            O, G = O+G*o, G*alpha
            xe = np.where(valid, (x-O)/G, 0.)
            b = (b-o)/alpha
        e = (e-o)/alpha
        rho = _noise_correlation(xe, valid, b, keep, r)
    noise = [_kernel_noise(h, rho) for h in MATCHED_KERNELS]
    kappa = [max(k, .1) for k, _ in noise]
    kappa_inf = max(1+2*rho.sum(), .1)
    phi = 1.
    if local_cells:
        H, W = x.shape
        es = _block_sums(np.where(valid, np.maximum(b, 0.), 0.), _LOCAL_BLOCK)
        c, phi4, inv_n = _local_scale(_block_sums(xe, _LOCAL_BLOCK), _block_sums(valid.astype(float), _LOCAL_BLOCK),
                                      es, rho, r, local_cells)
        b = _upsample(c, _LOCAL_BLOCK, x.shape)*np.maximum(b, 0.)
        phi = _upsample(phi4, _LOCAL_BLOCK, x.shape)
        inv_neff = inv_neff+_upsample(inv_n, _LOCAL_BLOCK, x.shape)
    if peak_threshold is None:
        area = float(valid.sum())
        thresholds = [_rft_threshold(area, lam, false_alarms/len(MATCHED_KERNELS)) for _, lam in noise]
    else:
        thresholds = [float(peak_threshold)]*len(MATCHED_KERNELS)
    if info is not None:
        info.update(_noise_info(gain, offset, G, O, r, rho, kappa, kappa_inf, thresholds))
    Z = _matched_scores(xe, valid, b, inv_neff, r, phi, kappa, kappa_inf, thresholds)
    candidates = valid & (Z[0] >= region_threshold)
    roots, boundary = _watershed(Z[0], candidates, contrast)
    flat = roots.ravel()
    selected = flat >= 0
    ids, inverse = np.unique(flat[selected], return_inverse=True)
    good = np.zeros(len(ids), bool)
    for z, t in zip(Z, thresholds):
        local = valid & (z >= maximum_filter(z, 3, mode="constant", cval=-np.inf))
        best = np.full(len(ids), -np.inf)
        np.maximum.at(best, inverse, np.where(local, z, -np.inf).ravel()[selected])
        good |= best >= t
    if zinger_sigma > 0:
        sigma = zinger_sigma*np.sqrt(kappa[0])
        for i in np.flatnonzero(good):
            good[i] = not _sharp(xe, b, r, valid, ids[i], zinger_fraction, sigma)
    keep = np.zeros(x.size, bool)
    keep[selected] = good[inverse]
    out = candidates & ~boundary & keep.reshape(x.shape)
    if component_min_size > 1 and out.any():
        groups, count = label(out)
        out &= np.bincount(groups.ravel(), minlength=count+1)[groups] >= component_min_size
    return out
