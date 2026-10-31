"""Held-out synthetic accuracy regression for the calibrated proposal statistic."""
import numpy as np
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment

from ariadne.spots import propose


def fixture(seed, size=320):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((size, size))
    background = float(rng.choice([1., 4., 10.]))
    readout = float(rng.choice([0., 3., 8.])) if rng.random() < .6 else 0.
    expectation = np.full((size, size), background)
    truth = []
    for y0 in np.linspace(28, size-28, 6):
        for x0 in np.linspace(28, size-28, 6):
            flux = 10*np.exp(rng.uniform(np.log(2), np.log(160)))
            sigma = rng.uniform(.55, 2.5)
            cy, cx = y0+rng.normal(0, 3), x0+rng.normal(0, 3)
            gaussian = np.exp(-.5*((yy-cy)**2+(xx-cx)**2)/sigma**2)
            expectation += flux*gaussian/gaussian.sum()
            truth.append([cy, cx])
    image = rng.poisson(expectation).astype(float)
    if readout:
        image += rng.normal(0, np.sqrt(readout), image.shape)
    return image, rng.random(image.shape) > .008, np.asarray(truth), readout


def components(mask, minimum):
    groups, count = label(mask)
    size = np.bincount(groups.ravel(), minlength=count+1)
    yy, xx = np.indices(mask.shape)
    flat = groups.ravel()
    sum_y = np.bincount(flat, weights=yy.ravel(), minlength=count+1)
    sum_x = np.bincount(flat, weights=xx.ravel(), minlength=count+1)
    keep = np.flatnonzero(size >= minimum)
    keep = keep[keep != 0]
    return np.column_stack((sum_y[keep]/size[keep], sum_x[keep]/size[keep]))


def score(mask, truth, minimum):
    found = components(mask, minimum)
    distances = np.linalg.norm(truth[:, None]-found[None, :], axis=2)
    rows, columns = linear_sum_assignment(distances)
    matched = distances[rows, columns] <= 3
    tp = int(matched.sum())
    fp = len(found)-tp
    fn = len(truth)-tp
    precision = tp/max(1, tp+fp)
    recall = tp/max(1, tp+fn)
    return precision, recall, 2*precision*recall/max(1e-12, precision+recall)


def run(image, mask, readout, statistic, threshold, support, minimum):
    return propose(
        image, mask, gain=1., offset=0., readout_variance=readout,
        noise_model='poisson_gaussian' if readout else 'poisson',
        background_width=31, exclusion_width=7, scales=[1, 3, 5],
        score_threshold=threshold, support_sigma=support,
        min_valid_fraction=.5, background_iterations=0,
        statistic=statistic, component_min_size=minimum)


def test_standardized_statistic_improves_held_out_weak_spot_f1():
    legacy = np.zeros(3)
    standardized = np.zeros(3)
    for seed in (300, 301, 302):
        image, mask, truth, readout = fixture(seed)
        legacy += score(run(image, mask, readout, 'deviance', 6., 1., 3), truth, 3)
        standardized += score(
            run(image, mask, readout, 'standardized', 4., .5, 4), truth, 4)
    assert standardized[2] > legacy[2] + .04
    assert standardized[1] > legacy[1]
    assert standardized[0] > .85


def test_component_size_filter_removes_noise_specks_without_removing_support():
    image = np.full((25, 25), 4.)
    image[10:14, 10:14] += 50.
    image[19, 19] += 80.
    mask = np.ones(image.shape, bool)
    selected = run(image, mask, 0., 'standardized', 2., .5, 4)
    groups, count = label(selected)
    sizes = np.bincount(groups.ravel(), minlength=count+1)[1:]
    assert len(sizes) == 1 and sizes[0] >= 4
