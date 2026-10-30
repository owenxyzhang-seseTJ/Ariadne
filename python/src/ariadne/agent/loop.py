"""A frozen, auditable structure-feedback experiment, not a self-reinforcing fit."""
from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any

import numpy as np

from ..crystal import predict
from ..feedback import calibrate, canonical_key, closed_split, condition, ensemble_prior
from .util import artifact, artifact_path, atomic_json, new_id, now, read_json


class ExposureLedger:
    """One final-test exposure per declared dataset in this local project registry.

    This is a reproducibility control, not remote attestation. Moving or deleting
    the registry deliberately defeats its scope and is never claimed to be safe.
    """
    def __init__(self, path: Path):
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS exposures(dataset TEXT PRIMARY KEY, frozen TEXT NOT NULL, '
                        'session TEXT NOT NULL, exposed TEXT NOT NULL, result TEXT)')
        self.db.commit()

    def claim(self, dataset: str, frozen: str, session: str):
        try:
            with self.db:
                self.db.execute('INSERT INTO exposures(dataset,frozen,session,exposed) VALUES(?,?,?,?)',
                                (dataset, frozen, session, now()))
        except sqlite3.IntegrityError as exc:
            raise RuntimeError('final test for this dataset has already been exposed in the project registry') from exc

    def complete(self, dataset: str, result_id: str):
        with self.db:
            self.db.execute('UPDATE exposures SET result=? WHERE dataset=?', (result_id, dataset))

    def close(self):
        self.db.close()


def measurement_identity(dataset: dict) -> str:
    """Identify declared raw measurements independently of analysis metadata.

    A new seed, model, preprocessing description or latent naming does not make
    already viewed measurements independent. Raw identities remain declarations;
    this local registry cannot attest to their experimental origin.
    """
    groups = []
    for block in dataset['blocks']:
        raw_ids = block.get('raw_ids')
        if not isinstance(raw_ids, list) or not raw_ids or any(not isinstance(x, str) or not x for x in raw_ids):
            raise ValueError('feedback requires nonempty declared raw pixel identities for every block')
        groups.append(sorted(raw_ids))
    return canonical_key(sorted(groups))


def run_loop(ctx, args: dict[str, Any]) -> dict[str, Any]:
    dataset = read_json(artifact_path(ctx.root, args['observations']))
    if dataset.get('schema') != 'ariadne-feedback-data/1' or dataset.get('measurement_semantics') != 'experimental_signed_information_modes':
        raise ValueError('feedback requires original signed information modes, not model-conditioned intensities')
    reflections, blocks = dataset['reflections'], dataset['blocks']
    n = len(reflections)
    ids = [r['id'] for r in reflections]
    if len(set(ids)) != n or not n:
        raise ValueError('unique nonempty reflection IDs are required')
    split = closed_split(blocks, n, str(args.get('seed', 20261030)))
    measurement_id = measurement_identity(dataset)
    folds = {name: [b for b in blocks if split[b['id']] == name] for name in ('train', 'validation', 'test')}
    train_ids = {b['id'] for b in folds['train']}
    pre = dataset.get('preprocessing', {})
    if pre.get('scope') not in {'external', 'fixed_physical', 'training'}:
        raise ValueError('declare preprocessing training provenance')
    if pre.get('scope') == 'training' and not set(pre.get('fit_observation_ids', [])) <= train_ids:
        raise ValueError('preprocessing used held-out observations')
    if pre.get('scope') != 'training' and pre.get('fit_observation_ids'):
        raise ValueError('external/fixed preprocessing cannot also declare fitted observation IDs')
    models = args['models']
    if not isinstance(models, list) or not 1 <= len(models) <= 32:
        raise ValueError('require 1..32 candidate model ensembles')
    if len({m.get('name') for m in models}) != len(models):
        raise ValueError('candidate names must be unique')
    absolute_sd, relative_sd = args.get('absolute_sd', 1.), args.get('relative_sd', .2)
    predictions, calibrated, validations = [], [], []
    for model in models:
        if not isinstance(model, dict) or set(model)-{'name', 'cifs', 'blocks', 'weights', 'provenance', 'fit_observation_ids'}:
            raise ValueError('unknown candidate model fields')
        if model.get('provenance') not in {'external', 'training'}:
            raise ValueError('candidate requires external or training provenance')
        fit_ids = set(model.get('fit_observation_ids', []))
        if not fit_ids <= train_ids or model['provenance'] == 'external' and fit_ids:
            raise ValueError('candidate training leaks into held-out groups')
        paths = [artifact_path(ctx.root, x) for x in model['cifs']]
        prediction = predict(paths, reflections, blocks=model.get('blocks'))
        if predictions and not np.allclose(prediction['audits'][0]['cell'], predictions[0]['audits'][0]['cell'], rtol=1e-6, atol=1e-6):
            raise ValueError('candidate cells/index settings differ; explicitly reindex before comparison')
        prior = ensemble_prior(prediction['predictions'], model.get('weights'), absolute_sd, relative_sd)
        calibration = calibrate(prior, [r.get('domain', 0) for r in reflections], folds['train'])
        # No posterior from validation is recycled into final-test prediction.
        score = condition(calibration['mean'], calibration['independent_variance'], calibration['factors'], folds['validation'])
        validations.append(dict(name=model['name'], predictive_nll=score['predictive_nll'],
                                data_rank=score['data_rank'], num_observed=score['num_observed']))
        predictions.append(prediction)
        calibrated.append(calibration)
    selected = min(range(len(validations)), key=lambda i: (validations[i]['predictive_nll'], i))
    frozen = dict(schema='ariadne-frozen-structure-candidate/1', candidate=models[selected]['name'],
                  prior=calibrated[selected], split=split, reflections=reflections,
                  data_identity=measurement_id, candidate_artifacts=models[selected]['cifs'],
                  measurement_identity=measurement_id,
                  scoring='joint Gaussian predictive NLL including log determinant',
                  conditioning_scope='fixed profiles, geometry and calibration approximation')
    atomic_json(ctx.directory/'frozen.json', frozen)
    ledger = ExposureLedger(ctx.root.parent/'.ariadne-feedback-ledger.sqlite')
    try:
        # Commit exposure before accessing the test measurements in a likelihood.
        frozen_id = new_id()
        ledger.claim(measurement_id, frozen_id, str(ctx.root))
        ctx.store.event('final_test_exposed', dict(dataset=measurement_id, frozen=frozen_id))
        prior = calibrated[selected]
        test = condition(prior['mean'], prior['independent_variance'], prior['factors'], folds['test'])
        atomic_json(ctx.directory/'test_posterior.json', test)
        result = dict(schema='ariadne-feedback-loop-result/1', selected_candidate=frozen['candidate'],
                      validation=validations, calibration=prior,
                      final_test=dict(predictive_nll=test['predictive_nll'], data_rank=test['data_rank'],
                                      num_observed=test['num_observed']),
                      fold_counts={k: len(v) for k, v in folds.items()}, split=split,
                      frozen=artifact(ctx.root, ctx.directory/'frozen.json', 'frozen_candidate'),
                      posterior=artifact(ctx.root, ctx.directory/'test_posterior.json', 'model_assisted_posterior'),
                      data_identity=measurement_id,
                      measurement_identity=measurement_id,
                      scientific_acceptance='not_established',
                      caveats=['Final-test posterior is not a new experimental observation.',
                               'CIF provenance is declared, not externally attested.',
                               'Calibration/structural cross-covariance is not estimated.',
                               'Prediction uses real IT92 X-ray scattering, without anomalous/solvent terms.'])
        atomic_json(ctx.directory/'feedback_result.json', result)
        ledger.complete(measurement_id, new_id())
        return result
    finally:
        ledger.close()
