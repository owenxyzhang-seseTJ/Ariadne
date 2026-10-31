#!/usr/bin/env python3
"""Build and run a real Gemmi + Gaussian feedback example, without DIALS/SHELX.

The CIFs and observations are synthetic. This exercises native structure factors,
calibration, held-out scoring, persistence and recovery, not ab-initio solving.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'python/src'))
from ariadne.agent import engine
from ariadne.agent.util import atomic_json
from ariadne.crystal import predict


def write_model(path: Path, perturb: float = 0.):
    import gemmi
    st = gemmi.SmallStructure()
    st.name = 'synthetic_model'
    st.cell = gemmi.UnitCell(16., 18., 20., 90., 96., 90.)
    st.spacegroup_hm = 'P 1'
    st.determine_and_set_spacegroup('hm')
    atoms = [('Cu', .12, .21, .14), ('O', .21, .28, .16), ('O', .04, .14, .12),
             ('N', .32, .32, .19), ('C', .39, .36, .27), ('C', .44, .41, .34)]
    for i, (element, x, y, z) in enumerate(atoms):
        site = gemmi.SmallStructure.Site()
        site.label = element+str(i+1)
        site.element = gemmi.Element(element)
        site.type_symbol = element
        site.fract = gemmi.Fractional(x+(perturb if i == 0 else 0), y, z)
        site.occ = 1.
        site.u_iso = .025
        st.add_site(site)
    st.make_cif_block().write_file(str(path))


def create(directory: Path, seed=20261030, groups=24, run=True):
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    write_model(directory/'model_good.cif')
    write_model(directory/'model_alternative.cif', .085)
    reflections = []
    for i in range(groups):
        hkl = [i+1, i+2, 1+i%3]
        for domain in range(2):
            reflections.append(dict(id=f'g{i:03d}_d{domain}', domain=domain,
                                    hkl=hkl if domain == 0 else [hkl[1], hkl[0], hkl[2]]))
    J = np.asarray(predict([directory/'model_good.cif'], reflections)['predictions'])[:, 0]
    coefficients = np.tile([2.1, .9], groups)
    truth = coefficients*J
    rng = np.random.default_rng(seed)
    blocks = []
    for i in range(groups):
        H = np.array([[1., 1.]]) if i%4 == 0 else np.array([[1., .2], [.15, 1.]])
        mu = H@truth[2*i:2*i+2]
        sigma = np.sqrt(np.maximum(mu, 1.) + 9.)
        R = np.diag(sigma**2)
        y = mu + rng.normal(size=len(mu))*sigma
        blocks.append(dict(id=f'obs_{i:03d}', columns=[2*i,2*i+1], design=H.tolist(), values=y.tolist(),
                           covariance=R.tolist(), raw_ids=[f'synthetic_pixel_group_{i}'],
                           equivalence_ids=[f'synthetic_twin_orbit_{i}']))
    dataset = dict(schema='ariadne-feedback-data/1', measurement_semantics='experimental_signed_information_modes',
                   reflections=reflections, blocks=blocks, preprocessing={'scope':'fixed_physical'})
    atomic_json(directory/'observations.json', dataset)
    request = dict(schema='ariadne-agent/1', mode='custom',
                   inputs={'observations':'observations.json', 'good':'model_good.cif', 'alternative':'model_alternative.cif'},
                   experts=['crystallography','twinning','mof'], nodes=[
        engine.node('audit_good', 'crystal.audit', {'cif':engine.input_ref('good')}),
        engine.node('audit_alternative','crystal.audit',{'cif':engine.input_ref('alternative')}),
        engine.node('feedback','feedback.loop',dict(observations=engine.input_ref('observations'), seed=seed,
            absolute_sd=1., relative_sd=.08, models=[
                {'name':'good','cifs':[engine.input_ref('good')],'provenance':'external'},
                {'name':'alternative','cifs':[engine.input_ref('alternative')],'provenance':'external'}]),
            ['audit_good','audit_alternative']),
        *engine.completion_nodes(['feedback'])])
    atomic_json(directory/'request.json',request)
    summary = dict(seed=seed, groups=groups, latent_intensities=len(reflections),
                   true_coefficients=[2.1,.9], true_fractions=[.7,.3],
                   negative_observations=sum(x < 0 for b in blocks for x in b['values']),
                   data_scope='synthetic CIFs and Gaussian measurements; real Gemmi structure factors')
    if run:
        original = (directory/'observations.json').read_bytes()
        engine.initialize(request, directory/'session', base=directory)
        paused=engine.run(directory/'session',max_steps=2)
        finished=engine.run(directory/'session')
        resumed=engine.run(directory/'session')
        assert paused['actions_used']==2 and finished['actions_used']==5
        assert resumed['actions_used']==finished['actions_used']
        assert original==(directory/'observations.json').read_bytes()
        from ariadne.agent.store import Store
        store=Store(directory/'session')
        result=store.node('feedback')['result'];assert store.verify_events();store.close()
        summary.update(actions=finished['actions_used'],statuses=finished['nodes'],
                       selected_candidate=result['selected_candidate'],calibrated_fractions=result['calibration']['fractions'],
                       validation=result['validation'],final_test=result['final_test'],
                       input_unchanged=True,resume_did_not_repeat=True)
        engine.export_session(directory/'session',directory/'session_export.zip',include_inputs=True)
    atomic_json(directory/'demo_summary.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--seed',type=int,default=20261030)
    p.add_argument('--groups',type=int,default=24)
    p.add_argument('--create-only',action='store_true')
    a=p.parse_args()
    print(create(a.directory,a.seed,a.groups,not a.create_only))
