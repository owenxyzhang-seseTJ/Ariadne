"""Compiled proposals against the complete reference algorithm, including masks."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from ariadne import spots


@pytest.mark.parametrize('noise',['poisson','poisson_gaussian'])
@pytest.mark.parametrize('iterations',[0,2])
def test_cpp_and_reference_decisions_agree_without_mutating_inputs(monkeypatch,noise,iterations):
    if spots._native_proposals is None:pytest.skip('optional C++ extension is not built')
    rng=np.random.default_rng(915)
    for shape in [(3,7),(64,80),(131,119)]:
        image=rng.poisson(4.,shape).astype(float)
        image[shape[0]//2:shape[0]//2+3,shape[1]//2:shape[1]//2+4]+=40
        mask=rng.random(shape)>.08;image[0,0]=np.nan
        # Strided arrays exercise the real Python/C array boundary.
        image=image[:,::-1];mask=mask[:,::-1]
        before=image.copy();mask_before=mask.copy()
        settings=dict(gain=1.,offset=0.,readout_variance=3. if noise=='poisson_gaussian' else 0.,
                      noise_model=noise,background_width=21,exclusion_width=7,scales=[1,3,5],
                      score_threshold=4.,support_sigma=1.,min_valid_fraction=.5,background_iterations=iterations)
        native=spots.propose(image,mask,**settings)
        with monkeypatch.context() as patch:
            patch.setattr(spots,'_native_proposals',None)
            reference=spots.propose(image,mask,**settings)
        np.testing.assert_array_equal(native,reference)
        np.testing.assert_array_equal(image,before);np.testing.assert_array_equal(mask,mask_before)


def test_compiled_kernel_has_no_cross_call_state_and_handles_empty_masks():
    if spots._native_proposals is None:pytest.skip('optional C++ extension is not built')
    image=np.full((40,50),4.)
    settings=dict(gain=1.,offset=0.,readout_variance=0.,noise_model='poisson',
                  background_width=21,exclusion_width=7,scales=[1,3,5],score_threshold=6.,
                  support_sigma=1.,min_valid_fraction=.5,background_iterations=2)
    masks=[np.zeros(image.shape,bool),np.ones(image.shape,bool)]
    with ThreadPoolExecutor(2) as pool:
        results=list(pool.map(lambda mask:spots.propose(image,mask,**settings),masks))
    assert not any(result.any() for result in results)
