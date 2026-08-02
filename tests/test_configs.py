import pytest

from pfdl.configs import esm2_checkpoints, get_esm_checkpoint


def test_get_esm_checkpoint_returns_known_checkpoint():
    assert get_esm_checkpoint("esm2-150M") == "facebook/esm2_t30_150M_UR50D"


@pytest.mark.parametrize("model_name", list(esm2_checkpoints.keys()))
def test_get_esm_checkpoint_covers_all_registered_models(model_name):
    assert get_esm_checkpoint(model_name) == esm2_checkpoints[model_name]


def test_get_esm_checkpoint_raises_on_unknown_model():
    with pytest.raises(KeyError):
        get_esm_checkpoint("not-a-real-model")
