import jax.numpy as jnp
from flax import nnx

from pfdl.models import SimpleMlp


def _make_model(emb_dim=8, num_targets=4, seed=0):
    return SimpleMlp(emb_dim=emb_dim, num_targets=num_targets, rngs=nnx.Rngs(params=seed))


def test_forward_pass_output_shape():
    model = _make_model(emb_dim=8, num_targets=4)
    x = jnp.zeros((3, 8))

    logits = model(x)

    assert logits.shape == (3, 4)


def test_forward_pass_output_dtype():
    model = _make_model(emb_dim=8, num_targets=4)
    x = jnp.zeros((3, 8), dtype=jnp.float32)

    logits = model(x)

    assert logits.dtype == jnp.float32


def test_get_model_name_returns_class_name():
    model = _make_model()
    assert model.get_model_name() == "SimpleMlp"
