import numpy as np
import pandas as pd
import pytest

from pfdl.evaluate import (
    calc_classification_metrics_loader,
    calc_losses,
    calculate_per_target_metrics,
    compute_metrics,
    make_coin_flip_predictions,
    make_proportional_predictions,
)


class _ConstantLogitModel:
    """Callable stand-in for an nnx.Module that always returns fixed logits."""

    def __init__(self, logits):
        self._logits = np.asarray(logits, dtype=np.float32)

    def __call__(self, x_batch):
        return self._logits[: x_batch.shape[0]]


def test_compute_metrics_perfect_predictions():
    targets = np.array([1.0, 0.0, 1.0, 0.0])
    probs = np.array([0.9, 0.1, 0.8, 0.2])

    metrics = compute_metrics(targets, probs, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_compute_metrics_single_class_sets_auc_to_zero():
    targets = np.array([1.0, 1.0, 1.0])
    probs = np.array([0.9, 0.6, 0.7])

    metrics = compute_metrics(targets, probs, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["pr_auc"] == 0.0
    assert metrics["roc_auc"] == 0.0


def test_calculate_per_target_metrics_returns_one_dict_per_row():
    logits = np.array([[10.0, -10.0], [-10.0, 10.0]])
    targets = np.array([[1.0, 0.0], [0.0, 1.0]])

    target_metrics = calculate_per_target_metrics(logits, targets, threshold=0.5)

    assert len(target_metrics) == 2
    assert all(m["accuracy"] == 1.0 for m in target_metrics)


def test_calc_losses_matches_hand_computed_value():
    # logits == 0 -> sigmoid == 0.5, so BCE == -log(0.5) == log(2) for any label.
    logits = np.array([[0.0, 0.0]])
    targets = np.array([[1.0, 0.0]])
    model = _ConstantLogitModel(logits)
    loader = [{"embedding": np.zeros((1, 4), dtype=np.float32), "target": targets}]

    train_loss, valid_loss = calc_losses(model, loader, loader, max_steps=1)

    expected = np.log(2)
    assert train_loss == pytest.approx(expected, abs=1e-5)
    assert valid_loss == pytest.approx(expected, abs=1e-5)


def test_calc_classification_metrics_loader_returns_mean_metrics():
    logits = np.array([[10.0, -10.0]])
    targets = np.array([[1.0, 0.0]])
    model = _ConstantLogitModel(logits)
    loader = [{"embedding": np.zeros((1, 4), dtype=np.float32), "target": targets}]

    metrics = calc_classification_metrics_loader(model, loader, threshold=0.5, max_steps=1)

    assert metrics["accuracy"] == 1.0


def test_make_coin_flip_predictions_shape_and_binary_values():
    valid_true_df = pd.DataFrame(np.zeros((5, 3)))
    targets = ["GO:1", "GO:2", "GO:3"]

    preds = make_coin_flip_predictions(valid_true_df, targets)

    assert preds.shape == (5, 3)
    assert set(preds.to_numpy().flatten()).issubset({0.0, 1.0})
    assert list(preds.columns) == targets
    assert list(preds.index) == list(valid_true_df.index)


def test_make_proportional_predictions_respects_train_base_rate():
    targets = ["GO:1"]
    train_df = pd.DataFrame({"GO:1": [1.0] * 100})  # always positive in training data
    valid_true_df = pd.DataFrame({"GO:1": [0.0] * 50})

    preds = make_proportional_predictions(valid_true_df, train_df, targets)

    assert (preds["GO:1"] == 1.0).all()
