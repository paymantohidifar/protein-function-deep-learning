import numpy as np
import pandas as pd
from flax import nnx
import optax
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    average_precision_score,
    roc_auc_score
)


def compute_metrics(
        targets: np.ndarray,
        probs: np.ndarray,
        thresh: float
        ) -> dict[str, float]:

    # Gaurd against division-vy-zero edge case when there are no positive targets
    if np.sum(targets) == 0:
        return{
            m: 0.0 for m in ['accuracy', 'recall', 'precision', 'prc_auc', 'roc_auc']
        }

    return {
        'accuracy': float(accuracy_score(targets, probs >= thresh)),
        'recall': recall_score(targets, probs >= thresh).item(),
        'precision': precision_score(targets, probs >= thresh, zero_division=0.0).item(),
        'prc_auc': average_precision_score(targets, probs).item(),
        'roc_auc': roc_auc_score(targets, probs).item()
    }


def evaluate(
    model: nnx.Module,
    valid_loader: dict[str, np.ndarray],
    thresh
        ):

    eval_metrics = []
    for batch in valid_loader:
        eval_metrics.append(eval_step(model, batch, thresh))

    return pd.DataFrame(eval_metrics).mean(axis=0).to_dict()


def eval_step(
    model:nnx.Module,
    batch: dict[str, np.ndarray],
    thresh: float
     ) -> dict[str, float]:

    logits = model(batch['embedding'])
    loss = optax.sigmoid_binary_cross_entropy(logits, batch['target']).mean()

    target_metrics = calculate_per_target_metrics(logits, batch['target'], thresh)

    metrics_summary = {
        'loss': loss.item(),
        **pd.DataFrame(target_metrics).mean(axis=0).to_dict()
    }

    return metrics_summary


def calculate_per_target_metrics(logits: np.ndarray, targets: np.ndarray, thresh: float) -> list[float]:

    probs = nnx.sigmoid(logits)
    target_metrics = []

    for target, prob in zip(targets, probs):
        target_metrics.append(compute_metrics(target, prob, thresh))

    return target_metrics
