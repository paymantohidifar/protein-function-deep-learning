from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    average_precision_score,
    roc_auc_score
)


def evaluate_model(
    model: nnx.Module,
    train_loader: Dict[str, np.ndarray],
    valid_loader: Dict[str, np.ndarray],
    max_steps: int = 5
) -> Tuple[float, float]:

    train_loss = calc_loss_loader(train_loader, model, num_batches=max_steps)
    valid_loss = calc_loss_loader(valid_loader, model, num_batches=max_steps)

    return train_loss, valid_loss
    

def calc_loss_loader(
        data_loader: Dict[str, np.ndarray],
        model: nnx.Module,
        num_batches: int | None = None
) -> float:

    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, batch in enumerate(data_loader):
        if i < num_batches:
            loss = eval_batch_loss(model, batch)
            total_loss += loss
        else:
            break
    return total_loss / num_batches


def eval_batch_loss(
    model:nnx.Module,
    batch: dict[str, np.ndarray],
     ) -> dict[str, float]:

    # Transfer host memory to JAX device arrays
    x_batch = jnp.asarray(batch["embedding"])
    y_batch = jnp.asarray(batch["target"])

    logits = model(x_batch)
    loss = optax.sigmoid_binary_cross_entropy(logits, y_batch).mean()

    return float(loss)


def calc_classification_metrics_loader(
        model: nnx.Module,
        data_loader: dict[str, np.ndarray],
        threshold: float,
        max_steps: int = 5
):
    eval_metrics = []
    for i, batch in enumerate(data_loader):
        if i < max_steps:
            eval_metrics.append(calc_batch_metrics(model, batch, threshold))
        else:
            break
    
    return pd.DataFrame(eval_metrics).mean(axis=0).to_dict()


def calc_batch_metrics(
        model:nnx.Module,
        batch: dict[str, np.ndarray],
        threshold: float
) -> dict[str, float]:

    # Transfer host memory to JAX device arrays
    x_batch = jnp.asarray(batch["embedding"])
    y_batch = jnp.asarray(batch["target"])

    logits = model(x_batch)
    target_metrics = calculate_per_target_metrics(logits, y_batch, threshold)

    return pd.DataFrame(target_metrics).mean(axis=0).to_dict()


def calculate_per_target_metrics(
        logits: jnp.ndarray,
        targets: jnp.ndarray,
        threshold: float
) -> List[Dict[str, float]]:

    probs = nnx.sigmoid(logits)
    target_metrics = []

    for target, prob in zip(targets, probs):
        target_metrics.append(compute_metrics(target, prob, threshold))

    return target_metrics


def compute_metrics(
        targets: jnp.ndarray,
        probs: jnp.ndarray,
        threshold: float = 0.5
) -> Dict[str, float]:

    # Explicitly pull JAX DeviceArrays to CPU host memory as NumPy arrays
    targets_np = np.asarray(jax.device_get(targets))
    probs_np = np.asarray(jax.device_get(probs))

    # Binarize predictions
    preds_np = probs_np >= threshold

    # Check for single-class edge cases (ROC-AUC / PR-AUC require at least 2 classes)
    unique_classes = np.unique(targets_np)

    if len(unique_classes) < 2:
        # Calculate threshold-dependent metrics safely
        acc = float(accuracy_score(targets_np, preds_np))
        rec = float(recall_score(targets_np, preds_np, zero_division=0.0))
        prec = float(precision_score(targets_np, preds_np, zero_division=0.0))

        # ROC-AUC / PR-AUC are mathematically undefined for single-class subsets; assign 0.0 or float('nan')
        return {
            "accuracy": acc,
            "recall": rec,
            "precision": prec,
            "prc_auc": 0.0,
            "roc_auc": 0.0,
        }

    # Compute full metric suite when both positive and negative samples exist
    return {
        "accuracy": float(accuracy_score(targets_np, preds_np)),
        "recall": float(recall_score(targets_np, preds_np, zero_division=0.0)),
        "precision": float(
            precision_score(targets_np, preds_np, zero_division=0.0)
        ),
        "prc_auc": float(average_precision_score(targets_np, probs_np)),
        "roc_auc": float(roc_auc_score(targets_np, probs_np)),
    }
