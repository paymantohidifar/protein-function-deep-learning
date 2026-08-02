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


def calc_losses(
    model: nnx.Module,
    train_loader: dict[str, np.ndarray],
    valid_loader: dict[str, np.ndarray],
    max_steps: int = 5
) -> tuple[float, float]:
    """Computes mean training and validation loss over a limited number of batches.

    Args:
        model: Flax NNX model to evaluate.
        train_loader: Iterable of `{"embedding", "target"}` training batches.
        valid_loader: Iterable of `{"embedding", "target"}` validation batches.
        max_steps: Maximum number of batches to average over per loader.

    Returns:
        A `(train_loss, valid_loss)` tuple of mean batch losses.
    """
    train_loss = calc_loss_loader(train_loader, model, num_batches=max_steps)
    valid_loss = calc_loss_loader(valid_loader, model, num_batches=max_steps)

    return train_loss, valid_loss


def calc_loss_loader(
        data_loader: dict[str, np.ndarray],
        model: nnx.Module,
        num_batches: int | None = None
) -> float:
    """Computes mean sigmoid binary cross-entropy loss over batches from a data loader.

    Args:
        data_loader: Iterable of `{"embedding", "target"}` batches.
        model: Flax NNX model to evaluate.
        num_batches: Maximum number of batches to average over. If None,
          iterates over the entire loader.

    Returns:
        Mean loss across the evaluated batches, or `nan` if the loader is empty.
    """
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
    model: nnx.Module,
    batch: dict[str, np.ndarray],
) -> float:
    """Computes sigmoid binary cross-entropy loss for a single batch.

    Args:
        model: Flax NNX model to evaluate.
        batch: A `{"embedding", "target"}` batch.

    Returns:
        Mean loss for the batch.
    """
    # Transfer host memory to JAX device arrays
    x_batch = jnp.asarray(batch["embedding"])
    y_batch = jnp.asarray(batch["target"])

    logits = model(x_batch)
    loss = optax.sigmoid_binary_cross_entropy(logits, y_batch).mean()

    return float(loss)


def calc_classification_metrics_loader(
        model: nnx.Module,
        data_loader: dict[str, np.ndarray],
        threshold: float | None = None,
        max_steps: int | None = None
) -> dict[str, float]:
    """Computes mean classification metrics over batches from a data loader.

    Args:
        model: Flax NNX model to evaluate.
        data_loader: Iterable of `{"embedding", "target"}` batches.
        threshold: Probability threshold used to binarize predictions.
          Defaults to 0.5.
        max_steps: Maximum number of batches to average over. If None,
          iterates over the entire loader.

    Returns:
        A dict mapping metric name (e.g. `"accuracy"`, `"pr_auc"`) to its
        mean value across the evaluated batches.
    """
    if max_steps is None:
        max_steps = len(data_loader)

    if threshold is None:
        threshold = 0.5

    eval_metrics = []
    for i, batch in enumerate(data_loader):
        if i < max_steps:
            eval_metrics.append(calc_batch_metrics(model, batch, threshold))
        else:
            break
    
    return pd.DataFrame(eval_metrics).mean(axis=0).to_dict()


def calc_batch_metrics(
        model: nnx.Module,
        batch: dict[str, np.ndarray],
        threshold: float
) -> dict[str, float]:
    """Computes mean per-target classification metrics for a single batch.

    Args:
        model: Flax NNX model to evaluate.
        batch: A `{"embedding", "target"}` batch.
        threshold: Probability threshold used to binarize predictions.

    Returns:
        A dict mapping metric name to its mean value across targets in the batch.
    """
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
) -> list[dict[str, float]]:
    """Computes classification metrics independently for each row in a batch.

    Args:
        logits: Raw model output logits of shape `(batch_size, num_targets)`.
        targets: Ground-truth binary labels of shape `(batch_size, num_targets)`.
        threshold: Probability threshold used to binarize predictions.

    Returns:
        A list of per-row metric dicts, one per entry in `targets`/`logits`.
    """
    probs = nnx.sigmoid(logits)
    target_metrics = []

    for target, prob in zip(targets, probs):
        target_metrics.append(compute_metrics(target, prob, threshold))

    return target_metrics


def compute_metrics(
        targets: jnp.ndarray,
        probs: jnp.ndarray,
        threshold: float = 0.5
) -> dict[str, float]:
    """Computes accuracy, recall, precision, PR-AUC, and ROC-AUC for one row.

    Args:
        targets: Ground-truth binary labels, shape `(num_targets,)`.
        probs: Predicted probabilities, shape `(num_targets,)`.
        threshold: Probability threshold used to binarize predictions.

    Returns:
        A dict with keys `"accuracy"`, `"recall"`, `"precision"`, `"pr_auc"`,
        and `"roc_auc"`. AUC metrics are set to `0.0` when `targets` contains
        only a single class, since they are undefined in that case.
    """
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
            "pr_auc": 0.0,
            "roc_auc": 0.0,
        }

    # Compute full metric suite when both positive and negative samples exist
    return {
        "accuracy": float(accuracy_score(targets_np, preds_np)),
        "recall": float(recall_score(targets_np, preds_np, zero_division=0.0)),
        "precision": float(
            precision_score(targets_np, preds_np, zero_division=0.0)
        ),
        "pr_auc": float(average_precision_score(targets_np, probs_np)),
        "roc_auc": float(roc_auc_score(targets_np, probs_np)),
    }

def make_coin_flip_predictions(
    valid_true_df: pd.DataFrame, targets: list[str]
) -> pd.DataFrame:
    """Makes random 50/50 coin-flip predictions for each protein function.

    Args:
        valid_true_df: DataFrame whose shape and index the predictions
          should match (one row per protein).
        targets: GO-term column names to generate predictions for.

    Returns:
        A DataFrame of binary predictions with the same index as
        `valid_true_df` and one column per entry in `targets`.
    """
    predictions = np.random.choice([0.0, 1.0], size=valid_true_df.shape)
    return pd.DataFrame(predictions, columns=targets, index=valid_true_df.index)


def make_proportional_predictions(
    valid_true_df: pd.DataFrame, train_df: pd.DataFrame, targets: list[str]
) -> pd.DataFrame:
    """Makes random predictions sampled at each target's training-set base rate.

    Args:
        valid_true_df: DataFrame whose shape and index the predictions
          should match (one row per protein).
        train_df: Training DataFrame used to compute each target's positive rate.
        targets: GO-term column names to generate predictions for.

    Returns:
        A DataFrame of binary predictions with the same index as
        `valid_true_df` and one column per entry in `targets`.
    """
    percent_1_train = dict(train_df[targets].mean())
    proportional_preds = []
    for target_column in targets:
        prob_1 = percent_1_train[target_column]
        prob_0 = 1 - prob_1
        proportional_preds.append(
            np.random.choice([0.0, 1.0], size=len(valid_true_df), p=[prob_0, prob_1])
        )
    return pd.DataFrame(
        np.stack(proportional_preds).T, columns=targets, index=valid_true_df.index
    )