import logging

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from torch.utils.data import DataLoader

from pfdl.evaluate import calc_losses, calc_classification_metrics_loader
from pfdl.utils import plot_results
from pfdl.utils import CheckpointClassifier

logger = logging.getLogger(__name__)


@nnx.jit
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    x_batch: jnp.ndarray,
    y_batch: jnp.ndarray
) -> float:
    """Runs one JIT-compiled forward/backward/optimizer-update step.

    Args:
        model: Flax NNX model, updated in place via the optimizer.
        optimizer: Flax NNX optimizer bound to `model`'s trainable params.
        x_batch: Input embeddings of shape `(batch_size, emb_dim)`.
        y_batch: Target labels of shape `(batch_size, num_targets)`.

    Returns:
        The scalar loss value for this batch.
    """
    def loss_fn(model_in: nnx.Module) -> float:

        logits = model_in(x_batch)
        return optax.sigmoid_binary_cross_entropy(logits, y_batch).mean()

    loss_val, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss_val
    

def train_loop(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    num_epochs: int,
    checkpoint_dir: str,
    checkpoint_freq: int = 5,
    checkpoint_best_metric: str = "val_loss",
    eval_freq: int = 5,
    classification_thresh: float = 0.5,
    display_eval_metric: str = "accuracy",
    eval_iter: int = 10,
) -> tuple[list[float], list[float], list[dict[str, float]], list[dict[str, float]], int]:
    """Stateful Flax NNX training loop with Orbax checkpoint restoration and evaluation.

    Args:
        model: Flax NNX model to train, updated in place.
        optimizer: Flax NNX optimizer bound to `model`'s trainable params.
        train_loader: DataLoader yielding `{"embedding", "target"}` training batches.
        valid_loader: DataLoader yielding `{"embedding", "target"}` validation batches.
        num_epochs: Number of epochs to train for.
        checkpoint_dir: Directory (under the checkpoints root) to store/restore
          checkpoints from.
        checkpoint_freq: Save a checkpoint every this many steps.
        checkpoint_best_metric: Metric key used by the checkpoint manager to
          rank checkpoints as "best".
        eval_freq: Compute train/validation loss and metrics every this many steps.
        classification_thresh: Probability threshold used to binarize
          predictions when computing classification metrics.
        display_eval_metric: Metric key printed in the end-of-epoch summary.
        eval_iter: Maximum number of batches used per loss/metric evaluation.

    Returns:
        A `(train_losses, valid_losses, train_metrics, valid_metrics, examples_seen)`
        tuple of per-evaluation-step histories and total examples seen.
    """
    # Instantiate Checkpoint Manager
    ckpt_mgr = CheckpointClassifier(
        checkpoint_dir=checkpoint_dir,
        max_to_keep=3,
        best_metric=checkpoint_best_metric,
        clean_existing=False,
    )

    start_step = 0
    examples_seen = 0
    train_losses, valid_losses = [], []
    train_metrics, valid_metrics = [], []

    # Attempt state restoration
    if ckpt_mgr.mngr.latest_step() is not None:
        try:
            start_step, _, metadata = ckpt_mgr.load_checkpoint(model, optimizer)
            start_step += 1  # Advance past saved step
            examples_seen = metadata.get("examples_seen", 0)
            train_losses = metadata.get("train_losses_history", [])
            valid_losses = metadata.get("valid_losses_history", [])
            train_metrics = metadata.get("train_metrics_history", [])
            valid_metrics = metadata.get("valid_metrics_history", [])
            logger.info(
                "[Orbax] Resumed training at Step %d (Epoch %s)",
                start_step, metadata.get('epoch', '?')
            )
        except Exception as e:
            logger.warning("[Warning] Failed to load checkpoint: %s. Starting fresh.", e)
    else:
        logger.info("[Orbax] No existing checkpoints found. Initializing fresh run.")

    global_step = 0

    # Initialize tracking variables to avoid UnboundLocalErrors
    latest_train_metric: dict[str, float] | None = None
    latest_valid_metric: dict[str, float] | None = None
    latest_train_loss: float = float("nan")
    latest_valid_loss: float = float("nan")

    for epoch in range(num_epochs):
        for batch in train_loader:
            # Efficient step skipping during resume
            if global_step < start_step:
                global_step += 1
                continue

            # Stream memory to JAX device arrays
            x_batch = jnp.asarray(batch["embedding"])
            y_batch = jnp.asarray(batch["target"])
            batch_size = x_batch.shape[0]

            examples_seen += batch_size

            # Execute JIT-compiled optimization step
            _ = train_step(model, optimizer, x_batch, y_batch)

            # Periodic Evaluation
            if global_step % eval_freq == 0:
                latest_train_loss, latest_valid_loss = calc_losses(
                    model, train_loader, valid_loader, max_steps=eval_iter
                )

                latest_train_metric = calc_classification_metrics_loader(
                    model, train_loader, threshold=classification_thresh, max_steps=eval_iter
                )
                latest_valid_metric = calc_classification_metrics_loader(
                    model, valid_loader, threshold=classification_thresh, max_steps=eval_iter
                )

                train_losses.append(latest_train_loss)
                valid_losses.append(latest_valid_loss)
                train_metrics.append(latest_train_metric)
                valid_metrics.append(latest_valid_metric)

                logger.info(
                    "Epoch [%03d/%03d] (Step %06d) | Train Loss: %.4f | Valid Loss: %.4f",
                    epoch + 1, num_epochs, global_step, latest_train_loss, latest_valid_loss
                )

            # Periodic Checkpointing
            if global_step > 0 and global_step % checkpoint_freq == 0 and latest_valid_metric is not None:
                logger.info("[Orbax] Step %d: Persisting checkpoint...", global_step)
                step_metrics = {
                    "val_loss": latest_valid_loss,
                    "val_accuracy": latest_valid_metric.get("accuracy", 0.0),
                    "val_pr_auc": latest_valid_metric.get("pr_auc", 0.0),
                    "val_roc_auc": latest_valid_metric.get("roc_auc", 0.0),
                }

                step_metadata = {
                    "model": model.get_model_name(),
                    "epoch": epoch + 1,
                    "examples_seen": examples_seen,
                    "train_losses_history": train_losses,
                    "valid_losses_history": valid_losses,
                    "train_metrics_history": train_metrics,
                    "valid_metrics_history": valid_metrics
                }

                ckpt_mgr.save_checkpoint(
                    step=global_step,
                    model=model,
                    optimizer=optimizer,
                    step_metrics=step_metrics,
                    step_metadata=step_metadata,
                )

            global_step += 1

        # Safe end-of-epoch metric display
        if latest_train_metric and latest_valid_metric:
            tr_metric = latest_train_metric.get(display_eval_metric, 0.0)
            val_metric = latest_valid_metric.get(display_eval_metric, 0.0)
            logger.info(
                "Epoch %03d Complete | Train %s: %.2f | Valid %s: %.2f",
                epoch + 1, display_eval_metric, tr_metric, display_eval_metric, val_metric
            )

    # Compute final metrics post-training loop
    final_train_loss, final_valid_loss = calc_losses(
        model, train_loader, valid_loader, max_steps=eval_iter
    )
    final_valid_metric = calc_classification_metrics_loader(
        model, valid_loader, threshold=classification_thresh, max_steps=eval_iter
    )

    final_metrics = {
        "val_loss": final_valid_loss,
        "val_accuracy": final_valid_metric.get("accuracy", 0.0),
        "val_pr_auc": final_valid_metric.get("pr_auc", 0.0),
        "val_roc_auc": final_valid_metric.get("roc_auc", 0.0),
    }

    final_metadata = {
        "model": model.get_model_name(),
        "epoch": num_epochs,
        "examples_seen": examples_seen,
        "train_losses_history": train_losses,
        "valid_losses_history": valid_losses,
        "train_metrics_history": train_metrics,
        "valid_metrics_history": valid_metrics,
        "Status": "Complete"
    }

    # Save final state
    logger.info("[Orbax] Training complete. Persisting final model checkpoint...")
    ckpt_mgr.save_checkpoint(
        step=global_step,
        model=model,
        optimizer=optimizer,
        step_metrics=final_metrics,
        step_metadata=final_metadata,
    )

    # Block until background Orbax I/O operations complete
    ckpt_mgr.close()

    return train_losses, valid_losses, train_metrics, valid_metrics, examples_seen


def run_train(
        model: type[nnx.Module],
        dataset_splits: dict[str, DataLoader],
        lr: float = 1e-3,
        num_epochs: int = 5,
        eval_freq: int = 20,
        eval_iter: int = 50,
        checkpoint_dir: str = "classifier-checkpoints",
        checkpoint_every: int = 2,
        binary_threshold: float = 0.5,
        metrics_plot_title: str = "Model",
        metrics_plots_dir: str = "test"
) -> str:
    """Instantiates a model, trains it, and plots the resulting metrics.

    Args:
        model: Model class to instantiate (e.g. `SimpleMlp`), called as
          `model(emb_dim, num_targets, rngs)`.
        dataset_splits: Dict with `"train"`/`"valid"` DataLoaders, as
          returned by `pfdl.data.build_dataset`.
        lr: Adam learning rate.
        num_epochs: Number of epochs to train for.
        eval_freq: Compute train/validation loss and metrics every this many steps.
        eval_iter: Maximum number of batches used per loss/metric evaluation.
        checkpoint_dir: Directory (under the checkpoints root) to store/restore
          checkpoints from.
        checkpoint_every: Save a checkpoint every this many steps.
        binary_threshold: Probability threshold used to binarize predictions
          when computing classification metrics.
        metrics_plot_title: Title for the saved metrics plot.
        metrics_plots_dir: Subdirectory to save the metrics plot under.

    Returns:
        Path to the saved metrics plot image.
    """
    # Initialize the functional random state using a base seed
    main_key = jax.random.PRNGKey(42)
    # Split the key: one for current parameter initialization, one reserved for future entropy
    model_key, main_key = jax.random.split(main_key, 2)

    # Get the model input and output dimensions
    dataset_split_batch = next(iter(dataset_splits['train']))
    emb_dim = dataset_split_batch['embedding'].shape[1]
    num_targets = dataset_split_batch['target'].shape[1]

    # Instantiate the model architecture
    model_rng = nnx.Rngs(params=model_key)
    training_model = model(emb_dim, num_targets, model_rng)

    # Bind the model parameters to an Optax-backed stateful optimizer
    optax_tx = optax.adam(learning_rate=lr)
    # wrt=nnx.Param instructs the optimizer to track specifically the trainable Param variables
    optimizer = nnx.Optimizer(training_model, optax_tx, wrt=nnx.Param)

    history = train_loop(
        model=training_model,
        optimizer=optimizer,
        train_loader=dataset_splits['train'],
        valid_loader=dataset_splits['valid'],
        num_epochs=num_epochs,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=checkpoint_every,
        eval_freq=eval_freq,
        classification_thresh=binary_threshold,
        eval_iter=eval_iter,
    )

    results_path = plot_results(
        num_epochs, *history, dataset=metrics_plots_dir, title=metrics_plot_title
    )

    return results_path
