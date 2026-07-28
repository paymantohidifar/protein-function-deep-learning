from typing import Any, Dict, List, TypedDict
import jax.numpy as jnp
from flax import nnx
import optax
from torch.utils.data import DataLoader
from tqdm import tqdm

from pfdl.evaluate import evaluate_model, calc_classification_metrics_loader


class EpochMetrics(TypedDict):
    epoch: int
    accuracy: float
    recall: float
    precision: float
    prc_auc: float
    roc_auc: float
    

@nnx.jit
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    x_batch: jnp.ndarray,
    y_batch: jnp.ndarray
) -> float:

    def loss_fn(model_in: nnx.Module) -> float:

        logits = model_in(x_batch)
        return optax.sigmoid_binary_cross_entropy(logits, y_batch).mean()

    loss_val, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss_val
    

def train_loop(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    dataset_splits: Dict[str, DataLoader],
    num_epochs: int = 10,
    eval_freq: int = 5,
    eval_iter: int = 10,
    binary_threshold: float = 0.5
):

    global_step = 0

    train_losses = []
    valid_losses = []

    train_metrics: List[EpochMetrics] = []
    valid_metrics: List[EpochMetrics] = []

    train_loader = dataset_splits['train']
    valid_loader = dataset_splits['valid']

    # pbar = tqdm(range(num_epochs))

    for epoch in range(num_epochs):
        # pbar.set_description(f"Epoch: {epoch + 1}")

        # --- Training Pass ---
        for batch in train_loader:
            # Transfer host memory to JAX device arrays
            x_batch = jnp.asarray(batch["embedding"])
            y_batch = jnp.asarray(batch["target"])

            # Execute JIT-compiled optimization step
            _ = train_step(model, optimizer, x_batch, y_batch)

            global_step += 1

            # --- Evaluation Pass ---
            if global_step % eval_freq == 0:

                train_loss, valid_loss = evaluate_model(model, train_loader, valid_loader, max_steps=eval_iter)
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)

                print(
                    f"Epoch [{epoch + 1:03d}/{num_epochs:03d}] (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Valid loss {valid_loss:.3f}"
                )

        avg_train = calc_classification_metrics_loader(model, train_loader, threshold=binary_threshold, max_steps=eval_iter)
        avg_valid = calc_classification_metrics_loader(model, valid_loader, threshold=binary_threshold, max_steps=eval_iter)
        
        train_entry: EpochMetrics = {"epoch": epoch + 1, **avg_train}  # type: ignore[arg-type]
        valid_entry: EpochMetrics = {"epoch": epoch + 1, **avg_valid}  # type: ignore[arg-type]

        train_metrics.append(train_entry)
        valid_metrics.append(valid_entry)

        # pbar.set_description(f"Epoch: {epoch + 1}")
        
        print(f"Train accuracy: {train_entry['accuracy'] * 100:.2f}% | ", end="")
        print(f"Validation accuracy: {valid_entry['accuracy'] * 100:.2f}%")
        # print(
        #     f"Train precision: {train_entry['precision']:.3f}, recall: {train_entry['recall']:.3f}, "
        #     f"ROC-AUC: {train_entry['roc_auc']:.3f}, PRC-AUC: {train_entry['prc_auc']:.3f} | "
        #     f"Valid precision: {valid_entry['precision']:.3f}, recall: {valid_entry['recall']:.3f}, "
        #     f"ROC-AUC: {valid_entry['roc_auc']:.3f}, PRC-AUC: {valid_entry['prc_auc']:.3f}"
        # )

        # pbar.set_postfix(epoch=f"{epoch+1}", loss=f"{average_epoch_metrics['loss']:.4f}")
        
    # return {'train': train_metrics, 'valid': valid_metrics}

    return (
        train_losses,
        valid_losses,
        train_metrics,
        valid_metrics
    )