import numpy as np
import jax.numpy as jnp
from flax import nnx
import optax
from torch.utils.data import DataLoader
from tqdm import tqdm

from pfdl.evaluate import evaluate


@nnx.jit
def train_step(
        model_arg: nnx.Module,
        optimizer_arg: nnx.Optimizer,
        x_batch: np.ndarray,
        y_batch: np.ndarray
        ) -> float:

    def loss_fn_for_grad(model_in_grad_fn: nnx.Module) -> float:

        logits = model_in_grad_fn(x_batch)
        loss = optax.sigmoid_binary_cross_entropy(logits, y_batch).mean()
        return loss

    loss_val, grads = nnx.value_and_grad(loss_fn_for_grad)(model_arg)
    optimizer_arg.update(model_arg, grads)
    return loss_val
    

def train_loop(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    dataset_splits: dict[str, DataLoader],
    num_epochs: int=10,
    eval_iter: int=5,
    thresh: float=0.5
          ):

    train_metrics = []
    valid_metrics = []

    train_loader = dataset_splits['train']

    pbar = tqdm(range(num_epochs))

    for epoch in pbar:
        # pbar.set_description(f"Epoch: {epoch + 1}")

        for batch in train_loader:
            loss = train_step(model, optimizer, batch['embedding'], batch['target'])
            train_metrics.append({'epoch': epoch, 'loss': loss.item()})


        if epoch % eval_iter == 0:

            valid_loader = dataset_splits['valid']
            average_epoch_metrics = evaluate(model, valid_loader, thresh)
            valid_metrics.append({'epoch': epoch, **average_epoch_metrics})

            # pbar.set_description(f"Epoch: {epoch + 1}")
        
        pbar.set_postfix(epoch=f"{epoch+1}", loss=f"{average_epoch_metrics['loss']:.4f}")
        
    return {'train': train_metrics, 'valid': valid_metrics}





# def train_loop(model: nnx.Module, optimizer: nnx.Optimizer, 
#                data_loader, num_epochs: int) -> list[float]:
#     """
#     Executes a standard training loop over either a PyTorch DataLoader 
#     or a Google Grain MapDataset pipeline with real-time tqdm metrics.
#     """
#     epoch_losses = []

#     # Assign the tqdm progress bar to a variable ('pbar')
#     pbar = tqdm(range(num_epochs), desc="Training Progress")

#     for epoch in pbar:
#         running_epoch_loss = 0.0
#         total_batches = 0
        
#         # Check if it's a Grain pipeline and extract its underlying iterator.
#         # This converts a lazy Grain pipeline into a clean, sliceable batch generator.
#         if hasattr(data_loader, "as_numpy_iterator"):
#             batch_stream = data_loader.as_numpy_iterator()
#         else:
#             batch_stream = data_loader # Default fallback for PyTorch DataLoader

#         # Stream batches out of our host-side pipeline
#         for x_batch, y_batch in batch_stream:
            
#             loss_value = train_step(model, optimizer, x_batch, y_batch)
            
#             # Cast the JAX array to a standard Python float 
#             # to break the tracking chain and prevent device memory bloat
#             running_epoch_loss += float(loss_value)
#             total_batches += 1
            
#         # Safely compute the average for the current epoch
#         if total_batches > 0:
#             average_epoch_loss = running_epoch_loss / total_batches
#             epoch_losses.append(average_epoch_loss)

#             # Update the progress bar postfix with the latest calculated loss
#             pbar.set_postfix(loss=f"{average_epoch_loss:.4f}")
#         else:
#             epoch_losses.append(0.0)
            
#     return epoch_losses




# # The nnx.jit decorator safely manages stateful objects crossing the JIT boundary
# @nnx.jit
# def train_step(model_arg: nnx.Module, optimizer_arg: nnx.Optimizer,
#                x_batch: jnp.ndarray, y_batch: jnp.ndarray) -> jnp.ndarray:
    
#     # Internal pure function mapping input parameters to a scalar loss
#     def loss_fn_for_grad(model_in_grad_fn: nnx.Module) -> float:
#         y_pred = model_in_grad_fn(x_batch)
#         return jnp.mean((y_batch - y_pred) ** 2)
    
#     # nnx.value_and_grad cleanly extracts the dynamic pytree values and their gradients
#     loss_value, grads = nnx.value_and_grad(loss_fn_for_grad)(model_arg)
    
#     # Mutates optimizer state in-place; NNX abstracts this side effect away safely
#     optimizer_arg.update(model_arg, grads)  
#     return loss_value

