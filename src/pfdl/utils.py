from typing import Any
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from flax import nnx
import shutil
import orbax.checkpoint as ocp

from pfdl import paths


class CheckpointClassifier:
    """Orbax-backed checkpoint manager for Flax NNX models and optimizers.

    Handles asynchronous state saving, step rotation, and state restoration
    back into Flax NNX graph structures.
    """

    def __init__(
        self,
        checkpoint_dir: str = "classifier-checkpoints",
        max_to_keep: int = 2,
        best_metric: str = "val_loss",
        clean_existing: bool = False,
    ) -> None:
        """Initializes the Orbax CheckpointManager.

        Args:
            checkpoint_dir: Target directory path for storing checkpoint files.
            max_to_keep: Maximum number of recent checkpoints to retain on
              disk.
            clean_existing: If True, wipes existing checkpoints in
              `checkpoint_dir` upon initialization. Default is False to enable
              resumption.
        """
        self.checkpoint_path = (paths.MODELS_DIR / checkpoint_dir).resolve()

        if clean_existing and self.checkpoint_path.exists():
            shutil.rmtree(self.checkpoint_path)

        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        mngr_options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            best_fn=lambda metrics: metrics[best_metric],
            best_mode='min',
            # create=True
        )
        self.mngr = ocp.CheckpointManager(
            self.checkpoint_path,
            options=mngr_options
        )

    def save_checkpoint(
        self,
        step: int,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        step_metrics: dict[str, Any],
        step_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Asynchronously saves model parameters, optimizer state, and step metadata.

        Args:
            step: Monotonically increasing training step or epoch index.
            model: Active Flax NNX model instance.
            optimizer: Active Flax NNX optimizer instance.
            metadata: Optional dictionary of scalar metrics or execution
              metadata.
        """
        model_state = nnx.state(model)
        optimizer_state = nnx.state(optimizer)

        # Asynchronous save: launches background I/O thread
        self.mngr.save(
            step=step,
            args=ocp.args.Composite(
                model=ocp.args.StandardSave(model_state),
                optimizer=ocp.args.StandardSave(optimizer_state)
                ),
            metrics=step_metrics,
            custom_metadata=step_metadata
        )
            

    def load_checkpoint(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        step: int | None = None,
    ) -> tuple[int, dict[str, Any], dict[str, Any]]:
        """Restores model and optimizer states in-place from the specified or latest step.

        Args:
            model: Instantiated Flax NNX model to receive restored parameters.
            optimizer: Instantiated Flax NNX optimizer to receive restored
              states.
            step: Specific checkpoint step to restore. If None, fetches the
              latest available step.

        Returns:
            A `(restored_step, step_metrics, custom_metadata)` tuple.

        Raises:
            FileNotFoundError: If no valid checkpoints are available on disk.
        """
        target_step = step if step is not None else self.mngr.latest_step()

        if target_step is None:
            raise FileNotFoundError(
                f"No checkpoints found in directory: {self.checkpoint_path}"
            )

        # Construct target state template directly from live NNX objects
        abstract_model_state = nnx.state(model)
        abstract_optimizer_state = nnx.state(optimizer)

        # Deserialize binary artifacts directly into matching PyTree structure
        restored_states = self.mngr.restore(
            step=target_step,
            args=ocp.args.Composite(
                model=ocp.args.StandardRestore(abstract_model_state),
                optimizer=ocp.args.StandardRestore(abstract_optimizer_state)
            )
        )

        # Re-bind restored PyTree states back into live Flax NNX objects in-place
        nnx.update(model, restored_states["model"])
        nnx.update(optimizer, restored_states["optimizer"])

        # Restore step metrics and metadata
        step_metrics = self.mngr.metrics(step=target_step)
        step_meta = self.mngr.metadata(step=target_step)
        custom_meta = step_meta.custom_metadata

        print(
            f"Successfully restored checkpoint from step {target_step} into NNX objects."
        )
        return target_step, step_metrics, custom_meta

    def close(self) -> None:
        """Blocks until pending background I/O save operations complete."""
        self.mngr.wait_until_finished()


def _draw_metric(
    ax: plt.Axes,
    epochs_seen: np.ndarray,
    examples_seen: np.ndarray,
    train_values: list[float],
    val_values: list[float],
    label: str,
) -> None:
    """Draw train/validation curves for one metric onto ``ax``.

    Args:
        ax: Axes to draw onto.
        epochs_seen: X-axis values (epochs) for the primary axis.
        examples_seen: X-axis values (examples seen) for the secondary axis.
        train_values: Training metric values.
        val_values: Validation metric values.
        label: Metric name, used in the legend and axis label.
    """
    ax.plot(epochs_seen, train_values, label=f"Training {label}")
    ax.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel(label.capitalize())
    if label != "loss":
        ax.set_ylim(0, 1)
    ax.legend()

    ax_top = ax.twiny()
    ax_top.plot(examples_seen, train_values, alpha=0)
    ax_top.set_xlabel("Examples seen")


def plot_results(
    num_epochs: int,
    train_losses: list[float],
    valid_losses: list[float],
    train_metrics: dict[str, float],
    valid_metrics: dict[str, float],
    examples_seen: int,
    dataset: str,
    title: str | None = None,
    plot_name: str | None = "metrics.png",
    output_dir: Path = paths.PLOTS_DIR,
) -> Path:
    """Plots train/validation loss and per-metric curves and saves the figure to disk.

    Args:
        num_epochs: Total number of training epochs, used for the epoch axis.
        train_losses: Training loss recorded at each evaluation step.
        valid_losses: Validation loss recorded at each evaluation step.
        train_metrics: Per-step training classification metrics, keyed by
          metric name.
        valid_metrics: Per-step validation classification metrics, keyed by
          metric name.
        examples_seen: Total number of training examples seen, used for the
          secondary "examples seen" axis.
        dataset: Subdirectory name (under `output_dir`) to save the plot into.
        title: Optional figure title.
        plot_name: Output image filename.
        output_dir: Base directory to save the plot under.

    Returns:
        Path to the saved plot image.
    """
    df_train = pd.DataFrame(train_metrics)
    df_valid = pd.DataFrame(valid_metrics)

    metrics_to_plot = [("loss", train_losses, valid_losses)]
    for col in df_train.columns:
        metrics_to_plot.append((col, df_train[col].tolist(), df_valid[col].tolist()))    

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, (label, train_values, val_values) in zip(axes, metrics_to_plot):
        epochs_tensor = np.linspace(0, num_epochs, len(train_values))
        examples_seen_tensor = np.linspace(0, examples_seen, len(train_values))

        _draw_metric(ax, epochs_tensor, examples_seen_tensor, train_values, val_values, label)

    if title:
          fig.suptitle(title, fontweight='bold')
    fig.tight_layout()

    output_dir = Path(output_dir) / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / plot_name
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    return output_path