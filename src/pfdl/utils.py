from transformers import PreTrainedModel, PreTrainedTokenizer
from matplotlib.figure import Figure
import flax.linen as nn
from flax.training import train_state

from sklearn import metrics

import os
import numpy as np
import pandas as pd

import obonet


import optax

from dlfb.utils.restore import restorable


class MaskPredictor:
    """Predict masked amino acids using a protein language model."""

    def __init__(self, tokenizer: PreTrainedTokenizer, model: PreTrainedModel):
        """Initialize with a tokenizer and pretrained model."""
        # Stores the Hugging Face components to the instance
        self.tokenizer = tokenizer
        self.model = model

    def plot_predictions(self, sequence: str, mask_index: int) -> Figure:
        """Plot predicted probabilities for the masked amino acid."""
        # Get the probability distribution for the masked position
        mask_probs = self.predict(sequence, mask_index)

        # Setup the Matplotlib visualization
        fig, _ = plt.subplots(figsize=(6, 4))
        plt.bar(list(self.tokenizer.get_vocab().keys()), mask_probs, color="grey")
        plt.xticks(rotation=90)

        # Use f-strings to dynamically label the true residue for comparison
        plt.title(
            "Model Probabilities for the Masked Amino Acid\n"
            f"at Index={mask_index} (True Amino Acid = {sequence[mask_index]})."
        )
        return fig

    def predict(self, sequence: str, mask_index: int) -> jax.Array:
        """Return model probabilities for masked amino acid at a position."""
        # Generate the masked string (e.g., "MA<mask/>WM")
        masked_sequence = self.mask_sequence(sequence, mask_index)

        # Tokenize and move to PyTorch ("pt")
        masked_inputs = self.tokenizer(masked_sequence, return_tensors="pt")

        # Inference: Get raw logit scores from the model
        model_outputs = self.model(**masked_inputs)

        # Extract the specific token prediction.
        # Index is 'mask_index + 1' to account for the prepended <cls> token.
        mask_preds = model_outputs.logits[0, mask_index + 1].detach().numpy()

        # Convert to probability distribution using Softmax
        mask_probs = jax.nn.softmax(mask_preds)
        return mask_probs

    @staticmethod
    def mask_sequence(sequence: str, mask_index: int) -> str:
        """Insert mask token at specified index in the input sequence."""
        # Boundary check to prevent slicing errors
        if mask_index < 0 or mask_index > len(sequence):
            raise ValueError("Mask index outside of sequence range.")

        # String slicing to replace the target residue with the <mask> tag
        return f"{sequence[0:mask_index]}<mask/>{sequence[(mask_index + 1):]}"
    

def get_go_term_descriptions(store_path: str) -> pd.DataFrame:
  """Return GO term to description mapping, downloading if needed."""
  if not os.path.exists(store_path):
    url = "https://current.geneontology.org/ontology/go-basic.obo"

    # --- To get around 403 error ---
    # import requests
    # import io

    # response = requests.get(url)
    # graph = obonet.read_obo(io.StringIO(response.text))
    # -------------------------------

    graph = obonet.read_obo(url)  

    # Extract GO term IDs and names from the graph nodes.
    id_to_name = {id: data.get("name") for id, data in graph.nodes(data=True)}
    go_term_descriptions = pd.DataFrame(
      zip(id_to_name.keys(), id_to_name.values()),
      columns=["term", "description"],
    )
    go_term_descriptions.to_csv(store_path, index=False)

  else:
    go_term_descriptions = pd.read_csv(store_path)
  return go_term_descriptions


def store_sequence_embeddings(
  sequence_df: pd.DataFrame,
  store_prefix: str,
  tokenizer: PreTrainedTokenizer,
  model: PreTrainedModel,
  batch_size: int = 64,
  force: bool = False,
) -> None:
  """Extract and store mean embeddings for each protein sequence."""
  model_name = str(model.name_or_path).replace("/", "_")
  store_file = f"{store_prefix}_{model_name}.feather"

  if not os.path.exists(store_file) or force:
    device = get_device()

    # Iterate through protein dataframe in batches, extracting embeddings.
    n_batches = ceil(sequence_df.shape[0] / batch_size)
    batches: list[np.ndarray] = []
    for i in range(n_batches):
      batch_seqs = list(
        sequence_df["Sequence"][i * batch_size : (i + 1) * batch_size]
      )
      batches.extend(get_mean_embeddings(batch_seqs, tokenizer, model, device))

    # Store each of the embedding values in a separate column in the dataframe.
    embeddings = pd.DataFrame(np.vstack(batches))
    embeddings.columns = [f"ME:{int(i)+1}" for i in range(embeddings.shape[1])]
    df = pd.concat([sequence_df.reset_index(drop=True), embeddings], axis=1)
    df.to_feather(store_file)


def load_sequence_embeddings(
  store_file_prefix: str, model_checkpoint: str
) -> pd.DataFrame:
  """Load stored embedding DataFrame from disk."""
  model_name = model_checkpoint.replace("/", "_")
  store_file = f"{store_file_prefix}_{model_name}.feather"
  return pd.read_feather(store_file)


def convert_to_tfds(
  df: pd.DataFrame,
  embeddings_prefix: str = "ME:",   # Prefix for Mean Embedding columns (e.g., ME:0, ME:1...)
  target_prefix: str = "GO:",       # Prefix for multi-hot encoded GO labels
  is_training: bool = False,       # Toggle for training-specific optimizations
  shuffle_buffer: int = 50,         # Determines how many items to buffer for random sampling
) -> tf.data.Dataset:
  """Convert embedding DataFrame into a TensorFlow dataset."""
  
  # Slice and Extract
  # filter(regex=...) identifies columns by their prefix.
  # to_numpy() converts the tabular data into dense numerical matrices.
  # from_tensor_slices() creates a dataset where each row in the array becomes an element.
  dataset = tf.data.Dataset.from_tensor_slices(
    {
      "embedding": df.filter(regex=f"^{embeddings_prefix}").to_numpy(),
      "target": df.filter(regex=f"^{target_prefix}").to_numpy(),
    }
  )

  # Training Pipeline Optimizations
  if is_training:
    # shuffle() ensures the model doesn't learn the order of the examples.
    # repeat() allows the dataset to be streamed indefinitely over multiple epochs.
    dataset = dataset.shuffle(shuffle_buffer).repeat()

  return dataset


def build_dataset(
  store_file_prefix: str,
  model_checkpoint: str
) -> dict[str, tf.data.Dataset]:
  """Build train/valid/test TensorFlow datasets from stored embeddings."""
  dataset_splits = {}

  for split in ["train", "valid", "test"]:
    dataset_splits[split] = convert_to_tfds(
      df=load_sequence_embeddings(
         store_file_prefix=f"{store_file_prefix}_{split}",
         model_checkpoint=model_checkpoint,
      ),
      is_training=(split == "train"),
    )
  return dataset_splits


class Model(nn.Module):
  """Simple MLP for protein function prediction."""

  # Hyperparameters
  num_targets: int       # Number of GO terms to predict (e.g., 303)
  dim: int = 256         # Base dimension for hidden layers

  @nn.compact
  def __call__(self, x):
    """Apply MLP layers to input features."""
    
    # Sequential Architecture
    x = nn.Sequential(
      [
        # Layer 1: Expansion. Projects embedding (e.g., 256) to a wider space (512).
        nn.Dense(self.dim * 2),
        jax.nn.gelu,          # Smooth activation (Gaussian Error Linear Unit)
        
        # Layer 2: Contraction.
        nn.Dense(self.dim),
        jax.nn.gelu,
        
        # Layer 3: Output Projection.
        # Projects to the number of functional labels.
        nn.Dense(self.num_targets),
      ]
    )(x)
    return x
  
def create_train_state(self, rng: jax.Array, dummy_input, tx) -> TrainState:
    """Initialize model parameters and return a training state."""
    
    # Parameter Initialization
    # self.init runs the forward pass with dummy_input to determine weight shapes.
    variables = self.init(rng, dummy_input)
    
    # Training State Creation
    # Encapsulates parameters, the forward function (apply_fn), and the optimizer (tx).
    return TrainState.create(
       apply_fn=self.apply,
       params=variables["params"],
       tx=tx
    )

@jax.jit
def train_step(state, batch):
  """Run a single training step and update model parameters."""

  # Define the Differentiable Objective
  def calculate_loss(params):
    """Compute sigmoid cross-entropy loss from logits."""
    # Pass embeddings through the model (state.apply_fn) to get raw logits
    logits = state.apply_fn({"params": params}, x=batch["embedding"])
    
    # Use Sigmoid Binary Cross Entropy for multi-label classification.
    loss = optax.sigmoid_binary_cross_entropy(logits, batch["target"]).mean()
    return loss

  # Gradient Calculation
  # value_and_grad returns both the loss (value) and the derivatives (grad).
  grad_fn = jax.value_and_grad(calculate_loss, has_aux=False)
  loss, grads = grad_fn(state.params)

  # Parameter Update
  # apply_gradients handles the optimizer logic (e.g., Adam) to update weights.
  state = state.apply_gradients(grads=grads)
  
  return state, loss

def compute_metrics(
    targets: np.ndarray, probs: np.ndarray, thresh=0.5
) -> dict[str, float]:
    """Compute accuracy, recall, precision, auPRC, and auROC."""
    
    # Edge-Case Guard: Prevent division-by-zero crashes if there are no positive targets
    # (e.g., auROC/Recall are mathematically undefined if the true positive count is 0)
    if np.sum(targets) == 0:
        return {
            m: 0.0 for m in ["accuracy", "recall", "precision", "auprc", "auroc"]
        }
        
    return {
        # Threshold-dependent metric: Evaluates hard discrete classifications (True vs False)
        "accuracy": float(metrics.accuracy_score(targets, probs >= thresh)),
        
        # Recall (Sensitivity): Proportion of actual positives correctly identified
        "recall": metrics.recall_score(targets, probs >= thresh).item(),
        
        # Precision (PPV): Proportion of predicted positives that are truly positive
        # zero_division=0.0 prevents a crash if the model predicts 0 positive instances total
        "precision": metrics.precision_score(
            targets,
            probs >= thresh,
            zero_division=0.0,
        ).item(),
        
        # Area Under the Precision-Recall Curve (auPRC): Threshold-agnostic metric 
        # that evaluates prediction confidence ranking (highly robust for imbalanced data)
        "auprc": metrics.average_precision_score(targets, probs).item(),
        
        # Area Under the Receiver Operating Characteristic (auROC): Probability that a 
        # randomly chosen positive sample ranks higher than a randomly chosen negative sample
        "auroc": metrics.roc_auc_score(targets, probs).item(),
    }


def eval_step(state, batch) -> dict[str, float]:
    """Run evaluation step and return mean metrics over targets."""
    
    # Forward Pass: Compute unnormalized log-probabilities (logits) using the model's apply function
    logits = state.apply_fn({"params": state.params}, x=batch["embedding"])
    
    # Compute the multi-label binary cross-entropy loss and average it across the batch
    loss = optax.sigmoid_binary_cross_entropy(logits, batch["target"]).mean()
    
    # Compute metrics column-by-column (per target) across the entire batch
    target_metrics = calculate_per_target_metrics(logits, batch["target"])
    
    # Construct the final dictionary: Extract scalar loss and calculate the mean 
    # of each evaluation metric across all evaluated targets using a Pandas DataFrame
    metrics_summary = {
        "loss": loss.item(),
        **pd.DataFrame(target_metrics).mean(axis=0).to_dict(),
    }
    return metrics_summary


def calculate_per_target_metrics(logits, targets) -> list[dict[str, float]]:
    """Compute metrics for each discrete target class across a multi-label batch."""
    
    probs = jax.nn.sigmoid(logits)
    target_metrics = []
    
    # Loop over every protein (example) independently
    for target, prob in zip(targets, probs):
        # compute_metrics evaluates accuracy/recall/precision/PRC/ROC for a single protein
        metric_dict = compute_metrics(target, prob)
        target_metrics.append(metric_dict)
        
    return target_metrics


def train(
    state: TrainState,
    dataset_splits: dict[str, tf.data.Dataset],
    batch_size: int,
    num_steps: int = 300,
    eval_every: int = 30,
    ):
    """Train model using batched TF datasets and track performance metrics."""
    
    # Create containers to handle calculated during training and evaluation.
    train_metrics, valid_metrics = [], []
    
    # Create batched dataset to pluck batches from for each step.
    train_batches = (
        dataset_splits["train"]
        .batch(batch_size, drop_remainder=True)
        .as_numpy_iterator()
    )

    steps = tqdm(range(num_steps)) # Steps with progress bar.
    for step in steps:
        steps.set_description(f"Step {step + 1}")
        
        # Get batch of training data, convert into a JAX array, and train.
        state, loss = train_step(state, next(train_batches))
        train_metrics.append({"step": step, "loss": loss.item()})
        
        if step % eval_every == 0:
            # For all the evaluation batches, calculate metrics.
            eval_metrics = []
            for eval_batch in (
                dataset_splits["valid"].batch(batch_size=batch_size).as_numpy_iterator()
                ):
                eval_metrics.append(eval_step(state, eval_batch))
            valid_metrics.append(
                {"step": step, **pd.DataFrame(eval_metrics).mean(axis=0).to_dict()}
        )
    return state, {"train": train_metrics, "valid": valid_metrics}
