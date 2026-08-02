import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
from math import ceil
import obonet
from tqdm import tqdm
import pyarrow.ipc as ipc

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from pfdl.downloads import download_data

logger = logging.getLogger(__name__)


class ProtDataset(Dataset):
    """PyTorch dataset backed by a feather-loaded DataFrame of embeddings and GO-term targets."""

    def __init__(
            self,
            df: pd.DataFrame,
            embedding_prefix: str = "ME",
            target_prefix: str = "GO",
        ) -> None:
        """Initializes the dataset from a DataFrame of embedding and target columns.

        Args:
            df: DataFrame containing embedding columns prefixed with
              `embedding_prefix` and target columns prefixed with `target_prefix`.
            embedding_prefix: Column-name prefix identifying embedding features.
            target_prefix: Column-name prefix identifying GO-term target labels.
        """
        # Set both embbedding and target data types numpy.float32
        # Compatible with both PyTorch and JAX/FLAX
        self.embedding = df.filter(regex=f"^{embedding_prefix}").to_numpy(dtype=np.float32)
        self.target = df.filter(regex=f"^{target_prefix}").to_numpy(dtype=np.float32)

    def __len__(self):
        return self.embedding.shape[0]

    def __getitem__(self, idx):

        return {
            'embedding': self.embedding[idx],
            'target': self.target[idx]
            }


def build_dataset(
        batch_size: int,
        store_file_prefix: str,
        model_checkpoint: str
    ) -> dict[str, DataLoader]:
    """Builds train/valid/test DataLoaders from cached sequence-embedding feathers.

    Args:
        batch_size: Batch size used for all three DataLoaders.
        store_file_prefix: Shared filename prefix used to locate each split's
          cached embedding feather (split name is appended, e.g. `{prefix}_train`).
        model_checkpoint: HuggingFace checkpoint id used to derive the cached
          embedding filename.

    Returns:
        A dict mapping split name (`"train"`, `"valid"`, `"test"`) to its DataLoader.
    """
    dataset_splits = {}

    for split in ["train", "valid", "test"]:
        df = load_sequence_embeddings(
            store_file_prefix=f"{store_file_prefix}_{split}",
            model_checkpoint=model_checkpoint
        )
        dataset_splits[split] = create_data_loader(
            df,
            batch_size=batch_size,
            is_training=(split == "train")
        )
    
    return dataset_splits


def numpy_collate(batch):
    """Intercepts the batch to collate samples of dictionaries into pure NumPy arrays."""
    # Assumes all dictionaries in the batch have the same keys
    first_item = batch[0]
    
    return {
        key: np.array([sample[key] for sample in batch]) 
        for key in first_item.keys()
    }


def create_data_loader(data: pd.DataFrame, batch_size: int = 32, is_training: bool = False) -> DataLoader:
    """Wraps a DataFrame in a `ProtDataset` and a PyTorch `DataLoader`.

    Args:
        data: DataFrame of embedding and target columns (see `ProtDataset`).
        batch_size: Number of samples per batch.
        is_training: If True, shuffles samples and drops the last incomplete
          batch to keep training batch sizes consistent.

    Returns:
        A DataLoader yielding NumPy-collated `{"embedding", "target"}` batches.
    """
    dataset = ProtDataset(data)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_training, # Only shuffles the training set
        num_workers=0,            # Keep execution on the main host thread for simplicity
        collate_fn=numpy_collate,
        drop_last=is_training, # drop_last keeps batch sizes consistent during training steps
    )


def get_go_term_descriptions(obo_url: str, store_path: Path) -> pd.DataFrame:
    """Return GO term to description mapping, downloading if needed."""
    if not store_path.exists():

        try:
            graph = obonet.read_obo(obo_url)

        except Exception as e:
            logger.warning("Error occured: %s. Trying another method ...", e)
            url_parts = obo_url.split('/')
            base_url = "/".join(url_parts[:-1])
            filename = url_parts[-1]
            graph = obonet.read_obo(download_data(base_url, filename))

        # Extract GO term IDs and names from the graph nodes.
        id_to_name = {id: data.get("name") for id, data in graph.nodes(data=True)}
        
        # Create a df from names and nodes
        go_term_descriptions = pd.DataFrame(
            zip(id_to_name.keys(), id_to_name.values()),
            columns=["term", "description"],
        )
        go_term_descriptions.to_csv(store_path, index=False)

        logger.info("File is saved to %s.", store_path)
    else:
        logger.info("File already exists. Parsing the file into dataframe.")
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
        
        # Set device (CPU or GPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        n_samples = sequence_df.shape[0]
        n_batches = ceil(sequence_df.shape[0] / batch_size)
        
        batches: list[np.ndarray] = []

        # Instantiate progress bar track across mini-batches
        with tqdm(
            total=n_batches, 
            desc=f"Inference [{model_name}]", 
            unit="batch",
            dynamic_ncols=True
        ) as pbar:
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                batch_seqs = list(sequence_df["Sequence"].iloc[start_idx:end_idx])
                
                # Extract mean pooling arrays via tokenized forward pass
                batch_embeddings = get_mean_embeddings(batch_seqs, tokenizer, model, device)
                batches.extend(batch_embeddings)
                
                # Update visual bar metrics inline
                pbar.update(1)

        # Store each of the embedding values in a separate column in the dataframe.
        embeddings = pd.DataFrame(np.vstack(batches))
        embeddings.columns = [f"ME:{int(i)+1}" for i in range(embeddings.shape[1])]
        
        # Combine original dataframe with embeddings dataframe along columns
        df = pd.concat([sequence_df.reset_index(drop=True), embeddings], axis=1)
        df.to_feather(store_file)
        logger.info("%s is saved to the disk.", store_file)


def get_mean_embeddings(
    sequences: list[str],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: torch.device | None = None,
) -> np.ndarray:
    """Compute mean embedding for each sequence using a protein LM."""
    
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Tokenize input sequences and pad them to equal length.
    model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

    # Move tokenized inputs to the target device (CPU or GPU).
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    # Move model to the target device and set it to evaluation mode.
    model = model.to(device)
    model.eval()

    # Forward pass without gradient tracking to obtain embeddings.
    with torch.no_grad():
        outputs = model(**model_inputs)
    mean_embeddings = outputs.last_hidden_state.mean(dim=1)

    return mean_embeddings.detach().cpu().numpy()


def load_sequence_embeddings(
    store_file_prefix: str, model_checkpoint: str
    ) -> pd.DataFrame:
    """Load stored embedding DataFrame from disk."""

    model_name = model_checkpoint.replace("/", "_")
    store_file = f"{store_file_prefix}_{model_name}.feather"

    with ipc.open_file(Path(store_file)) as reader:
        return reader.read_all().to_pandas()
