from pathlib import Path
import zarr
import numpy as np
import pandas as pd
from math import ceil
import obonet
import torch
import os
from transformers import PreTrainedModel, PreTrainedTokenizer


def get_go_term_descriptions(obo_url: str, store_path: Path) -> pd.DataFrame:
    """Return GO term to description mapping, downloading if needed."""
    # if not os.path.exists(store_path):
    if not store_path.exists():
        graph = obonet.read_obo(obo_url)

        # Extract GO term IDs and names from the graph nodes.
        id_to_name = {id: data.get("name") for id, data in graph.nodes(data=True)}
        
        # Create a df from names and nodes
        go_term_descriptions = pd.DataFrame(
            zip(id_to_name.keys(), id_to_name.values()),
            columns=["term", "description"],
        )
        go_term_descriptions.to_csv(store_path, index=False)

        print(f"File Sis saved to {store_path}.")

    else:
        print("File already exists. Parsing the file into dataframe.")
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
    # store_file = f"{store_prefix}_{model_name}.zarr"
    # store_file = store_prefix.with_stem(f"_{model_name}").with_suffix(".zarr")

    if not os.path.exists(store_file) or force:
    # if not store_file.exits() or force:
        # device = get_device()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        
        # Combine original dataframe with embeddings dataframe along columns
        df = pd.concat([sequence_df.reset_index(drop=True), embeddings], axis=1)
        df.to_feather(store_file)
        # zarr.save(store_file, df, chunks=(1024,))


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
    return pd.read_feather(store_file)