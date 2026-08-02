esm2_checkpoints: dict[str, str] = {
    "esm2-8M": "facebook/esm2_t6_8M_UR50D",
    "esm2-35M": "facebook/esm2_t12_35M_UR50D",
    "esm2-150M": "facebook/esm2_t30_150M_UR50D",
    "esm2-650M": "facebook/esm2_t33_650M_UR50D",
    "esm2-3B": "facebook/esm2_t36_3B_UR50D"
}


def get_esm_checkpoint(model_name: str) -> str:
    """Resolves a short ESM-2 model name to its HuggingFace checkpoint id.

    Args:
        model_name: Short model key, e.g. `"esm2-150M"`.

    Returns:
        The corresponding HuggingFace checkpoint id.

    Raises:
        KeyError: If `model_name` is not a recognized key in `esm2_checkpoints`.
    """
    if model_name not in esm2_checkpoints:
        raise KeyError(
            f"Unknown model_name {model_name!r}. Available options: "
            f"{sorted(esm2_checkpoints.keys())}"
        )
    return esm2_checkpoints[model_name]
