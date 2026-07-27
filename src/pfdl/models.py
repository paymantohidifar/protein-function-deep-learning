import torch
import torch.nn as nn

import jax
import jax.numpy as jnp
from flax import nnx


class SimpleMlp(nnx.Module):
    """
    Task-specific projection head for multi-label protein functional annotation.
    
    Maps fixed-size protein language model embeddings (e.g., ESM-2) into a 
    multi-label probability logit space across target functional annotations.
    """
    def __init__(self, emb_dim: int, num_targets: int, rngs: nnx.Rngs):
        super().__init__()
        
        # Latent Expansion Layer: Expands representation capacity (dim -> 2 * dim)
        self.dense_1 = nnx.Linear(emb_dim, 2 * emb_dim, rngs=rngs)
        
        # Latent Compression Layer: Bottlenecks features back to original embedding dimension
        self.dense_2 = nnx.Linear(2 * emb_dim, emb_dim, rngs=rngs)
        
        # Multi-Label Output Projection Head: Maps representations to unnormalized logits
        self.output_head = nnx.Linear(emb_dim, num_targets, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass executing non-linear projection across dense layers.
        
        Args:
            x: Input tensor of shape (batch_size, emb_dim) representing 
               mean-pooled ESM-2 protein embeddings.
               
        Returns:
            jnp.ndarray: Raw, unnormalized prediction logits of shape (batch_size, num_targets).
        """
        # First non-linear projection
        x = self.dense_1(x)
        x = nnx.gelu(x)
        
        # Second non-linear projection
        x = self.dense_2(x)
        x = nnx.gelu(x)
        
        # Final linear projection to multi-label task space
        logits = self.output_head(x)
        return logits


class SimpleMlpTorch(nn.Module):
    pass