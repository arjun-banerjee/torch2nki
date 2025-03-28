from neuronxcc import nki
import neuronxcc.nki.language as nl
import numpy as np

@nki.jit
def nki_vector_add(a_tensor, b_tensor):
    """
    Adds two vectors element-wise using NKI.
    
    Args:
        a_tensor: First input vector
        b_tensor: Second input vector
        
    Returns:
        Result tensor containing element-wise sum
    """
    # Verify input shapes match
    if a_tensor.shape[0] != b_tensor.shape[0]:
        raise ValueError("Vectors must be of the same length")
        
    # Get vector length
    vector_len = a_tensor.shape[0]
    
    # Calculate number of full tiles needed
    # Using tile_size.pmax (128) as the tile size for partition dimension
    num_tiles = (vector_len + nl.tile_size.pmax - 1) // nl.tile_size.pmax
    
    # Initialize output tensor with zeros
    result = nl.zeros(a_tensor.shape, dtype=a_tensor.dtype)
    
    # Process vector in tiles
    for i in nl.affine_range(num_tiles):
        # Calculate current tile indices
        start_idx = i * nl.tile_size.pmax
        i_p = start_idx + nl.arange(nl.tile_size.pmax)[:, None]
        
        # Load input tiles from HBM to on-chip memory
        # Use mask to handle the last tile that may be partial
        a_tile = nl.load(a_tensor[i_p], mask=(i_p < vector_len))
        b_tile = nl.load(b_tensor[i_p], mask=(i_p < vector_len))
        
        # Perform element-wise addition
        result_tile = nl.add(a_tile, b_tile)
        
        # Store result back to HBM
        nl.store(result[i_p], value=result_tile, mask=(i_p < vector_len))
        
    return result