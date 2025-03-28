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
    
    vector_length = a_tensor.shape[0]
    
    # Calculate number of full tiles needed
    tiles_needed = (vector_length + nl.tile_size.pmax - 1) // nl.tile_size.pmax
    
    # Initialize result tensor
    result = nl.zeros(a_tensor.shape, dtype=a_tensor.dtype)
    
    # Process the vectors in tiles
    for tile_idx in nl.affine_range(tiles_needed):
        # Calculate indices for current tile
        start_idx = tile_idx * nl.tile_size.pmax
        i_p = start_idx + nl.arange(nl.tile_size.pmax)[:, None]
        
        # Load tiles from input vectors with masking for the last tile
        a_tile = nl.load(a_tensor[i_p], mask=(i_p < vector_length))
        b_tile = nl.load(b_tensor[i_p], mask=(i_p < vector_length))
        
        # Perform addition
        sum_tile = nl.add(a_tile, b_tile)
        
        # Store result with masking for the last tile
        nl.store(result[i_p], value=sum_tile, mask=(i_p < vector_length))
    
    return result