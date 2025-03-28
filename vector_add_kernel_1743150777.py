from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_vector_add(vec1, vec2):
    """
    Adds two vectors element-wise using NKI.
    
    Args:
        vec1: First input vector tensor
        vec2: Second input vector tensor
        
    Returns:
        Result tensor containing element-wise sum
    """
    # Verify input dimensions match
    if vec1.shape != vec2.shape:
        raise ValueError("Input vectors must have the same shape")
    
    # Get vector length
    vec_len = vec1.shape[0]
    
    # Calculate number of tiles needed
    # Use max partition size allowed by hardware
    trip_count = math.ceil(vec_len / nl.tile_size.pmax)
    
    # Initialize result tensor with same shape and dtype as inputs
    result = nl.zeros(vec1.shape, dtype=vec1.dtype, buffer=nl.hbm)
    
    # Process vector in tiles
    for p in nl.affine_range(trip_count):
        # Generate indices for current tile
        i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
        
        # Load tiles from input vectors
        # Use mask to handle last tile that may be partial
        vec1_tile = nl.load(vec1[i_p], mask=(i_p < vec_len))
        vec2_tile = nl.load(vec2[i_p], mask=(i_p < vec_len))
        
        # Perform element-wise addition
        result_tile = nl.add(vec1_tile, vec2_tile)
        
        # Store result back to output tensor
        nl.store(result[i_p], value=result_tile, mask=(i_p < vec_len))
        
    return result