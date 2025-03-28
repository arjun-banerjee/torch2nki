from neuronxcc import nki
import neuronxcc.nki.language as nl
from torch_neuronx import nki_jit
import numpy as np

@nki_jit
def vector_add_kernel(vec1, vec2, result, vec_length):
    """
    NKI kernel for vector addition using AWS Neuron hardware.
    
    Args:
        vec1: First input vector on device memory
        vec2: Second input vector on device memory
        result: Output vector on device memory
        vec_length: Length of the vectors
    """
    # Calculate number of full tiles needed
    tile_size = nl.tile_size.pmax  # Maximum partition size (128)
    num_tiles = (vec_length + tile_size - 1) // tile_size
    
    # Process vectors in tiles
    for tile_idx in nl.affine_range(num_tiles):
        # Generate indices for current tile
        i_p = tile_idx * tile_size + nl.arange(tile_size)[:, None]
        
        # Create mask for handling non-full tiles at the end
        mask = (i_p < vec_length)
        
        # Load input tiles from device memory
        vec1_tile = nl.load(vec1[i_p], mask=mask)
        vec2_tile = nl.load(vec2[i_p], mask=mask)
        
        # Perform addition
        result_tile = nl.add(vec1_tile, vec2_tile, mask=mask)
        
        # Store result back to device memory
        nl.store(result[i_p], value=result_tile, mask=mask)

def vector_add(v1, v2):
    """
    Wrapper function to handle vector addition using NKI kernel.
    
    Args:
        v1: First input vector (numpy array)
        v2: Second input vector (numpy array)
    
    Returns:
        Result of vector addition as numpy array
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of the same length")
        
    # Convert inputs to float32 arrays if needed
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)
    
    # Create output array
    result = np.empty_like(v1)
    
    # Call the NKI kernel
    vector_add_kernel(v1, v2, result, len(v1))
    
    return result