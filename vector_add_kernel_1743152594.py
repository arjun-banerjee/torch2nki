from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(v1_tensor, v2_tensor):
    if v1_tensor.shape != v2_tensor.shape:
        raise ValueError("Vectors must be of the same length")
    
    # Initialize result tensor 
    result = nl.zeros((nl.tile_size.pmax,), dtype=v1_tensor.dtype)
    
    # Create indices for proper dimensionality
    i_p = nl.arange(v1_tensor.shape[0])[:, None]
    
    # Load input vectors with proper dimensions
    v1_tile = nl.load(v1_tensor[i_p])
    v2_tile = nl.load(v2_tensor[i_p])
    
    # Perform vector addition
    temp = nl.add(v1_tile, v2_tile)
    
    # Store result
    nl.store(result[i_p], temp)
    
    return result