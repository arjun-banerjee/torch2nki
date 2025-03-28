from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(v1_tensor, v2_tensor):
    if v1_tensor.shape != v2_tensor.shape:
        raise ValueError("Vectors must be of the same length")
    
    # Create proper indices for 2D operations
    i_p = nl.arange(v1_tensor.shape[0])[:, None]
    
    # Initialize result with proper dimensionality
    result = nl.zeros((v1_tensor.shape[0], 1), dtype=v1_tensor.dtype)
    
    # Load vectors with proper dimensionality
    v1_tile = nl.load(v1_tensor[i_p])
    v2_tile = nl.load(v2_tensor[i_p])
    
    # Perform addition maintaining dimensionality
    temp = nl.add(v1_tile, v2_tile)
    
    # Store result while maintaining shape
    nl.store(result[i_p], temp)
    
    return result