from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(v1_tensor, v2_tensor):
    if v1_tensor.shape != v2_tensor.shape:
        raise ValueError("Vectors must be of the same length")
    
    # Create proper indices for 1D operations
    i_p = nl.arange(v1_tensor.shape[0])
    
    # Initialize result with proper dimensionality
    result = nl.zeros((v1_tensor.shape[0],), dtype=v1_tensor.dtype)
    
    # Load vectors with 1D indexing
    v1_tile = nl.load(v1_tensor[i_p])
    v2_tile = nl.load(v2_tensor[i_p])
    
    # Perform vector addition 
    temp = nl.add(v1_tile, v2_tile)
    
    # Store result
    nl.store(result[i_p], value=temp)
    
    return result