from neuronxcc import nki
import neuronxcc.nki.language as nl
import numpy as np

@nki.jit
def nki_vector_add(a_tensor, b_tensor):
    if a_tensor.shape[0] != b_tensor.shape[0]:
        raise ValueError("Vectors must be of the same length")
    
    # Create index arrays for partition and free dimensions
    i_p = nl.arange(a_tensor.shape[0])[:, None]
    i_f = nl.arange(1)[None, :]
    
    # Load input vectors
    a_tile = nl.load(a_tensor[i_p, i_f])
    b_tile = nl.load(b_tensor[i_p, i_f])
    
    # Perform addition
    result = nl.zeros((a_tensor.shape[0], 1), dtype=a_tensor.dtype)
    result = nl.add(a_tile, b_tile)
    
    # Store result
    nl.store(result, value=result)
    
    return result