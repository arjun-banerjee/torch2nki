from neuronxcc import nki
import neuronxcc.nki.language as nl
import numpy as np

@nki.jit
def nki_vector_add(a_tensor, b_tensor):
    # Ensure vectors have same length
    if a_tensor.shape[0] != b_tensor.shape[0]:
        raise ValueError("Vectors must be of same length")

    # Create proper 2D indexing
    i_p = nl.arange(a_tensor.shape[0])[:, None]
    i_f = nl.arange(1)[None, :]
    
    # Reshape inputs to 2D
    a_2d = nl.expand_dims(a_tensor, 1)
    b_2d = nl.expand_dims(b_tensor, 1)

    # Load and add vectors
    a_tile = nl.load(a_2d[i_p, i_f])
    b_tile = nl.load(b_2d[i_p, i_f])
    result = nl.add(a_tile, b_tile)
    
    # Store result
    result_tensor = nl.zeros((a_tensor.shape[0], 1), dtype=a_tensor.dtype)
    nl.store(result_tensor, result)
    
    return result_tensor[:, 0]  # Return 1D result