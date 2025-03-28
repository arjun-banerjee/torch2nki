from neuronxcc import nki
import neuronxcc.nki.language as nl
import numpy as np

@nki.jit
def nki_vector_add(a_tensor, b_tensor):
    # Ensure vectors have same length
    if a_tensor.shape[0] != b_tensor.shape[0]:
        raise ValueError("Vectors must be of same length")

    # Create proper indexing for partition dimension
    i_p = nl.arange(a_tensor.shape[0])

    # Load input vectors
    a_tile = nl.load(a_tensor[i_p])
    b_tile = nl.load(b_tensor[i_p])

    # Perform addition
    result = nl.add(a_tile, b_tile)

    # Store result
    nl.store(a_tensor[i_p], result)
    
    return result