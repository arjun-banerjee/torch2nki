from neuronxcc import nki
import neuronxcc.nki.language as nl
import numpy as np

@nki.jit
def nki_vector_add(a_tensor, b_tensor):
    # Ensure vectors have same length
    if a_tensor.shape[0] != b_tensor.shape[0]:
        raise ValueError("Vectors must be of same length")

    # Add new axis to make 2D
    a_tensor = a_tensor[:, None]  
    b_tensor = b_tensor[:, None]

    # Create output tensor with same shape
    result = nl.zeros((a_tensor.shape[0], 1), dtype=a_tensor.dtype)

    # Load vectors into on-chip memory
    a_tile = nl.load(a_tensor)
    b_tile = nl.load(b_tensor)

    # Perform element-wise addition
    output_tile = nl.add(a_tile, b_tile)

    # Store result back to device memory
    nl.store(result, output_tile)

    # Return 1D result by removing extra dimension
    return result[:, 0]