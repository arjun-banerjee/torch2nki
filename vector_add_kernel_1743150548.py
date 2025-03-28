from neuronxcc import nki
import neuronxcc.nki.language as nl
import numpy as np

@nki.jit
def nki_vector_add(a_tensor, b_tensor):
    # Ensure vectors have same length
    if a_tensor.shape[0] != b_tensor.shape[0]:
        raise ValueError("Vectors must be of same length")

    # Reshape inputs to 2D
    a_tensor = a_tensor[:, None]
    b_tensor = b_tensor[:, None]

    # Create result tensor with proper 2D shape
    result = nl.zeros((a_tensor.shape[0], 1), dtype=a_tensor.dtype)

    # Load values from device memory to on-chip memory
    a_tile = nl.load(a_tensor)
    b_tile = nl.load(b_tensor)

    # Perform vector addition
    out_tile = nl.add(a_tile, b_tile)

    # Store result back to device memory
    nl.store(result, out_tile)

    return result.reshape(-1)  # Return 1D result