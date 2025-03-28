from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(a_tensor, b_tensor):
    if a_tensor.shape != b_tensor.shape:
        raise ValueError("Vectors must be of the same length")

    # Load tensors into on-chip memory
    a_tile = nl.load(a_tensor)
    b_tile = nl.load(b_tensor)

    # Perform addition and store in temporary result
    temp_result = nl.add(a_tile, b_tile)

    # Create result tile and store the addition result
    result = nl.zeros(a_tensor.shape, dtype=a_tensor.dtype)
    nl.store(result, temp_result)

    return result