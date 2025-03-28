from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(a_tensor, b_tensor):
    if a_tensor.shape != b_tensor.shape:
        raise ValueError("Vectors must be of the same length")

    # Load tensors into on-chip memory
    a_tile = nl.load(a_tensor)
    b_tile = nl.load(b_tensor)

    # Create result tensor
    result = nl.zeros(a_tensor.shape, dtype=a_tensor.dtype)
    
    # Perform addition and store result
    temp = nl.add(a_tile, b_tile)
    nl.store(result, temp)

    return result