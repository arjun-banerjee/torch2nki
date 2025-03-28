from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(a_tensor, b_tensor):
    if a_tensor.shape != b_tensor.shape:
        raise ValueError("Vectors must be of the same length")

    # Create index for the vector dimension
    i_p = nl.arange(a_tensor.shape[0])

    # Load tensors into on-chip memory
    a_tile = nl.load(a_tensor[i_p])
    b_tile = nl.load(b_tensor[i_p])

    # Perform addition
    result = nl.add(a_tile, b_tile)

    return result