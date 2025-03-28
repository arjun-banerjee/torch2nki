from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(a_tensor, b_tensor):
    if a_tensor.shape != b_tensor.shape:
        raise ValueError("Vectors must be of the same length")

    # Create indices for partition and free dimensions
    i_p = nl.arange(a_tensor.shape[0])[:, None]
    i_f = nl.arange(1)[None, :]

    # Load tensors into on-chip memory with 2D indexing
    a_tile = nl.load(a_tensor[i_p, i_f])
    b_tile = nl.load(b_tensor[i_p, i_f])

    # Perform addition
    result = nl.add(a_tile, b_tile)

    return result