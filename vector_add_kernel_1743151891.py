from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(a_tensor, b_tensor):
    # Check that vectors have same length
    if a_tensor.shape != b_tensor.shape:
        raise ValueError("Vectors must be of the same length")
    
    # Load tensors from HBM to on-chip memory (SBUF)
    a_tile = nl.load(a_tensor)
    b_tile = nl.load(b_tensor)
    
    # Perform element-wise addition
    result = nl.add(a_tile, b_tile)
    
    return result