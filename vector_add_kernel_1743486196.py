from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(a_tensor, b_tensor):
    # Initialize result tensor with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load tensors from HBM to on-chip memory
    a_tile = nl.load(a_tensor)
    b_tile = nl.load(b_tensor)
    
    # Perform addition
    temp = nl.add(a_tile, b_tile)
    
    # Store result back to HBM
    nl.store(result, temp)
    
    return result