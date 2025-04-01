from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_vector_add(a_tensor, b_tensor):
    # Initialize result array in shared HBM
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load vectors from HBM to on-chip memory (SBUF)
    a_tile = nl.load(a_tensor)
    b_tile = nl.load(b_tensor)
    
    # Perform element-wise addition
    temp = nl.add(a_tile, b_tile)
    
    # Store result back to result array
    nl.store(result, temp)
    
    return result