from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_sqrt(a_tensor):
    # Initialize result tensor with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input data from HBM to on-chip memory
    in_tile = nl.load(a_tensor)
    
    # Compute square root
    temp = nl.sqrt(in_tile)
    
    # Store result back to HBM
    nl.store(result, value=temp)
    
    return result