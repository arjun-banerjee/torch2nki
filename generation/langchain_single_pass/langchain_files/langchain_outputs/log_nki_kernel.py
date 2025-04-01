from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_log(a_tensor):
    # Initialize result tensor with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input data from HBM to on-chip memory
    in_tile = nl.load(a_tensor)
    
    # Compute natural logarithm
    out_tile = nl.log(in_tile)
    
    # Store result back to HBM
    nl.store(result, value=out_tile)
    
    return result