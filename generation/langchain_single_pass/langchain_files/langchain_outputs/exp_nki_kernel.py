from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_exp(a_tensor):
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input data from HBM to on-chip memory
    data_tile = nl.load(a_tensor)
    
    # Compute exponential
    out_tile = nl.exp(data_tile)
    
    # Store result back to HBM
    nl.store(result, value=out_tile)
    
    return result