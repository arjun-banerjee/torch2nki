from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_cos(a_tensor):
    # Initialize result tensor with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input data from HBM to on-chip memory
    input_tile = nl.load(a_tensor)
    
    # Compute cosine using built-in cos function
    temp_result = nl.cos(input_tile)
    
    # Store result back to HBM
    nl.store(result, temp_result)
    
    return result