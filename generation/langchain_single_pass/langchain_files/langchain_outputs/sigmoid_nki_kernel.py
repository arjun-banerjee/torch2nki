from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_sigmoid(a_tensor):
    # Initialize result array with same shape and dtype as input tensor
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input data from HBM to on-chip memory
    input_tile = nl.load(a_tensor)
    
    # Compute sigmoid using NKI's optimized implementation
    output_tile = nl.sigmoid(input_tile, dtype=a_tensor.dtype)
    
    # Store result back to HBM
    nl.store(result, output_tile)
    
    return result