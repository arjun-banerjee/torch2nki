from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_clamp(a_tensor, min_value, max_value):
    # Initialize result array in shared HBM
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input tensor into on-chip memory
    input_tile = nl.load(a_tensor)
    
    # First get the maximum between input and min_value
    temp = nl.maximum(input_tile, min_value)
    
    # Then get the minimum between that result and max_value
    output_tile = nl.minimum(temp, max_value)
    
    # Store result back to HBM
    nl.store(result, output_tile)
    
    return result