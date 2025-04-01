from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_threshold(a_tensor, thresh):
    # Initialize result array
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input data 
    input_tile = nl.load(a_tensor)
    
    # Compare against threshold
    mask = nl.greater(input_tile, nl.float32(thresh))
    
    # Store result directly
    nl.store(result, mask)
    
    return result