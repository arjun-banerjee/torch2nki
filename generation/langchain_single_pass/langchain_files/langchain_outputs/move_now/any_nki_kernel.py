from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_any(a_tensor):
    # Initialize result array
    result = nl.ndarray((), dtype=nl.bool_, buffer=nl.shared_hbm)
    
    # Load and transpose input data
    input_sbuf = nl.load_transpose2d(a_tensor)
    
    # Start with first value
    temp = input_sbuf[0, 0]
    
    # Combine elements using logical_or
    for i in nl.affine_range(input_sbuf.shape[0]):
        for j in nl.affine_range(input_sbuf.shape[1]):
            temp = nl.logical_or(temp, input_sbuf[i, j])
    
    # Store result
    nl.store(result, temp)
    
    return result