from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_tan(a_tensor):
    # Initialize result tensor
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input data
    x = nl.load(a_tensor)
    
    # Compute tangent directly using NKI's tan function
    temp = nl.tan(x)
    
    # Store result
    nl.store(result, temp)
    
    return result