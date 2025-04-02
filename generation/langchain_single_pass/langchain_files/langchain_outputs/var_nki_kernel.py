from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_var(a_tensor):
    # Initialize result array in shared HBM
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load tensor into SBUF
    a_sbuf = nl.load(a_tensor)
    
    # Calculate mean with proper broadcasting
    mean = nl.mean(a_sbuf, axis=-1, keepdims=True)
    
    # Calculate variance in PSUM buffer
    diff = a_sbuf - mean
    squared_diff = nl.square(diff)
    variance = nl.mean(squared_diff, axis=-1)
    
    # Store variance result back to HBM
    nl.store(result, variance)
    
    return result