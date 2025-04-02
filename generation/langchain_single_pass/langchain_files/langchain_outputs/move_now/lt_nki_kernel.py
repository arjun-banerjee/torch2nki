from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_lt(a_tensor, b_tensor):
    # Initialize result array in shared HBM
    result = nl.ndarray(a_tensor.shape, dtype=nl.bool_, buffer=nl.shared_hbm)
    
    # Load tensors from HBM to SBUF
    a_sbuf = nl.load(a_tensor)
    b_sbuf = nl.load(b_tensor)
    
    # Perform less than comparison on SBUF tensors
    lt_output = nl.less(a_sbuf, b_sbuf)
    
    # Store result back to HBM
    nl.store(result, lt_output)
    
    return result