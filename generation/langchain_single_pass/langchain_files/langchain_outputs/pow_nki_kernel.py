from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_pow(a_tensor, exp_tensor):
    # Initialize result array
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load tensors into SBUF memory space
    a_sbuf = nl.load(a_tensor)
    exp_sbuf = nl.load(exp_tensor)
    
    # Perform power operation on SBUF data
    temp = nl.power(a_sbuf, exp_sbuf)
    
    # Store result back to HBM
    nl.store(result, temp)
    
    return result