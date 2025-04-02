from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_softmax(a_tensor):
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input from HBM to SBUF
    input_sbuf = nl.load(a_tensor)
    
    # Apply softmax with explicit parameters
    temp = nl.softmax(input_sbuf, axis=-1, compute_dtype=nl.float32)
    nl.store(result, temp)
    
    return result