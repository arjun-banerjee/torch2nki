from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_linalg_vecdot(a_tensor, b_tensor):
    # Initialize result as a scalar in float32
    result = nl.zeros((), dtype=nl.float32, buffer=nl.shared_hbm)
    
    # Convert inputs to float32 and reshape for matmul
    a_reshaped = nl.reshape(nl.cast(a_tensor, dtype=nl.float32), (1, -1))
    b_reshaped = nl.reshape(nl.cast(b_tensor, dtype=nl.float32), (-1, 1))
    
    # Compute dot product using matrix multiplication
    dummy = nl.matmul(a_reshaped, b_reshaped, dtype=nl.float32)
    
    # Store result
    nl.store(result, nl.reshape(dummy, ()))
    
    return result