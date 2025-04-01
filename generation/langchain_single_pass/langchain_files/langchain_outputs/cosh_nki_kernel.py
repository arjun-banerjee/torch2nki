from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_cosh(a_tensor):
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input data from HBM to on-chip memory
    x = nl.load(a_tensor)
    
    # Calculate e^x
    pos_exp = nl.exp(x)
    
    # Calculate e^(-x) by negating x first
    neg_exp = nl.exp(nl.multiply(x, -1.0))
    
    # Calculate (e^x + e^-x)
    sum_exp = nl.add(pos_exp, neg_exp)
    
    # Divide by 2 to get final result
    temp_result = nl.multiply(sum_exp, 0.5)
    
    # Store result back to HBM
    nl.store(result, temp_result)
    
    return result