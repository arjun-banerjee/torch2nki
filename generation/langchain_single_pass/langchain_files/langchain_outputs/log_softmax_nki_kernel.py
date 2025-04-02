from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_log_softmax(a_tensor):
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input data from HBM
    input_data = nl.load(a_tensor)
    
    # Compute max for numerical stability
    max_vals = nl.max(input_data, axis=1, keepdims=True)
    
    # Shift input by max values
    shifted = nl.subtract(input_data, max_vals)
    
    # Compute exp(x - max(x))
    exp_shifted = nl.exp(shifted)
    
    # Compute sum(exp(x - max(x)))
    sum_exp = nl.sum(exp_shifted, axis=1, keepdims=True)
    
    # Compute log(sum(exp(x - max(x))))
    log_sum = nl.log(sum_exp)
    
    # Final result: x - max(x) - log(sum(exp(x - max(x))))
    log_softmax_result = nl.subtract(shifted, log_sum)
    
    # Store result back to HBM
    nl.store(result, log_softmax_result)
    
    return result