from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_sort(input_tensor):
    # Initialize result array in HBM
    result = nl.ndarray(input_tensor.shape, dtype=input_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input data
    x = nl.load(input_tensor)
    n = input_tensor.shape[0]
    
    # Store initial values in result
    nl.store(result, x)
    
    # Bubble sort implementation
    for i in nl.affine_range(n):
        curr = nl.load(result)
        
        # Compare adjacent elements
        swap_mask = nl.greater(curr[:-1], curr[1:])
        
        # Perform swaps
        curr_next = nl.where(swap_mask, curr[1:], curr[:-1])
        curr_prev = nl.where(swap_mask, curr[:-1], curr[1:])
        
        # Store back into result
        nl.store(result[:-1], curr_prev)
        nl.store(result[1:], curr_next)
    
    return result