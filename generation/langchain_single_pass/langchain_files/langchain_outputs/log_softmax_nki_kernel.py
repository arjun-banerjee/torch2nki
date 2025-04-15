from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_log_softmax(a_tensor):
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Calculate the number of tiles needed
    sz_p, sz_f = a_tensor.shape
    trip_count = math.ceil(sz_p/nl.tile_size.pmax)
    
    # Generate the free dimension index array
    i_f = nl.arange(sz_f)[None, :]
    
    # Process the tensor in tiles to respect hardware limitations
    for p in nl.affine_range(trip_count):
        # Generate tensor indices for the current tile
        i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
        
        # Load input data from external memory to on-chip memory
        # Only load up to the actual size of the tensor
        x_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < sz_p))
        
        # Step 1: Find the maximum value for each row (for numerical stability)
        max_vals = nl.max(x_tile, axis=1, keepdims=True)
        
        # Step 2: Shift values by subtracting the max (for numerical stability)
        shifted_x = nl.subtract(x_tile, max_vals)
        
        # Step 3: Compute exp of the shifted values
        exp_x = nl.exp(shifted_x)
        
        # Step 4: Sum the exp values along each row
        sum_exp = nl.sum(exp_x, axis=1, keepdims=True)
        
        # Step 5: Take the log of the sum
        log_sum = nl.log(sum_exp)
        
        # Step 6: Compute log_softmax = shifted_x - log_sum
        log_softmax_tile = nl.subtract(shifted_x, log_sum)
        
        # Store the results back to external memory
        nl.store(result[i_p, i_f], value=log_softmax_tile, mask=(i_p < sz_p))
    
    return result