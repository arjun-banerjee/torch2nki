from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_mode(a_tensor, dim=-1):
    # Handle the dim parameter (convert negative indices)
    if dim < 0:
        dim = len(a_tensor.shape) + dim
    
    # Get shape information
    input_shape = a_tensor.shape
    
    # Convert int64 to int32 if needed
    input_dtype = a_tensor.dtype
    if input_dtype == nl.int64:
        input_dtype = nl.int32
    
    # Initialize result arrays for values and counts
    # For the most frequent value and its index
    result = nl.ndarray((1,), dtype=input_dtype, buffer=nl.shared_hbm)
    indices = nl.ndarray((1,), dtype=nl.int32, buffer=nl.shared_hbm)
    
    # Get the size of the input tensor
    input_size = a_tensor.shape[0]
    
    # If input is empty, return empty results
    if input_size == 0:
        return result, indices
    
    # Maximum frequency seen so far
    max_freq = nl.zeros((), dtype=nl.int32, buffer=nl.psum)
    
    # Current mode value
    mode_value = nl.zeros((), dtype=input_dtype, buffer=nl.shared_hbm)
    
    # Calculate the number of tiles needed
    trip_count = math.ceil(input_size/nl.tile_size.pmax)
    
    # Process the tensor in tiles to respect hardware limitations
    for i in nl.affine_range(input_size):
        # Current element to check
        current_value = nl.load(a_tensor[i])
        current_freq = nl.zeros((), dtype=nl.int32, buffer=nl.psum)
        
        # Count frequency of the current value
        for j in nl.affine_range(input_size):
            value_j = nl.load(a_tensor[j])
            is_equal = nl.equal(current_value, value_j)
            current_freq += nl.cast(is_equal, dtype=nl.int32)
        
        # Update mode if we found a higher frequency
        is_higher = current_freq > max_freq
        max_freq = nl.maximum(max_freq, current_freq)
        
        # Update the mode value if this has a higher frequency
        if is_higher:
            mode_value = current_value
            nl.store(indices[0], nl.cast(i, dtype=nl.int32))
    
    # Store the mode value in the result
    nl.store(result[0], mode_value)
    
    return result, indices