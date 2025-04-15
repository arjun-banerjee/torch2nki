from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_amax(a_tensor):
    # Get the shape information of the input tensor
    shape = a_tensor.shape
    total_elements = math.prod(shape)
    
    # Initialize result as a single element array with minimum possible value for the dtype
    result = nl.ndarray((), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Initialize max_value based on data type
    if a_tensor.dtype in [nl.float32, nl.float16, nl.bfloat16, nl.tfloat32, nl.float8_e4m3, nl.float8_e5m2]:
        max_value = nl.zeros((), dtype=a_tensor.dtype, buffer=nl.psum)
        # Set to negative infinity (minimum value for the dtype)
        max_value -= nl.load(max_value) - nl.load(max_value)  # This is a trick to get 0
        max_value -= float('inf')  # Set to negative infinity
    else:  # Integer types
        max_value = nl.zeros((), dtype=a_tensor.dtype, buffer=nl.psum)
        # Set to minimum value for the dtype
        if a_tensor.dtype in [nl.int8, nl.int16, nl.int32]:
            max_value -= (1 << (8 * a_tensor.dtype.itemsize - 1))
        else:  # unsigned types
            max_value -= max_value  # Set to 0 for unsigned types
    
    # Reshape input tensor to 2D for easier processing
    sz_p_total = total_elements
    
    # Calculate the number of tiles needed
    tile_size = nl.tile_size.pmax
    trip_count = math.ceil(sz_p_total / tile_size)
    
    # Process the tensor in tiles
    for i in nl.affine_range(trip_count):
        # Calculate indices for current tile
        start_idx = i * tile_size
        
        # Generate tensor indices
        indices = start_idx + nl.arange(tile_size)
        
        # Create a flat view of the tensor for processing
        flat_tensor = a_tensor.reshape(-1)
        
        # Load tile data
        tile_data = nl.load(flat_tensor[indices], mask=(indices < sz_p_total))
        
        # Find maximum in tile
        tile_max = nl.max(tile_data, axis=0)
        
        # Update global maximum if tile_max is greater
        max_value = nl.maximum(max_value, tile_max)
    
    # Store the final maximum value to result
    nl.store(result, max_value)
    
    return result