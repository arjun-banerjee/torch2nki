from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_round(a_tensor, decimals=0):
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Calculate the number of tiles needed based on tensor shape
    if len(a_tensor.shape) == 1:
        sz_p = a_tensor.shape[0]
        sz_f = 1
        # Handle 1D input by treating it as 2D with 1 column
        two_d_shape = True
    else:
        sz_p, sz_f = a_tensor.shape
        two_d_shape = False
    
    trip_count = math.ceil(sz_p/nl.tile_size.pmax)
    
    # Calculate multiplier based on decimals
    multiplier = nl.full((1, 1), 10 ** decimals, dtype=a_tensor.dtype)
    
    # Generate the free dimension index array
    i_f = nl.arange(sz_f)[None, :]
    
    # Process the tensor in tiles to respect hardware limitations
    for p in nl.affine_range(trip_count):
        # Generate tensor indices for the current tile
        i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
        
        # Load input data from external memory to on-chip memory
        if two_d_shape:
            # Load 1D input as 2D with proper masking
            x_tile = nl.load(a_tensor[i_p[:, 0]], mask=(i_p[:, 0] < sz_p))
            x_tile = x_tile[:, None]  # Add dimension for consistent processing
        else:
            # Load 2D input with proper masking
            x_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < sz_p))
        
        # Scale by multiplier for decimal handling
        scaled = nl.multiply(x_tile, multiplier)
        
        # Determine if values are positive or negative
        is_positive = nl.greater_equal(scaled, 0)
        
        # Round positive numbers using floor(x + 0.5)
        positive_values = nl.floor(nl.add(scaled, 0.5))
        
        # Round negative numbers using ceil(x - 0.5)
        negative_values = nl.ceil(nl.add(scaled, -0.5))
        
        # Select appropriate values based on sign
        rounded = nl.where(is_positive, positive_values, negative_values)
        
        # Scale back by dividing by multiplier
        out_tile = nl.divide(rounded, multiplier)
        
        # Store the results back to external memory
        if two_d_shape:
            # Store 1D output with proper masking
            nl.store(result[i_p[:, 0]], value=out_tile[:, 0], mask=(i_p[:, 0] < sz_p))
        else:
            # Store 2D output with proper masking
            nl.store(result[i_p, i_f], value=out_tile, mask=(i_p < sz_p))
    
    return result