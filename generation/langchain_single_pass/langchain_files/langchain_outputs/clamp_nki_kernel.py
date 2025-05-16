from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_clamp(a_tensor, *, min=None, max=None):
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Calculate the number of tiles needed
    if len(a_tensor.shape) == 1:
        sz_p = a_tensor.shape[0]
        sz_f = 1
        # Reshape for consistent handling
        a_tensor_2d = a_tensor.reshape(sz_p, sz_f)
        result_2d = result.reshape(sz_p, sz_f)
    else:
        sz_p, sz_f = a_tensor.shape
        a_tensor_2d = a_tensor
        result_2d = result
    
    trip_count = math.ceil(sz_p/nl.tile_size.pmax)
    
    # Generate the free dimension index array
    i_f = nl.arange(sz_f)[None, :]
    
    # Process the tensor in tiles to respect hardware limitations
    for p in nl.affine_range(trip_count):
        # Generate tensor indices for the current tile
        i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
        
        # Load input data from external memory to on-chip memory
        x_tile = nl.load(a_tensor_2d[i_p, i_f], mask=(i_p < sz_p))
        
        # Apply minimum and maximum operations to clamp values
        if min is not None and max is not None:
            # Apply both min and max bounds
            clamped_tile = nl.minimum(nl.maximum(x_tile, min), max)
        elif min is not None:
            # Apply only min bound
            clamped_tile = nl.maximum(x_tile, min)
        elif max is not None:
            # Apply only max bound
            clamped_tile = nl.minimum(x_tile, max)
        else:
            # No clamping needed
            clamped_tile = x_tile
        
        # Store the results back to external memory
        nl.store(result_2d[i_p, i_f], value=clamped_tile, mask=(i_p < sz_p))
    
    return result