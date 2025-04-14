from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_clamp(a_tensor, min=None, max=None):
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Calculate the number of tiles needed
    shape = a_tensor.shape
    
    # Handle tensors of any dimension
    if len(shape) == 1:
        # For 1D tensor
        sz_p = shape[0]
        trip_count = math.ceil(sz_p/nl.tile_size.pmax)
        
        for p in nl.affine_range(trip_count):
            # Generate tensor indices for the current tile
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            
            # Load input data from external memory to on-chip memory
            x_tile = nl.load(a_tensor[i_p], mask=(i_p < sz_p))
            
            # Apply clamp operation using minimum and maximum functions
            if min is not None:
                x_tile = nl.maximum(x_tile, min)
            if max is not None:
                x_tile = nl.minimum(x_tile, max)
            
            # Store the results back to external memory
            nl.store(result[i_p], value=x_tile, mask=(i_p < sz_p))
    else:
        # For multi-dimensional tensors (assuming 2D for simplicity)
        sz_p, sz_f = shape
        trip_count = math.ceil(sz_p/nl.tile_size.pmax)
        
        # Generate the free dimension index array
        i_f = nl.arange(sz_f)[None, :]
        
        for p in nl.affine_range(trip_count):
            # Generate tensor indices for the current tile
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
            
            # Load input data from external memory to on-chip memory
            x_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < sz_p))
            
            # Apply clamp operation using minimum and maximum functions
            if min is not None:
                x_tile = nl.maximum(x_tile, min)
            if max is not None:
                x_tile = nl.minimum(x_tile, max)
            
            # Store the results back to external memory
            nl.store(result[i_p, i_f], value=x_tile, mask=(i_p < sz_p))
    
    return result