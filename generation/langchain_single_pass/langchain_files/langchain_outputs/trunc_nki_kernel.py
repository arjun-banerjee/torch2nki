from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_trunc(a_tensor):
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Calculate the number of tiles needed
    shape = a_tensor.shape
    
    # Handle 1D tensor case
    if len(shape) == 1:
        sz_p = shape[0]
        trip_count = math.ceil(sz_p / nl.tile_size.pmax)
        
        # Process the tensor in tiles
        for p in nl.affine_range(trip_count):
            # Generate tensor indices for the current tile
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            
            # Load input data from external memory to on-chip memory
            # Only load up to the actual size of the tensor
            x_tile = nl.load(a_tensor[i_p], mask=(i_p < sz_p))
            
            # Compute trunc using nl.trunc function
            out_tile = nl.trunc(x_tile)
            
            # Store the results back to external memory
            nl.store(result[i_p], value=out_tile, mask=(i_p < sz_p))
    
    # Handle 2D tensor case
    else:
        sz_p, sz_f = shape
        trip_count = math.ceil(sz_p / nl.tile_size.pmax)
        
        # Generate the free dimension index array
        i_f = nl.arange(sz_f)[None, :]
        
        # Process the tensor in tiles
        for p in nl.affine_range(trip_count):
            # Generate tensor indices for the current tile
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
            
            # Load input data from external memory to on-chip memory
            # Only load up to the actual size of the tensor
            x_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < sz_p))
            
            # Compute trunc using nl.trunc function
            out_tile = nl.trunc(x_tile)
            
            # Store the results back to external memory
            nl.store(result[i_p, i_f], value=out_tile, mask=(i_p < sz_p))
    
    return result