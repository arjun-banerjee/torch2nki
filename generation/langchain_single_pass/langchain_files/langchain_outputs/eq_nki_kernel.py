from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_eq(a_tensor, b_tensor):
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=nl.bool_, buffer=nl.shared_hbm)
    
    # Calculate the dimensions
    if len(a_tensor.shape) == 1:
        # For 1D tensors
        sz_p = a_tensor.shape[0]
        trip_count = math.ceil(sz_p/nl.tile_size.pmax)
        
        # Process the tensor in tiles to respect hardware limitations
        for p in nl.affine_range(trip_count):
            # Generate tensor indices for the current tile
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            
            # Load input data from external memory to on-chip memory
            a_tile = nl.load(a_tensor[i_p], mask=(i_p < sz_p))
            b_tile = nl.load(b_tensor[i_p], mask=(i_p < sz_p))
            
            # Compare equality
            out_tile = nl.equal(a_tile, b_tile)
            
            # Store the results back to external memory
            nl.store(result[i_p], value=out_tile, mask=(i_p < sz_p))
    else:
        # For multi-dimensional tensors (handling 2D case)
        sz_p, sz_f = a_tensor.shape
        trip_count = math.ceil(sz_p/nl.tile_size.pmax)
        
        # Generate the free dimension index array
        i_f = nl.arange(sz_f)[None, :]
        
        # Process the tensor in tiles to respect hardware limitations
        for p in nl.affine_range(trip_count):
            # Generate tensor indices for the current tile
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
            
            # Load input data from external memory to on-chip memory
            a_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < sz_p))
            b_tile = nl.load(b_tensor[i_p, i_f], mask=(i_p < sz_p))
            
            # Compare equality
            out_tile = nl.equal(a_tile, b_tile)
            
            # Store the results back to external memory
            nl.store(result[i_p, i_f], value=out_tile, mask=(i_p < sz_p))
    
    return result