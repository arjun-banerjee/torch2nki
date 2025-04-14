from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_lt(a_tensor, b_tensor):
    # Initialize result array with boolean dtype
    result = nl.ndarray(a_tensor.shape, dtype=nl.bool_, buffer=nl.shared_hbm)
    
    # Calculate tensor dimensions
    sz_p, sz_f = a_tensor.shape
    
    # Calculate the number of tiles needed based on hardware limitations
    trip_count = math.ceil(sz_p/nl.tile_size.pmax)
    
    # Generate the free dimension index array
    i_f = nl.arange(sz_f)[None, :]
    
    # Process the tensor in tiles
    for p in nl.affine_range(trip_count):
        # Generate tensor indices for the current tile
        i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
        
        # Load input data from external memory to on-chip memory
        # Only load up to the actual size of the tensor
        a_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < sz_p))
        b_tile = nl.load(b_tensor[i_p, i_f], mask=(i_p < sz_p))
        
        # Perform element-wise less than comparison
        out_tile = nl.less(a_tile, b_tile)
        
        # Store the results back to external memory
        nl.store(result[i_p, i_f], value=out_tile, mask=(i_p < sz_p))
    
    return result