from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_ne(a_tensor, b_tensor):
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=nl.bool_, buffer=nl.shared_hbm)
    
    # Get the dimensions of the input tensors
    sz_p, sz_f = a_tensor.shape
    
    # Calculate the number of tiles needed
    trip_count = math.ceil(sz_p/nl.tile_size.pmax)
    
    # Generate the free dimension index array
    i_f = nl.arange(sz_f)[None, :]
    
    # Process the tensor in tiles to respect hardware limitations
    for p in nl.affine_range(trip_count):
        # Generate tensor indices for the current tile
        i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
        
        # Load input data from external memory to on-chip memory
        # Only load up to the actual size of the tensor
        a_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < sz_p))
        b_tile = nl.load(b_tensor[i_p, i_f], mask=(i_p < sz_p))
        
        # Compute the element-wise not equal comparison
        out_tile = nl.not_equal(a_tile, b_tile)
        
        # Store the results back to external memory
        nl.store(result[i_p, i_f], value=out_tile, mask=(i_p < sz_p))
    
    return result