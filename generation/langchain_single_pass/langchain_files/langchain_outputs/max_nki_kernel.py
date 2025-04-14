from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_max(a_tensor, b_tensor):
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
        a_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < sz_p))
        b_tile = nl.load(b_tensor[i_p, i_f], mask=(i_p < sz_p))
        
        # Compute element-wise maximum
        out_tile = nl.maximum(a_tile, b_tile)
        
        # Store the results back to external memory
        nl.store(result[i_p, i_f], value=out_tile, mask=(i_p < sz_p))
    
    return result