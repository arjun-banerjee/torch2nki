from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_where(condition, x, y):
    # Initialize result array with same shape and dtype as x
    result = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    
    # Calculate the number of tiles needed
    sz_p, sz_f = condition.shape
    trip_count = math.ceil(sz_p/nl.tile_size.pmax)
    
    # Generate the free dimension index array
    i_f = nl.arange(sz_f)[None, :]
    
    # Process the tensor in tiles to respect hardware limitations
    for p in nl.affine_range(trip_count):
        # Generate tensor indices for the current tile
        i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
        
        # Load input data from external memory to on-chip memory
        # Only load up to the actual size of the tensor
        cond_tile = nl.load(condition[i_p, i_f], mask=(i_p < sz_p))
        x_tile = nl.load(x[i_p, i_f], mask=(i_p < sz_p))
        y_tile = nl.load(y[i_p, i_f], mask=(i_p < sz_p))
        
        # Compute where operation using nl.where function
        out_tile = nl.where(cond_tile, x_tile, y_tile)
        
        # Store the results back to external memory
        nl.store(result[i_p, i_f], value=out_tile, mask=(i_p < sz_p))
    
    return result