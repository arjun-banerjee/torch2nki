from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_floor(a_tensor):
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Calculate dimensions and tiling strategy
    tensor_shape = a_tensor.shape
    
    # For 1D tensor
    if len(tensor_shape) == 1:
        sz_p = tensor_shape[0]
        trip_count = math.ceil(sz_p/nl.tile_size.pmax)
        
        for p in nl.affine_range(trip_count):
            # Generate tensor indices for the current tile
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            
            # Load input data from external memory to on-chip memory
            # Only load up to the actual size of the tensor
            x_tile = nl.load(a_tensor[i_p], mask=(i_p < sz_p))
            
            # Compute floor using nl.floor function
            out_tile = nl.floor(x_tile)
            
            # Store the results back to external memory
            nl.store(result[i_p], value=out_tile, mask=(i_p < sz_p))
    
    # For 2D or higher-dimensional tensor
    else:
        sz_p, sz_f = tensor_shape[0], tensor_shape[1:]
        flat_sz_f = 1
        for dim in sz_f:
            flat_sz_f *= dim
            
        trip_count = math.ceil(sz_p/nl.tile_size.pmax)
        
        # Generate the free dimension index array for multi-dimensional case
        if len(tensor_shape) == 2:
            i_f = nl.arange(sz_f[0])[None, :]
        else:
            # For higher dimensions, we'll use flat indexing and reshape later
            i_f = nl.arange(flat_sz_f)[None, :]
        
        for p in nl.affine_range(trip_count):
            # Generate tensor indices for the current tile
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
            
            # Load input data from external memory to on-chip memory
            x_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < sz_p))
            
            # Compute floor using nl.floor function
            out_tile = nl.floor(x_tile)
            
            # Store the results back to external memory
            nl.store(result[i_p, i_f], value=out_tile, mask=(i_p < sz_p))
    
    return result