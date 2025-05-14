from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get the shape of the input tensor
    shape = a_tensor.shape
    ndim = len(shape)
    
    # Handle negative dimension index
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First, copy the input tensor to the result tensor
    # Calculate the number of tiles needed for copying
    sz_p = 1
    for i in range(ndim-1):
        sz_p *= shape[i]
    
    sz_f = shape[ndim-1]
    trip_count = math.ceil(sz_p / nl.tile_size.pmax)
    
    # Create reshaping indices for copying
    i_f = nl.arange(sz_f)[None, :]
    
    # Copy the input tensor to result
    for p in nl.affine_range(trip_count):
        i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
        
        # Load from input
        x_tile = nl.load(a_tensor.reshape(sz_p, sz_f)[i_p, i_f], mask=(i_p < sz_p))
        
        # Store to result
        nl.store(result.reshape(sz_p, sz_f)[i_p, i_f], value=x_tile, mask=(i_p < sz_p))
    
    # Now perform the sorting along the specified dimension
    # We'll reshape the tensor to make the sort dimension the last dimension
    # This simplifies the sorting logic
    
    # Calculate sizes for reshaping
    pre_dim_size = 1
    for i in range(dim):
        pre_dim_size *= shape[i]
    
    sort_dim_size = shape[dim]
    
    post_dim_size = 1
    for i in range(dim+1, ndim):
        post_dim_size *= shape[i]
    
    # Total size of all dimensions before the sort dimension
    outer_size = pre_dim_size * post_dim_size
    trip_count = math.ceil(outer_size / nl.tile_size.pmax)
    
    # Bubble sort for each slice along the sort dimension
    for n in nl.affine_range(sort_dim_size - 1):
        for p in nl.affine_range(trip_count):
            # Generate indices for current tile
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
            
            # Load adjacent elements for comparison
            curr_indices = nl.zeros((nl.tile_size.pmax, 1), dtype=nl.int32, buffer=nl.sbuf)
            next_indices = nl.zeros((nl.tile_size.pmax, 1), dtype=nl.int32, buffer=nl.sbuf)
            
            # Calculate indices for the current and next elements
            for idx in range(pre_dim_size):
                for jdx in range(post_dim_size):
                    flat_idx = idx * post_dim_size + jdx
                    curr_i = flat_idx // post_dim_size
                    curr_j = flat_idx % post_dim_size
                    
                    # Load current and next elements
                    curr_tile = nl.load(result.reshape(outer_size, sort_dim_size)[i_p, n], mask=(i_p < outer_size))
                    next_tile = nl.load(result.reshape(outer_size, sort_dim_size)[i_p, n+1], mask=(i_p < outer_size))
                    
                    # Compare and swap if needed
                    swap_mask = nl.greater(curr_tile, next_tile)
                    
                    # Create temporary tiles for swapping
                    temp_curr = nl.zeros_like(curr_tile, buffer=nl.sbuf)
                    temp_next = nl.zeros_like(next_tile, buffer=nl.sbuf)
                    
                    # Where swap_mask is true, swap the values
                    temp_curr = nl.where(swap_mask, next_tile, curr_tile)
                    temp_next = nl.where(swap_mask, curr_tile, next_tile)
                    
                    # Store back the swapped values
                    nl.store(result.reshape(outer_size, sort_dim_size)[i_p, n], value=temp_curr, mask=(i_p < outer_size))
                    nl.store(result.reshape(outer_size, sort_dim_size)[i_p, n+1], value=temp_next, mask=(i_p < outer_size))
    
    return result