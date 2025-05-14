from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get shape information
    shape = a_tensor.shape
    ndim = len(shape)
    
    # Handle negative dimension indexing
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result with same shape and dtype as input
    result = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First copy input tensor to result
    if ndim == 1:
        # For 1D tensor, copy directly
        size = shape[0]
        trip_count = math.ceil(size / nl.tile_size.pmax)
        
        for p in nl.affine_range(trip_count):
            # Generate indices for the current tile
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            
            # Load input data, only up to size
            x_tile = nl.load(a_tensor[i_p], mask=(i_p < size))
            
            # Store to result
            nl.store(result[i_p], value=x_tile, mask=(i_p < size))
    elif ndim == 2:
        # For 2D tensor
        sz_0, sz_1 = shape[0], shape[1]
        trip_count_0 = math.ceil(sz_0 / nl.tile_size.pmax)
        
        for p in nl.affine_range(trip_count_0):
            # Generate indices for the current tile
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
            i_f = nl.arange(sz_1)[None, :]
            
            # Load input data, only up to size
            x_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < sz_0))
            
            # Store to result
            nl.store(result[i_p, i_f], value=x_tile, mask=(i_p < sz_0))
    
    # Now perform bubble sort on the result tensor
    if ndim == 1:
        # For 1D tensor, sort the entire array
        size = shape[0]
        
        # Bubble sort algorithm
        for i in nl.affine_range(size):
            # Use a smaller trip count to avoid exceeding hardware limits
            trip_count = math.ceil(size / nl.tile_size.pmax)
            
            for p in nl.affine_range(trip_count):
                # Generate indices for the current tile
                i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                i_p_next = i_p + 1
                
                # Load current elements
                curr_elems = nl.load(result[i_p], mask=((i_p < size - 1) & (i_p < size)))
                # Load next elements
                next_elems = nl.load(result[i_p_next], mask=((i_p_next < size) & (i_p < size - 1)))
                
                # Compare and swap if current > next
                swap_needed = nl.greater(curr_elems, next_elems)
                
                # Where swap is needed, prepare new values
                new_curr = nl.where(swap_needed, next_elems, curr_elems)
                new_next = nl.where(swap_needed, curr_elems, next_elems)
                
                # Store the swapped values
                nl.store(result[i_p], value=new_curr, mask=((i_p < size - 1) & (i_p < size)))
                nl.store(result[i_p_next], value=new_next, mask=((i_p_next < size) & (i_p < size - 1)))
                
    elif ndim == 2:
        # Sort along the specified dimension
        if dim == 0:
            # Sort along rows (dimension 0)
            sz_0, sz_1 = shape[0], shape[1]
            
            # Bubble sort algorithm - sorting each column
            for i in nl.affine_range(sz_0):
                trip_count_1 = math.ceil(sz_1 / nl.tile_size.pmax)
                
                for f in nl.affine_range(trip_count_1):
                    i_f = f * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                    
                    for j in nl.affine_range(sz_0 - 1):
                        # Load elements to compare
                        curr_elems = nl.load(result[j, i_f], mask=(i_f < sz_1))
                        next_elems = nl.load(result[j+1, i_f], mask=(i_f < sz_1))
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(curr_elems, next_elems)
                        
                        # Prepare new values
                        new_curr = nl.where(swap_needed, next_elems, curr_elems)
                        new_next = nl.where(swap_needed, curr_elems, next_elems)
                        
                        # Store the swapped values
                        nl.store(result[j, i_f], value=new_curr, mask=(i_f < sz_1))
                        nl.store(result[j+1, i_f], value=new_next, mask=(i_f < sz_1))
        else:
            # Sort along columns (dimension 1)
            sz_0, sz_1 = shape[0], shape[1]
            trip_count_0 = math.ceil(sz_0 / nl.tile_size.pmax)
            
            for p in nl.affine_range(trip_count_0):
                # Generate indices for the current tile of rows
                i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
                
                # For each valid row in this tile
                for j in nl.affine_range(sz_1):
                    for k in nl.affine_range(sz_1 - 1):
                        # Load elements to compare
                        curr_elems = nl.load(result[i_p, k], mask=(i_p < sz_0))
                        next_elems = nl.load(result[i_p, k+1], mask=(i_p < sz_0))
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(curr_elems, next_elems)
                        
                        # Prepare new values
                        new_curr = nl.where(swap_needed, next_elems, curr_elems)
                        new_next = nl.where(swap_needed, curr_elems, next_elems)
                        
                        # Store the swapped values
                        nl.store(result[i_p, k], value=new_curr, mask=(i_p < sz_0))
                        nl.store(result[i_p, k+1], value=new_next, mask=(i_p < sz_0))
    
    return result