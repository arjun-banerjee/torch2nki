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
    # For 1D case
    if ndim == 1:
        sz = shape[0]
        trip_count = math.ceil(sz / nl.tile_size.pmax)
        
        for p in nl.affine_range(trip_count):
            # Generate indices for the current tile
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            
            # Load input data with masking for boundary
            x_tile = nl.load(a_tensor[i_p], mask=(i_p < sz))
            
            # Store to result
            nl.store(result[i_p], value=x_tile, mask=(i_p < sz))
            
    # For multi-dimensional case
    else:
        # Handle different dimensions separately
        dim_size = shape[dim]
        
        # Calculate the product of dimensions before the sorting dimension
        outer_dims = 1
        for i in range(dim):
            outer_dims *= shape[i]
            
        # Calculate the product of dimensions after the sorting dimension
        inner_dims = 1
        for i in range(dim + 1, ndim):
            inner_dims *= shape[i]
            
        # Process in chunks to respect hardware limitations
        outer_trip_count = math.ceil(outer_dims / nl.tile_size.pmax)
        
        for o in nl.affine_range(outer_trip_count):
            # Generate indices for outer dimensions
            i_o = o * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None, None]
            i_d = nl.arange(dim_size)[None, :, None]
            i_i = nl.arange(inner_dims)[None, None, :]
            
            # Load the full slice with masking for boundary
            x_tile = nl.load(a_tensor.reshape((outer_dims, dim_size, inner_dims))[i_o, i_d, i_i], 
                             mask=(i_o < outer_dims))
            
            # Store to result
            nl.store(result.reshape((outer_dims, dim_size, inner_dims))[i_o, i_d, i_i], 
                     value=x_tile, mask=(i_o < outer_dims))
    
    # Now perform bubble sort on the copied data
    if ndim == 1:
        # For 1D tensor, sort the entire array
        sz = shape[0]
        
        # Outer loop for bubble sort
        for i in nl.static_range(sz - 1):
            # Inner loop for each pass of bubble sort
            for j in nl.static_range(sz - 1 - i):
                # Process in chunks to respect hardware limitations
                trip_count = math.ceil(sz / nl.tile_size.pmax)
                
                for p in nl.affine_range(trip_count):
                    # Generate indices for the current and next elements
                    i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                    
                    # Load current and next elements with masking
                    curr = nl.load(result[i_p], mask=((i_p < sz - 1 - i) & (i_p >= j) & (i_p < j + 1)))
                    next_val = nl.load(result[i_p + 1], mask=((i_p < sz - 1 - i) & (i_p >= j) & (i_p < j + 1)))
                    
                    # Compare and swap if needed
                    mask = ((i_p < sz - 1 - i) & (i_p >= j) & (i_p < j + 1) & (curr > next_val))
                    
                    # Store swapped values if needed
                    nl.store(result[i_p], value=next_val, mask=mask)
                    nl.store(result[i_p + 1], value=curr, mask=mask)
    
    else:
        # For multi-dimensional tensor, sort along the specified dimension
        dim_size = shape[dim]
        
        # Calculate the product of dimensions before the sorting dimension
        outer_dims = 1
        for i in range(dim):
            outer_dims *= shape[i]
            
        # Calculate the product of dimensions after the sorting dimension
        inner_dims = 1
        for i in range(dim + 1, ndim):
            inner_dims *= shape[i]
        
        # Process in chunks to respect hardware limitations
        outer_trip_count = math.ceil(outer_dims / nl.tile_size.pmax)
        
        # Bubble sort algorithm
        for i in nl.static_range(dim_size - 1):
            for j in nl.static_range(dim_size - 1 - i):
                for o in nl.affine_range(outer_trip_count):
                    # Generate indices for the outer dimensions
                    i_o = o * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
                    
                    # Create mask for valid outer indices
                    outer_mask = (i_o < outer_dims)
                    
                    # Create inner indices
                    i_i = nl.arange(inner_dims)[None, :]
                    
                    # Load current and next elements
                    curr = nl.load(result.reshape((outer_dims, dim_size, inner_dims))[i_o, j, i_i], mask=outer_mask)
                    next_val = nl.load(result.reshape((outer_dims, dim_size, inner_dims))[i_o, j+1, i_i], mask=outer_mask)
                    
                    # Compare and create swap mask
                    swap_mask = nl.greater(curr, next_val) & outer_mask[:, None]
                    
                    # Conditionally swap elements
                    temp_curr = nl.zeros(curr.shape, dtype=curr.dtype, buffer=nl.sbuf)
                    temp_next = nl.zeros(next_val.shape, dtype=next_val.dtype, buffer=nl.sbuf)
                    
                    # Copy values based on swap condition
                    for ii in nl.static_range(inner_dims):
                        for oo in nl.static_range(nl.tile_size.pmax):
                            if oo < outer_dims and swap_mask[oo, ii]:
                                temp_curr[oo, ii] = next_val[oo, ii]
                                temp_next[oo, ii] = curr[oo, ii]
                            else:
                                temp_curr[oo, ii] = curr[oo, ii]
                                temp_next[oo, ii] = next_val[oo, ii]
                    
                    # Store the swapped values
                    nl.store(result.reshape((outer_dims, dim_size, inner_dims))[i_o, j, i_i], 
                             value=temp_curr, mask=outer_mask)
                    nl.store(result.reshape((outer_dims, dim_size, inner_dims))[i_o, j+1, i_i], 
                             value=temp_next, mask=outer_mask)
    
    return result