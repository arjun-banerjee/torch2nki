from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get shape information
    shape = a_tensor.shape
    ndim = len(shape)
    
    # If dim is negative, convert to positive index
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First, copy input to result since we'll sort in-place
    if ndim == 1:
        # For 1D tensors
        size = shape[0]
        trip_count = math.ceil(size / nl.tile_size.pmax)
        
        for p in nl.affine_range(trip_count):
            # Generate indices for current tile
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            
            # Load input data
            in_tile = nl.load(a_tensor[i_p], mask=(i_p < size))
            
            # Store to result
            nl.store(result[i_p], value=in_tile, mask=(i_p < size))
            
        # Bubble sort implementation for 1D tensor
        for i in nl.affine_range(size):
            for j in nl.affine_range(size - 1):
                # Need to load elements in tiles
                trip_count = math.ceil((size - 1) / nl.tile_size.pmax)
                
                for p in nl.affine_range(trip_count):
                    # Generate indices for current tile
                    i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                    
                    # Only process valid indices less than size-1
                    mask = (i_p < size - 1)
                    
                    # Load pairs of elements to compare
                    curr = nl.load(result[i_p], mask=mask)
                    next_elem = nl.load(result[i_p + 1], mask=mask)
                    
                    # Compare and swap if needed
                    swap_needed = nl.greater(curr, next_elem)
                    
                    # Prepare swapped values
                    new_curr = nl.where(swap_needed, next_elem, curr)
                    new_next = nl.where(swap_needed, curr, next_elem)
                    
                    # Store the swapped values
                    nl.store(result[i_p], value=new_curr, mask=mask)
                    nl.store(result[i_p + 1], value=new_next, mask=mask)
    else:
        # For multi-dimensional tensors, we need to sort along specified dimension
        # Determine sizes and reshape logic for the sort dimension
        sort_dim_size = shape[dim]
        
        # Create a new shape where the sort dimension is the last dimension
        # This simplifies our implementation
        
        # First copy the tensor to result
        if dim == ndim - 1:  # If already the last dimension, no need for complicated logic
            # Calculate total elements and tiles needed
            total_elements = 1
            for s in shape:
                total_elements *= s
            
            outer_dims_size = total_elements // sort_dim_size
            outer_trip_count = math.ceil(outer_dims_size / nl.tile_size.pmax)
            
            for outer in nl.affine_range(outer_trip_count):
                outer_indices = outer * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
                
                inner_indices = nl.arange(sort_dim_size)[None, :]
                
                # Load and store data
                in_tile = nl.load(a_tensor.reshape(outer_dims_size, sort_dim_size)[outer_indices, inner_indices], 
                                 mask=(outer_indices < outer_dims_size))
                
                nl.store(result.reshape(outer_dims_size, sort_dim_size)[outer_indices, inner_indices], 
                        value=in_tile, mask=(outer_indices < outer_dims_size))
            
            # Now sort each "row" separately
            for outer in nl.affine_range(outer_dims_size):
                # Bubble sort for this row
                for i in nl.affine_range(sort_dim_size):
                    for j in nl.affine_range(sort_dim_size - 1):
                        # Load adjacent elements
                        curr = nl.load(result.reshape(outer_dims_size, sort_dim_size)[outer, j])
                        next_elem = nl.load(result.reshape(outer_dims_size, sort_dim_size)[outer, j + 1])
                        
                        # Compare and swap if needed
                        if nl.greater(curr, next_elem):
                            # Swap elements
                            nl.store(result.reshape(outer_dims_size, sort_dim_size)[outer, j], value=next_elem)
                            nl.store(result.reshape(outer_dims_size, sort_dim_size)[outer, j + 1], value=curr)
        else:
            # For now, we only support sorting along the last dimension
            # Copy input to output
            flat_size = 1
            for s in shape:
                flat_size *= s
                
            trip_count = math.ceil(flat_size / nl.tile_size.pmax)
            
            for p in nl.affine_range(trip_count):
                i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                
                # Load input data
                in_tile = nl.load(a_tensor.reshape(flat_size)[i_p], mask=(i_p < flat_size))
                
                # Store to result
                nl.store(result.reshape(flat_size)[i_p], value=in_tile, mask=(i_p < flat_size))
    
    return result