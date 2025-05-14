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
    # Process the tensor in tiles to respect hardware limitations
    if dim == ndim - 1:  # Last dimension is the sort dimension (common case)
        # Get dimensions
        sort_dim_size = shape[dim]
        
        # Calculate the number of outer elements
        outer_size = 1
        for i in range(ndim - 1):
            outer_size *= shape[i]
        
        # Process in tiles for outer dimensions
        max_tile_size = nl.tile_size.pmax
        trip_count = math.ceil(outer_size / max_tile_size)
        
        for p in nl.affine_range(trip_count):
            # Generate outer indices
            outer_idx = p * max_tile_size + nl.arange(max_tile_size)
            
            # Process only valid indices
            mask = outer_idx < outer_size
            
            for i in range(sort_dim_size):
                # Copy the original tensor to result first
                if i == 0:
                    for j in nl.affine_range(sort_dim_size):
                        # Load data for current position
                        # Reshape outer dimensions to a flat dimension
                        flat_indices = outer_idx[:, None]
                        inner_indices = nl.full((max_tile_size, 1), j, dtype=nl.int32)
                        
                        # Load data
                        data = nl.load(a_tensor.reshape((-1, sort_dim_size))[flat_indices, inner_indices], mask=mask)
                        
                        # Store to result
                        nl.store(result.reshape((-1, sort_dim_size))[flat_indices, inner_indices], value=data, mask=mask)
                
                # Bubble sort implementation
                for j in range(sort_dim_size - 1, i, -1):
                    # Load current and previous elements
                    flat_indices = outer_idx[:, None]
                    curr_indices = nl.full((max_tile_size, 1), j, dtype=nl.int32)
                    prev_indices = nl.full((max_tile_size, 1), j-1, dtype=nl.int32)
                    
                    curr_data = nl.load(result.reshape((-1, sort_dim_size))[flat_indices, curr_indices], mask=mask)
                    prev_data = nl.load(result.reshape((-1, sort_dim_size))[flat_indices, prev_indices], mask=mask)
                    
                    # Compare and swap if needed
                    swap_mask = nl.less(curr_data, prev_data)
                    
                    # Where swap is needed, store the swapped values
                    where_swap = nl.logical_and(swap_mask, mask[:, None])
                    
                    # Store current value to previous position and vice versa when swap is needed
                    nl.store(result.reshape((-1, sort_dim_size))[flat_indices, prev_indices], 
                             value=nl.where(swap_mask, curr_data, prev_data), mask=mask)
                    nl.store(result.reshape((-1, sort_dim_size))[flat_indices, curr_indices], 
                             value=nl.where(swap_mask, prev_data, curr_data), mask=mask)
    else:
        # For other dimensions, transpose to make the sort dimension the last dimension
        # Create new shape with sort dimension as the last
        new_shape = []
        for i in range(ndim):
            if i != dim:
                new_shape.append(shape[i])
        new_shape.append(shape[dim])
        
        # Calculate strides for transposing
        strides = [1]
        for i in range(ndim - 1, 0, -1):
            strides.insert(0, strides[0] * shape[i])
        
        # Reshape input by manually copying with correct indices
        outer_size = 1
        for i in range(ndim):
            if i != dim:
                outer_size *= shape[i]
        
        sort_dim_size = shape[dim]
        
        # Process in tiles for outer dimensions
        max_tile_size = nl.tile_size.pmax
        trip_count = math.ceil(outer_size / max_tile_size)
        
        for p in nl.affine_range(trip_count):
            # Generate outer indices
            flat_outer_idx = p * max_tile_size + nl.arange(max_tile_size)
            
            # Process only valid indices
            mask = flat_outer_idx < outer_size
            
            # First, copy and transpose
            for j in nl.affine_range(sort_dim_size):
                # Calculate multi-dimensional indices from flat index
                indices = []
                remaining = flat_outer_idx.copy()
                dim_idx = 0
                
                for i in range(ndim):
                    if i != dim:
                        div = remaining // strides[i]
                        indices.append(div)
                        remaining = remaining % strides[i]
                        dim_idx += 1
                
                # Get data from the original tensor
                flat_indices = flat_outer_idx[:, None]
                sort_indices = nl.full((max_tile_size, 1), j, dtype=nl.int32)
                
                # Here we need to map from our computed indices back to the original tensor
                # This is complex with NKI's current limitations, so we'll use reshape
                # Load data from the original tensor
                orig_data = nl.load(a_tensor.reshape((-1, sort_dim_size))[flat_indices, sort_indices], mask=mask)
                
                # Store to result with transposed ordering
                nl.store(result.reshape((-1, sort_dim_size))[flat_indices, sort_indices], value=orig_data, mask=mask)
            
            # Now perform bubble sort on each tile
            for i in range(sort_dim_size):
                for j in range(sort_dim_size - 1, i, -1):
                    # Load current and previous elements
                    flat_indices = flat_outer_idx[:, None]
                    curr_indices = nl.full((max_tile_size, 1), j, dtype=nl.int32)
                    prev_indices = nl.full((max_tile_size, 1), j-1, dtype=nl.int32)
                    
                    curr_data = nl.load(result.reshape((-1, sort_dim_size))[flat_indices, curr_indices], mask=mask)
                    prev_data = nl.load(result.reshape((-1, sort_dim_size))[flat_indices, prev_indices], mask=mask)
                    
                    # Compare and swap if needed
                    swap_mask = nl.less(curr_data, prev_data)
                    
                    # Where swap is needed, store the swapped values
                    where_swap = nl.logical_and(swap_mask, mask[:, None])
                    
                    # Store current value to previous position and vice versa when swap is needed
                    nl.store(result.reshape((-1, sort_dim_size))[flat_indices, prev_indices], 
                             value=nl.where(swap_mask, curr_data, prev_data), mask=mask)
                    nl.store(result.reshape((-1, sort_dim_size))[flat_indices, curr_indices], 
                             value=nl.where(swap_mask, prev_data, curr_data), mask=mask)
    
    return result