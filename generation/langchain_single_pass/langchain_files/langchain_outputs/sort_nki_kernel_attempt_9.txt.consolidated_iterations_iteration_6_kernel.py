from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get tensor shape
    tensor_shape = a_tensor.shape
    ndim = len(tensor_shape)
    
    # Handle negative dimensions
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Handle 1D tensor case
    if ndim == 1:
        size = tensor_shape[0]
        # Copy input to result first
        i_range = nl.arange(size)
        nl.store(result[i_range], nl.load(a_tensor[i_range]))
        
        # Bubble sort implementation
        for i in range(size):
            for j in range(size - i - 1):
                # Create indices for current and next element
                idx_j = nl.full((1,), j, dtype=nl.int32)
                idx_j_next = nl.full((1,), j+1, dtype=nl.int32)
                
                # Load elements
                curr_val = nl.load(result[idx_j])
                next_val = nl.load(result[idx_j_next])
                
                # Compare and swap if needed
                swap_needed = nl.greater(curr_val, next_val)
                
                # Swap elements if needed
                temp_val = curr_val
                curr_val = nl.where(swap_needed, next_val, curr_val)
                next_val = nl.where(swap_needed, temp_val, next_val)
                
                # Store updated values
                nl.store(result[idx_j], curr_val)
                nl.store(result[idx_j_next], next_val)
    
    # Handle 2D tensor case
    elif ndim == 2:
        dim0_size = tensor_shape[0]
        dim1_size = tensor_shape[1]
        
        # Sort along dimension 0
        if dim == 0:
            # Process in tiles to respect architecture limitations
            trip_count = math.ceil(dim1_size / nl.tile_size.pmax)
            
            for f in nl.affine_range(trip_count):
                # Generate tensor indices for the current tile
                f_start = f * nl.tile_size.pmax
                i_f = f_start + nl.arange(min(nl.tile_size.pmax, dim1_size - f_start))[None, :]
                
                # Copy input to result first for this column
                for p in range(dim0_size):
                    idx_p = nl.full((1, 1), p, dtype=nl.int32)
                    val = nl.load(a_tensor[idx_p, i_f], mask=(i_f[0, :] < dim1_size))
                    nl.store(result[idx_p, i_f], val, mask=(i_f[0, :] < dim1_size))
                
                # Bubble sort each column
                for i in range(dim0_size):
                    for j in range(dim0_size - i - 1):
                        # Create indices for current and next row
                        idx_j = nl.full((1, 1), j, dtype=nl.int32)
                        idx_j_next = nl.full((1, 1), j+1, dtype=nl.int32)
                        
                        # Load elements
                        curr_vals = nl.load(result[idx_j, i_f], mask=(i_f[0, :] < dim1_size))
                        next_vals = nl.load(result[idx_j_next, i_f], mask=(i_f[0, :] < dim1_size))
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(curr_vals, next_vals)
                        
                        # Swap elements if needed
                        temp_vals = curr_vals
                        curr_vals = nl.where(swap_needed, next_vals, curr_vals)
                        next_vals = nl.where(swap_needed, temp_vals, next_vals)
                        
                        # Store updated values
                        nl.store(result[idx_j, i_f], curr_vals, mask=(i_f[0, :] < dim1_size))
                        nl.store(result[idx_j_next, i_f], next_vals, mask=(i_f[0, :] < dim1_size))
        
        # Sort along dimension 1
        else:
            # Process in tiles to respect architecture limitations
            trip_count = math.ceil(dim0_size / nl.tile_size.pmax)
            
            for p in nl.affine_range(trip_count):
                # Generate tensor indices for the current tile
                p_start = p * nl.tile_size.pmax
                i_p = p_start + nl.arange(min(nl.tile_size.pmax, dim0_size - p_start))[:, None]
                i_f = nl.arange(dim1_size)[None, :]
                
                # Load input data from external memory to on-chip memory
                x_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p[:, 0] < dim0_size))
                
                # Sort each row using bubble sort
                for i in range(dim1_size):
                    for j in range(dim1_size - i - 1):
                        # Create indices for current and next column
                        j_idx = nl.full((1, 1), j, dtype=nl.int32)
                        j_next_idx = nl.full((1, 1), j+1, dtype=nl.int32)
                        
                        # Extract current and next columns
                        curr_vals = nl.load(x_tile[:, j_idx], mask=(i_p[:, 0] < dim0_size))
                        next_vals = nl.load(x_tile[:, j_next_idx], mask=(i_p[:, 0] < dim0_size))
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(curr_vals, next_vals)
                        
                        # Perform the swap
                        temp_vals = curr_vals
                        curr_vals = nl.where(swap_needed, next_vals, curr_vals)
                        next_vals = nl.where(swap_needed, temp_vals, next_vals)
                        
                        # Update the values in x_tile
                        nl.store(x_tile[:, j_idx], curr_vals, mask=(i_p[:, 0] < dim0_size))
                        nl.store(x_tile[:, j_next_idx], next_vals, mask=(i_p[:, 0] < dim0_size))
                
                # Store the sorted tile back to result
                nl.store(result[i_p, i_f], x_tile, mask=(i_p[:, 0] < dim0_size))
    
    # Handle higher dimensional tensors by reshaping
    else:
        # Get the size of the dimension to sort along
        dim_size = tensor_shape[dim]
        
        # First copy input to result
        for p in nl.affine_range(math.ceil(a_tensor.size / nl.tile_size.pmax)):
            # Calculate flat indices
            start_idx = p * nl.tile_size.pmax
            flat_indices = start_idx + nl.arange(min(nl.tile_size.pmax, a_tensor.size - start_idx))
            
            # Load and store values
            vals = nl.load(a_tensor.reshape(-1)[flat_indices], mask=(flat_indices < a_tensor.size))
            nl.store(result.reshape(-1)[flat_indices], vals, mask=(flat_indices < a_tensor.size))
        
        # Calculate the number of slices to sort
        total_elements = 1
        for i in range(ndim):
            if i != dim:
                total_elements *= tensor_shape[i]
        
        # Sort each slice along the specified dimension
        for slice_idx in range(total_elements):
            # Calculate multi-dimensional indices for this slice
            indices = []
            remaining = slice_idx
            for i in range(ndim):
                if i != dim:
                    dim_size_i = tensor_shape[i]
                    indices.append(remaining % dim_size_i)
                    remaining //= dim_size_i
                else:
                    indices.append(0)  # Placeholder, will be replaced in the inner loop
            
            # Sort this slice using bubble sort
            for i in range(dim_size):
                for j in range(dim_size - i - 1):
                    # Set indices for current and next element
                    indices[dim] = j
                    curr_idx = tuple(indices)
                    indices[dim] = j + 1
                    next_idx = tuple(indices)
                    
                    # Create index arrays
                    curr_idx_array = []
                    next_idx_array = []
                    for idx_val in curr_idx:
                        curr_idx_array.append(nl.full((1,), idx_val, dtype=nl.int32))
                    for idx_val in next_idx:
                        next_idx_array.append(nl.full((1,), idx_val, dtype=nl.int32))
                    
                    # Load elements
                    curr_val = nl.load(result[tuple(curr_idx_array)])
                    next_val = nl.load(result[tuple(next_idx_array)])
                    
                    # Compare and swap if needed
                    swap_needed = nl.greater(curr_val, next_val)
                    
                    # Swap elements if needed
                    temp_val = curr_val
                    curr_val = nl.where(swap_needed, next_val, curr_val)
                    next_val = nl.where(swap_needed, temp_val, next_val)
                    
                    # Store updated values
                    nl.store(result[tuple(curr_idx_array)], curr_val)
                    nl.store(result[tuple(next_idx_array)], next_val)
    
    return result