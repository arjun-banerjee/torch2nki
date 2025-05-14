from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get tensor shape and handle negative dimension
    shape = a_tensor.shape
    ndim = len(shape)
    
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result array for sorted values with same shape and dtype as input
    result = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    # Initialize indices array with same shape as input but int32 dtype
    indices = nl.ndarray(shape, dtype=nl.int32, buffer=nl.shared_hbm)
    
    # Handle 1D tensor case
    if ndim == 1:
        size = shape[0]
        max_tile_size = min(nl.tile_size.pmax, size)
        
        # Copy input to result first
        for p in nl.affine_range(math.ceil(size / nl.tile_size.pmax)):
            start_idx = p * nl.tile_size.pmax
            i_p = nl.arange(nl.tile_size.pmax)
            
            # Load input data
            input_tile = nl.load(a_tensor[start_idx + i_p], mask=(start_idx + i_p < size))
            
            # Initialize indices (0, 1, 2, ...)
            idx_tile = nl.add(start_idx, i_p)
            
            # Store initial values and indices
            nl.store(result[start_idx + i_p], value=input_tile, mask=(start_idx + i_p < size))
            nl.store(indices[start_idx + i_p], value=idx_tile, mask=(start_idx + i_p < size))
        
        # Bubble sort
        for i in nl.affine_range(size):
            for j in nl.affine_range(size - 1):
                # Load current and next element
                curr_idx = j
                next_idx = j + 1
                
                if curr_idx < size and next_idx < size:
                    curr_val = nl.load(result[curr_idx])
                    next_val = nl.load(result[next_idx])
                    curr_idx_val = nl.load(indices[curr_idx])
                    next_idx_val = nl.load(indices[next_idx])
                    
                    # Compare and swap if needed
                    swap_needed = nl.greater(curr_val, next_val)
                    
                    # Update values
                    result_curr = nl.where(swap_needed, next_val, curr_val)
                    result_next = nl.where(swap_needed, curr_val, next_val)
                    indices_curr = nl.where(swap_needed, next_idx_val, curr_idx_val)
                    indices_next = nl.where(swap_needed, curr_idx_val, next_idx_val)
                    
                    # Store back
                    nl.store(result[curr_idx], value=result_curr)
                    nl.store(result[next_idx], value=result_next)
                    nl.store(indices[curr_idx], value=indices_curr)
                    nl.store(indices[next_idx], value=indices_next)
    
    # Handle multi-dimensional tensors
    else:
        # Determine sizes for processing
        sort_dim_size = shape[dim]
        
        # For each "row" along the sort dimension, we need to sort independently
        if dim == ndim - 1:  # Last dimension
            # Calculate the total number of rows
            total_rows = 1
            for d in range(ndim - 1):
                total_rows *= shape[d]
            
            # Process each row
            for row in nl.affine_range(total_rows):
                # Calculate multi-dimensional indices
                indices_list = []
                remaining = row
                for d in range(ndim - 1):
                    dim_size = shape[d]
                    idx = remaining // (total_rows // (dim_size * (1 if d == 0 else indices_list[-1][1])))
                    remaining = remaining % (total_rows // (dim_size * (1 if d == 0 else indices_list[-1][1])))
                    if d == 0:
                        indices_list.append((idx, dim_size))
                    else:
                        indices_list.append((idx, dim_size * indices_list[-1][1]))
                
                # Initialize indices for this row
                for i in nl.affine_range(sort_dim_size):
                    idx_val = nl.full((), i, dtype=nl.int32)
                    
                    # Build the index tuple for this element
                    idx_tuple = []
                    for d in range(ndim - 1):
                        idx_tuple.append(indices_list[d][0])
                    idx_tuple.append(i)
                    
                    # Store the value and index
                    val = nl.load(a_tensor[tuple(idx_tuple)])
                    nl.store(result[tuple(idx_tuple)], value=val)
                    nl.store(indices[tuple(idx_tuple)], value=idx_val)
                
                # Bubble sort this row
                for i in nl.affine_range(sort_dim_size):
                    for j in nl.affine_range(sort_dim_size - 1):
                        # Build indices for current and next element
                        curr_idx_tuple = []
                        next_idx_tuple = []
                        for d in range(ndim - 1):
                            curr_idx_tuple.append(indices_list[d][0])
                            next_idx_tuple.append(indices_list[d][0])
                        curr_idx_tuple.append(j)
                        next_idx_tuple.append(j + 1)
                        
                        # Load values and indices
                        curr_val = nl.load(result[tuple(curr_idx_tuple)])
                        next_val = nl.load(result[tuple(next_idx_tuple)])
                        curr_idx_val = nl.load(indices[tuple(curr_idx_tuple)])
                        next_idx_val = nl.load(indices[tuple(next_idx_tuple)])
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(curr_val, next_val)
                        
                        # Update values
                        result_curr = nl.where(swap_needed, next_val, curr_val)
                        result_next = nl.where(swap_needed, curr_val, next_val)
                        indices_curr = nl.where(swap_needed, next_idx_val, curr_idx_val)
                        indices_next = nl.where(swap_needed, curr_idx_val, next_idx_val)
                        
                        # Store back
                        nl.store(result[tuple(curr_idx_tuple)], value=result_curr)
                        nl.store(result[tuple(next_idx_tuple)], value=result_next)
                        nl.store(indices[tuple(curr_idx_tuple)], value=indices_curr)
                        nl.store(indices[tuple(next_idx_tuple)], value=indices_next)
        else:
            # For other dimensions, similar approach but different indexing
            # This is a simplified 2D case where dim=0
            if ndim == 2 and dim == 0:
                rows = shape[1]
                cols = shape[0]
                
                # Initialize with original values and indices
                for col in nl.affine_range(cols):
                    for row in nl.affine_range(rows):
                        val = nl.load(a_tensor[col, row])
                        nl.store(result[col, row], value=val)
                        nl.store(indices[col, row], value=nl.full((), col, dtype=nl.int32))
                
                # Sort each column independently
                for row in nl.affine_range(rows):
                    for i in nl.affine_range(cols):
                        for j in nl.affine_range(cols - 1):
                            # Load values and indices
                            curr_val = nl.load(result[j, row])
                            next_val = nl.load(result[j + 1, row])
                            curr_idx_val = nl.load(indices[j, row])
                            next_idx_val = nl.load(indices[j + 1, row])
                            
                            # Compare and swap if needed
                            swap_needed = nl.greater(curr_val, next_val)
                            
                            # Update values
                            result_curr = nl.where(swap_needed, next_val, curr_val)
                            result_next = nl.where(swap_needed, curr_val, next_val)
                            indices_curr = nl.where(swap_needed, next_idx_val, curr_idx_val)
                            indices_next = nl.where(swap_needed, curr_idx_val, next_idx_val)
                            
                            # Store back
                            nl.store(result[j, row], value=result_curr)
                            nl.store(result[j + 1, row], value=result_next)
                            nl.store(indices[j, row], value=indices_curr)
                            nl.store(indices[j + 1, row], value=indices_next)
    
    # Return both the sorted values and indices
    return result, indices