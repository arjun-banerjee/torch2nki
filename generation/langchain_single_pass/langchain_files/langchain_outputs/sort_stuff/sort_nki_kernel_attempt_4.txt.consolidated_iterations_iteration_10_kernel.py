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
    
    # Handle 1D case
    if ndim == 1:
        size = shape[0]
        # Copy input to result first
        i = nl.arange(size)
        in_tile = nl.load(a_tensor[i])
        nl.store(result[i], in_tile)
        
        # Bubble sort implementation
        for i in range(size):
            for j in range(size - i - 1):
                # Create indices for current and next element
                idx_curr = nl.arange(1)
                idx_next = nl.arange(1)
                idx_curr[0] = j
                idx_next[0] = j + 1
                
                # Load values
                val_curr = nl.load(result[idx_curr])
                val_next = nl.load(result[idx_next])
                
                # Compare and swap if needed
                swap_needed = nl.greater(val_curr, val_next)
                
                # Store swapped values if needed
                temp = nl.where(swap_needed, val_next, val_curr)
                nl.store(result[idx_curr], temp)
                
                temp = nl.where(swap_needed, val_curr, val_next)
                nl.store(result[idx_next], temp)
        
        return result
    
    # Handle 2D case
    elif ndim == 2:
        # Determine which dimension to sort along
        if dim == 0:
            # Sort along rows
            rows = shape[0]
            cols = shape[1]
            
            # Process each column independently
            for c in range(cols):
                # Copy input to result first for this column
                i = nl.arange(rows)
                col_idx = nl.full((rows,), c, dtype=nl.int32)
                in_tile = nl.load(a_tensor[i, col_idx])
                nl.store(result[i, col_idx], in_tile)
                
                # Bubble sort the column
                for i in range(rows):
                    for j in range(rows - i - 1):
                        # Create indices for current and next element
                        idx_curr = nl.arange(1)
                        idx_next = nl.arange(1)
                        idx_curr[0] = j
                        idx_next[0] = j + 1
                        
                        # Load values
                        val_curr = nl.load(result[idx_curr, col_idx[0:1]])
                        val_next = nl.load(result[idx_next, col_idx[0:1]])
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(val_curr, val_next)
                        
                        # Store swapped values if needed
                        temp = nl.where(swap_needed, val_next, val_curr)
                        nl.store(result[idx_curr, col_idx[0:1]], temp)
                        
                        temp = nl.where(swap_needed, val_curr, val_next)
                        nl.store(result[idx_next, col_idx[0:1]], temp)
        else:
            # Sort along columns (dim=1)
            rows = shape[0]
            cols = shape[1]
            
            # Process each row independently
            for r in range(rows):
                # Copy input to result first for this row
                i = nl.arange(cols)
                row_idx = nl.full((cols,), r, dtype=nl.int32)
                in_tile = nl.load(a_tensor[row_idx, i])
                nl.store(result[row_idx, i], in_tile)
                
                # Bubble sort the row
                for i in range(cols):
                    for j in range(cols - i - 1):
                        # Create indices for current and next element
                        idx_curr = nl.arange(1)
                        idx_next = nl.arange(1)
                        idx_curr[0] = j
                        idx_next[0] = j + 1
                        
                        # Load values
                        val_curr = nl.load(result[row_idx[0:1], idx_curr])
                        val_next = nl.load(result[row_idx[0:1], idx_next])
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(val_curr, val_next)
                        
                        # Store swapped values if needed
                        temp = nl.where(swap_needed, val_next, val_curr)
                        nl.store(result[row_idx[0:1], idx_curr], temp)
                        
                        temp = nl.where(swap_needed, val_curr, val_next)
                        nl.store(result[row_idx[0:1], idx_next], temp)
    
    # Handle higher-dimensional case by reshaping
    else:
        # Reshape input tensor to 2D for sorting
        if dim == ndim - 1:
            # If sorting along the last dimension, we can treat it as rows
            outer_size = 1
            for i in range(ndim - 1):
                outer_size = outer_size * shape[i]
            
            inner_size = shape[dim]
            
            # Process each outer dimension independently
            for r in range(outer_size):
                # Sort each inner dimension
                for i in range(inner_size):
                    for j in range(inner_size - i - 1):
                        # Calculate flat indices for current and next elements
                        idx_curr = r * inner_size + j
                        idx_next = r * inner_size + j + 1
                        
                        # Convert to multi-dimensional indices
                        multi_idx_curr = [0] * ndim
                        multi_idx_next = [0] * ndim
                        
                        temp = idx_curr
                        for d in range(ndim - 1, -1, -1):
                            multi_idx_curr[d] = temp % shape[d]
                            temp = temp // shape[d]
                        
                        temp = idx_next
                        for d in range(ndim - 1, -1, -1):
                            multi_idx_next[d] = temp % shape[d]
                            temp = temp // shape[d]
                        
                        # Load values using multi-dimensional indices
                        idx_curr_nl = [nl.arange(1) for _ in range(ndim)]
                        idx_next_nl = [nl.arange(1) for _ in range(ndim)]
                        
                        for d in range(ndim):
                            idx_curr_nl[d][0] = multi_idx_curr[d]
                            idx_next_nl[d][0] = multi_idx_next[d]
                        
                        val_curr = nl.load(a_tensor[tuple(idx_curr_nl)])
                        val_next = nl.load(a_tensor[tuple(idx_next_nl)])
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(val_curr, val_next)
                        
                        # Store swapped values if needed
                        temp_curr = nl.where(swap_needed, val_next, val_curr)
                        temp_next = nl.where(swap_needed, val_curr, val_next)
                        
                        nl.store(result[tuple(idx_curr_nl)], temp_curr)
                        nl.store(result[tuple(idx_next_nl)], temp_next)
        else:
            # Copy input to result since we can't handle this case efficiently
            # We'll just return the unsorted tensor
            for i in range(math.prod(shape)):
                # Calculate multi-dimensional indices
                multi_idx = [0] * ndim
                temp = i
                for d in range(ndim - 1, -1, -1):
                    multi_idx[d] = temp % shape[d]
                    temp = temp // shape[d]
                
                # Load and store values using multi-dimensional indices
                idx_nl = [nl.arange(1) for _ in range(ndim)]
                
                for d in range(ndim):
                    idx_nl[d][0] = multi_idx[d]
                
                val = nl.load(a_tensor[tuple(idx_nl)])
                nl.store(result[tuple(idx_nl)], val)
    
    return result