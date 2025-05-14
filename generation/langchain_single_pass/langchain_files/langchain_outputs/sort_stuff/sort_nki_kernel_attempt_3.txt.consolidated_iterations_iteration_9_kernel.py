from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Handle negative dim index
    if dim < 0:
        dim = len(a_tensor.shape) + dim
        
    # Initialize result with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Copy input to result first
    if len(a_tensor.shape) == 1:
        # 1D tensor case
        size = a_tensor.shape[0]
        max_tile_size = nl.tile_size.pmax
        
        # Process in tiles to respect hardware limitations
        trip_count = math.ceil(size / max_tile_size)
        
        for p in nl.affine_range(trip_count):
            # Calculate indices for this tile
            start_idx = p * max_tile_size
            
            # Load data from input tensor
            in_indices = nl.arange(max_tile_size)
            mask = (in_indices + start_idx) < size
            in_data = nl.load(a_tensor[start_idx + in_indices], mask=mask)
            
            # Store data to result tensor
            nl.store(result[start_idx + in_indices], value=in_data, mask=mask)
            
        # Perform bubble sort on the entire array
        # We need fixed iteration counts for hardware compatibility
        for i in nl.affine_range(size):
            for j in nl.affine_range(size - 1):
                # Process in tiles for each comparison
                for p in nl.affine_range(trip_count):
                    start_idx = p * max_tile_size
                    
                    # Load current elements
                    idx = nl.arange(max_tile_size)
                    curr_mask = (idx + start_idx) < (size - 1)
                    curr_idx = start_idx + idx
                    curr_vals = nl.load(result[curr_idx], mask=curr_mask)
                    
                    # Load next elements
                    next_idx = curr_idx + 1
                    next_mask = (next_idx) < size
                    next_vals = nl.load(result[next_idx], mask=next_mask)
                    
                    # Compare and swap if needed
                    swap_mask = nl.greater(curr_vals, next_vals)
                    final_mask = curr_mask & next_mask & swap_mask
                    
                    # Store swapped values
                    if nl.any(final_mask):
                        temp = nl.where(final_mask, next_vals, curr_vals)
                        nl.store(result[curr_idx], value=temp, mask=final_mask)
                        
                        temp = nl.where(final_mask, curr_vals, next_vals)
                        nl.store(result[next_idx], value=temp, mask=final_mask)
                        
    else:
        # Multi-dimensional tensor case
        # For simplicity, we'll handle 2D case explicitly
        if len(a_tensor.shape) == 2:
            rows, cols = a_tensor.shape
            
            if dim == 0:
                # Sort along rows
                max_tile_size = nl.tile_size.pmax
                trip_count_rows = math.ceil(rows / max_tile_size)
                
                # Copy input to result
                for p in nl.affine_range(trip_count_rows):
                    start_row = p * max_tile_size
                    row_indices = nl.arange(max_tile_size)[:, None]
                    col_indices = nl.arange(cols)[None, :]
                    mask = (row_indices + start_row) < rows
                    
                    in_data = nl.load(a_tensor[start_row + row_indices, col_indices], mask=mask)
                    nl.store(result[start_row + row_indices, col_indices], value=in_data, mask=mask)
                
                # Sort each column independently
                for c in nl.affine_range(cols):
                    # Bubble sort algorithm
                    for i in nl.affine_range(rows):
                        for j in nl.affine_range(rows - 1):
                            # Process in tiles
                            for p in nl.affine_range(trip_count_rows):
                                start_row = p * max_tile_size
                                row_indices = nl.arange(max_tile_size)
                                curr_mask = (row_indices + start_row) < (rows - 1)
                                
                                # Load current elements
                                curr_row_idx = start_row + row_indices
                                curr_vals = nl.load(result[curr_row_idx, c], mask=curr_mask)
                                
                                # Load next elements
                                next_row_idx = curr_row_idx + 1
                                next_mask = next_row_idx < rows
                                next_vals = nl.load(result[next_row_idx, c], mask=next_mask)
                                
                                # Compare and swap if needed
                                swap_mask = nl.greater(curr_vals, next_vals)
                                final_mask = curr_mask & next_mask & swap_mask
                                
                                # Store swapped values
                                if nl.any(final_mask):
                                    temp = nl.where(final_mask, next_vals, curr_vals)
                                    nl.store(result[curr_row_idx, c], value=temp, mask=final_mask)
                                    
                                    temp = nl.where(final_mask, curr_vals, next_vals)
                                    nl.store(result[next_row_idx, c], value=temp, mask=final_mask)
                                    
            else:  # dim == 1
                # Sort along columns
                max_tile_size = nl.tile_size.pmax
                trip_count_cols = math.ceil(cols / max_tile_size)
                
                # Copy input to result
                for r in nl.affine_range(rows):
                    for p in nl.affine_range(trip_count_cols):
                        start_col = p * max_tile_size
                        col_indices = nl.arange(max_tile_size)
                        mask = (col_indices + start_col) < cols
                        
                        in_data = nl.load(a_tensor[r, start_col + col_indices], mask=mask)
                        nl.store(result[r, start_col + col_indices], value=in_data, mask=mask)
                
                # Sort each row independently
                for r in nl.affine_range(rows):
                    # Bubble sort algorithm
                    for i in nl.affine_range(cols):
                        for j in nl.affine_range(cols - 1):
                            # Process in tiles
                            for p in nl.affine_range(trip_count_cols):
                                start_col = p * max_tile_size
                                col_indices = nl.arange(max_tile_size)
                                curr_mask = (col_indices + start_col) < (cols - 1)
                                
                                # Load current elements
                                curr_col_idx = start_col + col_indices
                                curr_vals = nl.load(result[r, curr_col_idx], mask=curr_mask)
                                
                                # Load next elements
                                next_col_idx = curr_col_idx + 1
                                next_mask = next_col_idx < cols
                                next_vals = nl.load(result[r, next_col_idx], mask=next_mask)
                                
                                # Compare and swap if needed
                                swap_mask = nl.greater(curr_vals, next_vals)
                                final_mask = curr_mask & next_mask & swap_mask
                                
                                # Store swapped values
                                if nl.any(final_mask):
                                    temp = nl.where(final_mask, next_vals, curr_vals)
                                    nl.store(result[r, curr_col_idx], value=temp, mask=final_mask)
                                    
                                    temp = nl.where(final_mask, curr_vals, next_vals)
                                    nl.store(result[r, next_col_idx], value=temp, mask=final_mask)
        else:
            # For higher dimensions, we just copy the input to the output
            # This is a fallback for unsupported dimensions
            # In a real implementation, this would need to be extended
            for p in nl.affine_range(math.ceil(a_tensor.size / nl.tile_size.pmax)):
                start_idx = p * nl.tile_size.pmax
                indices = nl.arange(nl.tile_size.pmax)
                mask = (indices + start_idx) < a_tensor.size
                
                in_data = nl.load(a_tensor.reshape(-1)[start_idx + indices], mask=mask)
                nl.store(result.reshape(-1)[start_idx + indices], value=in_data, mask=mask)
    
    return result