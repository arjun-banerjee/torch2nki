from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Handle negative dimensions
    if dim < 0:
        dim = len(a_tensor.shape) + dim
        
    # Get shape of input tensor
    shape = a_tensor.shape
    sort_dim_size = shape[dim]
    
    # Initialize result arrays with same shape and dtype as input
    values = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    indices = nl.ndarray(a_tensor.shape, dtype=nl.int32, buffer=nl.shared_hbm)
    
    # Handle 1D tensor case
    if len(shape) == 1:
        # Copy input to values
        input_sbuf = nl.load(a_tensor)
        
        # Initialize indices
        idx_sbuf = nl.arange(sort_dim_size)
        
        # Bubble sort implementation
        for i in nl.affine_range(sort_dim_size-1):
            for j in nl.affine_range(sort_dim_size-i-1):
                # Compare adjacent elements
                j_idx = nl.full((1,), j, dtype=nl.int32)
                j_plus_1 = nl.full((1,), j+1, dtype=nl.int32)
                
                val_j = input_sbuf[j_idx]
                val_j_plus_1 = input_sbuf[j_plus_1]
                
                # If current value is greater than next value, swap them
                swap_condition = nl.greater(val_j, val_j_plus_1)
                
                # Swap values
                temp_val = nl.where(swap_condition, val_j, val_j)
                input_sbuf = nl.where(swap_condition, val_j_plus_1, input_sbuf)
                input_sbuf = nl.where(swap_condition, temp_val, input_sbuf)
                
                # Swap indices
                idx_j = idx_sbuf[j_idx]
                idx_j_plus_1 = idx_sbuf[j_plus_1]
                
                temp_idx = nl.where(swap_condition, idx_j, idx_j)
                idx_sbuf = nl.where(swap_condition, idx_j_plus_1, idx_sbuf)
                idx_sbuf = nl.where(swap_condition, temp_idx, idx_sbuf)
        
        # Store results
        nl.store(values, input_sbuf)
        nl.store(indices, idx_sbuf)
        
    # Handle 2D tensor case
    elif len(shape) == 2:
        rows, cols = shape
        
        if dim == 0:  # Sort along rows
            # Process in tiles to handle large tensors
            tile_size = min(128, cols)
            
            for c in nl.affine_range(math.ceil(cols / tile_size)):
                c_start = c * tile_size
                c_actual = min(tile_size, cols - c_start)
                
                # Create indices for loading
                i_c = c_start + nl.arange(tile_size)[None, :]
                i_r = nl.arange(rows)[:, None]
                
                # Load data for this tile
                input_sbuf = nl.load(a_tensor[i_r, i_c], mask=(i_c < cols))
                
                # Initialize indices for this tile
                idx_sbuf = nl.zeros((rows, tile_size), dtype=nl.int32)
                for r in nl.affine_range(rows):
                    r_idx = nl.full((1, 1), r, dtype=nl.int32)
                    idx_row = nl.arange(tile_size)[None, :]
                    idx_sbuf = nl.where(nl.equal(i_r, r_idx), idx_row, idx_sbuf)
                
                # Bubble sort implementation for each column
                for i in nl.affine_range(rows-1):
                    for j in nl.affine_range(rows-i-1):
                        j_idx = nl.full((1, 1), j, dtype=nl.int32)
                        j_plus_1 = nl.full((1, 1), j+1, dtype=nl.int32)
                        
                        # Compare adjacent elements in each column
                        val_j = input_sbuf[j_idx]
                        val_j_plus_1 = input_sbuf[j_plus_1]
                        
                        # Condition for swapping
                        swap_condition = nl.greater(val_j, val_j_plus_1)
                        
                        # Swap values
                        temp_val = nl.where(swap_condition, val_j, val_j)
                        input_sbuf = nl.where(swap_condition, val_j_plus_1, input_sbuf)
                        input_sbuf = nl.where(swap_condition, temp_val, input_sbuf)
                        
                        # Swap indices
                        idx_j = idx_sbuf[j_idx]
                        idx_j_plus_1 = idx_sbuf[j_plus_1]
                        
                        temp_idx = nl.where(swap_condition, idx_j, idx_j)
                        idx_sbuf = nl.where(swap_condition, idx_j_plus_1, idx_sbuf)
                        idx_sbuf = nl.where(swap_condition, temp_idx, idx_sbuf)
                
                # Store results for this tile
                nl.store(values[i_r, i_c], input_sbuf, mask=(i_c < cols))
                nl.store(indices[i_r, i_c], idx_sbuf, mask=(i_c < cols))
                
        else:  # Sort along columns (dim=1)
            # Process in tiles to handle large tensors
            tile_size = min(128, rows)
            
            for r in nl.affine_range(math.ceil(rows / tile_size)):
                r_start = r * tile_size
                r_actual = min(tile_size, rows - r_start)
                
                # Create indices for loading
                i_r = r_start + nl.arange(tile_size)[:, None]
                i_c = nl.arange(cols)[None, :]
                
                # Load data for this tile
                input_sbuf = nl.load(a_tensor[i_r, i_c], mask=(i_r < rows))
                
                # Initialize indices for this tile
                idx_sbuf = nl.zeros((tile_size, cols), dtype=nl.int32)
                for c in nl.affine_range(cols):
                    c_idx = nl.full((1, 1), c, dtype=nl.int32)
                    idx_col = nl.arange(tile_size)[:, None]
                    idx_sbuf = nl.where(nl.equal(i_c, c_idx), idx_col, idx_sbuf)
                
                # Bubble sort implementation for each row
                for i in nl.affine_range(cols-1):
                    for j in nl.affine_range(cols-i-1):
                        j_idx = nl.full((1, 1), j, dtype=nl.int32)
                        j_plus_1 = nl.full((1, 1), j+1, dtype=nl.int32)
                        
                        # Compare adjacent elements in each row
                        val_j = input_sbuf[:, j_idx]
                        val_j_plus_1 = input_sbuf[:, j_plus_1]
                        
                        # Condition for swapping
                        swap_condition = nl.greater(val_j, val_j_plus_1)
                        
                        # Swap values
                        temp_val = nl.where(swap_condition, val_j, val_j)
                        input_sbuf[:, j_idx] = nl.where(swap_condition, val_j_plus_1, input_sbuf[:, j_idx])
                        input_sbuf[:, j_plus_1] = nl.where(swap_condition, temp_val, input_sbuf[:, j_plus_1])
                        
                        # Swap indices
                        idx_j = idx_sbuf[:, j_idx]
                        idx_j_plus_1 = idx_sbuf[:, j_plus_1]
                        
                        temp_idx = nl.where(swap_condition, idx_j, idx_j)
                        idx_sbuf[:, j_idx] = nl.where(swap_condition, idx_j_plus_1, idx_sbuf[:, j_idx])
                        idx_sbuf[:, j_plus_1] = nl.where(swap_condition, temp_idx, idx_sbuf[:, j_plus_1])
                
                # Store results for this tile
                nl.store(values[i_r, i_c], input_sbuf, mask=(i_r < rows))
                nl.store(indices[i_r, i_c], idx_sbuf, mask=(i_r < rows))
    
    # Return result as tuple
    return values, indices