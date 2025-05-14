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
    
    # Initialize result array for values with same shape and dtype as input
    result_values = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Initialize result array for indices with same shape but int32 dtype
    result_indices = nl.ndarray(shape, dtype=nl.int32, buffer=nl.shared_hbm)
    
    # Handle 1D tensor case
    if ndim == 1:
        size = shape[0]
        max_tile_size = min(nl.tile_size.pmax, size)
        
        # Process in tiles to respect hardware limitations
        for p_start in nl.affine_range(math.ceil(size / max_tile_size)):
            # Calculate the start and end of the current tile
            start_idx = p_start * max_tile_size
            
            # Create indices for the current tile
            i_p = nl.arange(max_tile_size)
            
            # Load the current tile from the input tensor
            a_tile = nl.load(a_tensor[start_idx + i_p], mask=(start_idx + i_p < size))
            
            # Initialize indices for this tile
            idx_tile = nl.zeros((max_tile_size,), dtype=nl.int32, buffer=nl.sbuf)
            for i in nl.affine_range(max_tile_size):
                idx_tile_val = start_idx + i
                idx_tile_cond = nl.less(i, size - start_idx)
                idx_tile = nl.where(idx_tile_cond, idx_tile_val, idx_tile)
            
            # Bubble sort algorithm on the tile
            for i in nl.affine_range(max_tile_size):
                for j in nl.affine_range(max_tile_size - 1):
                    j_cond = nl.less(j + 1, max_tile_size)
                    # Compare adjacent elements
                    is_greater = nl.greater(a_tile[j], a_tile[j+1])
                    # Swap values if needed
                    if_true = a_tile[j+1]
                    if_false = a_tile[j]
                    a_tile_j = nl.where(is_greater, if_true, if_false)
                    
                    if_true = a_tile[j]
                    if_false = a_tile[j+1]
                    a_tile_j1 = nl.where(is_greater, if_true, if_false)
                    
                    a_tile = nl.where(nl.equal(nl.arange(max_tile_size), j), a_tile_j, a_tile)
                    a_tile = nl.where(nl.equal(nl.arange(max_tile_size), j+1), a_tile_j1, a_tile)
                    
                    # Swap indices if needed
                    if_true = idx_tile[j+1]
                    if_false = idx_tile[j]
                    idx_tile_j = nl.where(is_greater, if_true, if_false)
                    
                    if_true = idx_tile[j]
                    if_false = idx_tile[j+1]
                    idx_tile_j1 = nl.where(is_greater, if_true, if_false)
                    
                    idx_tile = nl.where(nl.equal(nl.arange(max_tile_size), j), idx_tile_j, idx_tile)
                    idx_tile = nl.where(nl.equal(nl.arange(max_tile_size), j+1), idx_tile_j1, idx_tile)
            
            # Store the sorted results
            nl.store(result_values[start_idx + i_p], value=a_tile, mask=(start_idx + i_p < size))
            nl.store(result_indices[start_idx + i_p], value=idx_tile, mask=(start_idx + i_p < size))
    
    # Handle 2D tensor case
    elif ndim == 2:
        if dim == 0:
            # Sort along first dimension (columns)
            rows, cols = shape
            max_tile_rows = min(nl.tile_size.pmax, rows)
            
            for c in nl.affine_range(cols):
                for r_start in nl.affine_range(math.ceil(rows / max_tile_rows)):
                    # Calculate the start row of the current tile
                    start_row = r_start * max_tile_rows
                    
                    # Create indices for the current tile
                    i_r = nl.arange(max_tile_rows)
                    
                    # Load the current column tile
                    col_tile = nl.load(a_tensor[start_row + i_r, c], mask=(start_row + i_r < rows))
                    
                    # Initialize indices for this tile
                    idx_tile = nl.zeros((max_tile_rows,), dtype=nl.int32, buffer=nl.sbuf)
                    for i in nl.affine_range(max_tile_rows):
                        idx_tile_val = start_row + i
                        idx_tile_cond = nl.less(i, rows - start_row)
                        idx_tile = nl.where(idx_tile_cond, idx_tile_val, idx_tile)
                    
                    # Bubble sort algorithm on the tile
                    for i in nl.affine_range(max_tile_rows):
                        for j in nl.affine_range(max_tile_rows - 1):
                            # Compare adjacent elements
                            is_greater = nl.greater(col_tile[j], col_tile[j+1])
                            
                            # Swap values if needed
                            if_true = col_tile[j+1]
                            if_false = col_tile[j]
                            col_tile_j = nl.where(is_greater, if_true, if_false)
                            
                            if_true = col_tile[j]
                            if_false = col_tile[j+1]
                            col_tile_j1 = nl.where(is_greater, if_true, if_false)
                            
                            col_tile = nl.where(nl.equal(nl.arange(max_tile_rows), j), col_tile_j, col_tile)
                            col_tile = nl.where(nl.equal(nl.arange(max_tile_rows), j+1), col_tile_j1, col_tile)
                            
                            # Swap indices if needed
                            if_true = idx_tile[j+1]
                            if_false = idx_tile[j]
                            idx_tile_j = nl.where(is_greater, if_true, if_false)
                            
                            if_true = idx_tile[j]
                            if_false = idx_tile[j+1]
                            idx_tile_j1 = nl.where(is_greater, if_true, if_false)
                            
                            idx_tile = nl.where(nl.equal(nl.arange(max_tile_rows), j), idx_tile_j, idx_tile)
                            idx_tile = nl.where(nl.equal(nl.arange(max_tile_rows), j+1), idx_tile_j1, idx_tile)
                    
                    # Store the sorted results
                    nl.store(result_values[start_row + i_r, c], value=col_tile, mask=(start_row + i_r < rows))
                    nl.store(result_indices[start_row + i_r, c], value=idx_tile, mask=(start_row + i_r < rows))
        
        else:  # dim == 1
            # Sort along second dimension (rows)
            rows, cols = shape
            max_tile_cols = min(nl.tile_size.pmax, cols)
            
            for r in nl.affine_range(rows):
                for c_start in nl.affine_range(math.ceil(cols / max_tile_cols)):
                    # Calculate the start column of the current tile
                    start_col = c_start * max_tile_cols
                    
                    # Create indices for the current tile
                    i_c = nl.arange(max_tile_cols)
                    
                    # Load the current row tile
                    row_tile = nl.load(a_tensor[r, start_col + i_c], mask=(start_col + i_c < cols))
                    
                    # Initialize indices for this tile
                    idx_tile = nl.zeros((max_tile_cols,), dtype=nl.int32, buffer=nl.sbuf)
                    for i in nl.affine_range(max_tile_cols):
                        idx_tile_val = start_col + i
                        idx_tile_cond = nl.less(i, cols - start_col)
                        idx_tile = nl.where(idx_tile_cond, idx_tile_val, idx_tile)
                    
                    # Bubble sort algorithm on the tile
                    for i in nl.affine_range(max_tile_cols):
                        for j in nl.affine_range(max_tile_cols - 1):
                            # Compare adjacent elements
                            is_greater = nl.greater(row_tile[j], row_tile[j+1])
                            
                            # Swap values if needed
                            if_true = row_tile[j+1]
                            if_false = row_tile[j]
                            row_tile_j = nl.where(is_greater, if_true, if_false)
                            
                            if_true = row_tile[j]
                            if_false = row_tile[j+1]
                            row_tile_j1 = nl.where(is_greater, if_true, if_false)
                            
                            row_tile = nl.where(nl.equal(nl.arange(max_tile_cols), j), row_tile_j, row_tile)
                            row_tile = nl.where(nl.equal(nl.arange(max_tile_cols), j+1), row_tile_j1, row_tile)
                            
                            # Swap indices if needed
                            if_true = idx_tile[j+1]
                            if_false = idx_tile[j]
                            idx_tile_j = nl.where(is_greater, if_true, if_false)
                            
                            if_true = idx_tile[j]
                            if_false = idx_tile[j+1]
                            idx_tile_j1 = nl.where(is_greater, if_true, if_false)
                            
                            idx_tile = nl.where(nl.equal(nl.arange(max_tile_cols), j), idx_tile_j, idx_tile)
                            idx_tile = nl.where(nl.equal(nl.arange(max_tile_cols), j+1), idx_tile_j1, idx_tile)
                    
                    # Store the sorted results
                    nl.store(result_values[r, start_col + i_c], value=row_tile, mask=(start_col + i_c < cols))
                    nl.store(result_indices[r, start_col + i_c], value=idx_tile, mask=(start_col + i_c < cols))
    
    # Return both sorted values and indices
    return result_values, result_indices