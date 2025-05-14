from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Handle negative dim index
    if dim < 0:
        dim = len(a_tensor.shape) + dim
        
    # Get shape information
    shape = a_tensor.shape
    dim_size = shape[dim]
    
    # Initialize result arrays
    values = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    indices = nl.ndarray(shape, dtype=nl.int32, buffer=nl.shared_hbm)
    
    # Special case for 1D tensor
    if len(shape) == 1:
        # Copy input to values
        max_tile_size = nl.tile_size.pmax
        num_tiles = math.ceil(shape[0] / max_tile_size)
        
        for tile_idx in nl.affine_range(num_tiles):
            start_idx = tile_idx * max_tile_size
            # Use proper masking to handle boundary conditions
            mask = (nl.arange(max_tile_size) + start_idx < shape[0])
            
            # Load input tile
            input_tile = nl.load(a_tensor[start_idx:start_idx + max_tile_size], mask=mask)
            
            # Initialize indices for this tile
            idx_tile = nl.zeros((max_tile_size,), dtype=nl.int32, buffer=nl.sbuf)
            for i in nl.affine_range(max_tile_size):
                idx_tile[i] = start_idx + i
            
            # Bubble sort implementation
            for i in nl.affine_range(max_tile_size):
                for j in nl.affine_range(max_tile_size - 1):
                    # Check if current element > next element
                    condition = nl.greater(input_tile[j], input_tile[j+1])
                    
                    # Swap values if needed
                    temp_val = nl.where(condition, input_tile[j+1], input_tile[j])
                    input_tile[j+1] = nl.where(condition, input_tile[j], input_tile[j+1])
                    input_tile[j] = temp_val
                    
                    # Also swap indices
                    temp_idx = nl.where(condition, idx_tile[j+1], idx_tile[j])
                    idx_tile[j+1] = nl.where(condition, idx_tile[j], idx_tile[j+1])
                    idx_tile[j] = temp_idx
            
            # Store results
            nl.store(values[start_idx:start_idx + max_tile_size], value=input_tile, mask=mask)
            nl.store(indices[start_idx:start_idx + max_tile_size], value=idx_tile, mask=mask)
    
    # For 2D tensor
    elif len(shape) == 2:
        rows, cols = shape
        
        # Determine which dimension to sort along
        if dim == 0:  # Sort along rows
            max_tile_rows = min(nl.tile_size.pmax, rows)
            
            for col_idx in nl.affine_range(cols):
                # Create temporary buffers for this column
                col_values = nl.zeros((rows,), dtype=a_tensor.dtype, buffer=nl.sbuf)
                col_indices = nl.zeros((rows,), dtype=nl.int32, buffer=nl.sbuf)
                
                # Load column data in tiles
                for row_tile in nl.affine_range(math.ceil(rows / max_tile_rows)):
                    start_row = row_tile * max_tile_rows
                    mask = (nl.arange(max_tile_rows) + start_row < rows)
                    
                    # Load tile
                    row_indices = nl.arange(max_tile_rows) + start_row
                    tile_data = nl.load(a_tensor[row_indices, col_idx], mask=mask)
                    
                    # Store into column buffer
                    for i in nl.affine_range(max_tile_rows):
                        if start_row + i < rows:
                            col_values[start_row + i] = tile_data[i]
                            col_indices[start_row + i] = start_row + i
                
                # Sort the column
                for i in nl.affine_range(rows):
                    for j in nl.affine_range(rows - 1):
                        condition = nl.greater(col_values[j], col_values[j+1])
                        
                        # Swap values if needed
                        temp_val = nl.where(condition, col_values[j+1], col_values[j])
                        col_values[j+1] = nl.where(condition, col_values[j], col_values[j+1])
                        col_values[j] = temp_val
                        
                        # Also swap indices
                        temp_idx = nl.where(condition, col_indices[j+1], col_indices[j])
                        col_indices[j+1] = nl.where(condition, col_indices[j], col_indices[j+1])
                        col_indices[j] = temp_idx
                
                # Store sorted column back in tiles
                for row_tile in nl.affine_range(math.ceil(rows / max_tile_rows)):
                    start_row = row_tile * max_tile_rows
                    mask = (nl.arange(max_tile_rows) + start_row < rows)
                    
                    # Extract tile data
                    val_tile = nl.zeros((max_tile_rows,), dtype=a_tensor.dtype, buffer=nl.sbuf)
                    idx_tile = nl.zeros((max_tile_rows,), dtype=nl.int32, buffer=nl.sbuf)
                    
                    for i in nl.affine_range(max_tile_rows):
                        if start_row + i < rows:
                            val_tile[i] = col_values[start_row + i]
                            idx_tile[i] = col_indices[start_row + i]
                    
                    # Store results
                    row_indices = nl.arange(max_tile_rows) + start_row
                    nl.store(values[row_indices, col_idx], value=val_tile, mask=mask)
                    nl.store(indices[row_indices, col_idx], value=idx_tile, mask=mask)
        
        else:  # Sort along columns (dim == 1)
            max_tile_cols = min(nl.tile_size.pmax, cols)
            
            for row_idx in nl.affine_range(rows):
                # Load row data in tiles
                for col_tile in nl.affine_range(math.ceil(cols / max_tile_cols)):
                    start_col = col_tile * max_tile_cols
                    mask = (nl.arange(max_tile_cols) + start_col < cols)
                    
                    # Load tile
                    col_indices = nl.arange(max_tile_cols) + start_col
                    tile_data = nl.load(a_tensor[row_idx, col_indices], mask=mask)
                    
                    # Initialize indices
                    idx_tile = nl.zeros((max_tile_cols,), dtype=nl.int32, buffer=nl.sbuf)
                    for i in nl.affine_range(max_tile_cols):
                        idx_tile[i] = start_col + i
                    
                    # Sort this tile using bubble sort
                    for i in nl.affine_range(max_tile_cols):
                        for j in nl.affine_range(max_tile_cols - 1):
                            condition = nl.greater(tile_data[j], tile_data[j+1])
                            
                            # Swap values if needed
                            temp_val = nl.where(condition, tile_data[j+1], tile_data[j])
                            tile_data[j+1] = nl.where(condition, tile_data[j], tile_data[j+1])
                            tile_data[j] = temp_val
                            
                            # Also swap indices
                            temp_idx = nl.where(condition, idx_tile[j+1], idx_tile[j])
                            idx_tile[j+1] = nl.where(condition, idx_tile[j], idx_tile[j+1])
                            idx_tile[j] = temp_idx
                    
                    # Store results
                    nl.store(values[row_idx, col_indices], value=tile_data, mask=mask)
                    nl.store(indices[row_idx, col_indices], value=idx_tile, mask=mask)
    
    # Return both sorted values and indices
    return values, indices