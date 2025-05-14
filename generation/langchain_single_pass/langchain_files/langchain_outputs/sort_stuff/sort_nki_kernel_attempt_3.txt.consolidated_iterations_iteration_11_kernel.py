from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Handle negative dim index
    if dim < 0:
        dim = len(a_tensor.shape) + dim
    
    # Get input tensor shape
    tensor_shape = a_tensor.shape
    
    # Initialize result arrays for sorted values and indices
    values = nl.ndarray(tensor_shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    indices = nl.ndarray(tensor_shape, dtype=nl.int32, buffer=nl.shared_hbm)
    
    # Handle 1D tensor case
    if len(tensor_shape) == 1:
        size = tensor_shape[0]
        max_tile_size = nl.tile_size.pmax
        
        # Process in tiles to respect hardware limitations
        trip_count = math.ceil(size / max_tile_size)
        
        for t in nl.affine_range(trip_count):
            # Calculate actual tile size for this iteration
            start_idx = t * max_tile_size
            actual_tile_size = min(max_tile_size, size - start_idx)
            
            # Load tile data
            tile_indices = nl.zeros((max_tile_size,), dtype=nl.int32, buffer=nl.sbuf)
            tile_values = nl.zeros((max_tile_size,), dtype=a_tensor.dtype, buffer=nl.sbuf)
            
            # Initialize indices and load values
            for i in nl.affine_range(max_tile_size):
                is_valid = i < actual_tile_size
                if is_valid:
                    tile_indices[i] = start_idx + i
                    tile_values[i] = nl.load(a_tensor[start_idx + i])
            
            # Bubble sort implementation for this tile
            for i in nl.affine_range(actual_tile_size):
                for j in nl.affine_range(actual_tile_size - 1):
                    # Check if we need to swap
                    needs_swap = nl.greater(tile_values[j], tile_values[j+1])
                    
                    # Swap values if needed
                    if needs_swap:
                        temp_val = tile_values[j]
                        tile_values[j] = tile_values[j+1]
                        tile_values[j+1] = temp_val
                        
                        temp_idx = tile_indices[j]
                        tile_indices[j] = tile_indices[j+1]
                        tile_indices[j+1] = temp_idx
            
            # Store sorted values and indices back to result tensors
            for i in nl.affine_range(max_tile_size):
                is_valid = i < actual_tile_size
                if is_valid:
                    nl.store(values[start_idx + i], tile_values[i])
                    nl.store(indices[start_idx + i], tile_indices[i])
    
    # Handle 2D tensor case
    elif len(tensor_shape) == 2:
        rows, cols = tensor_shape
        
        if dim == 0:  # Sort along rows
            max_tile_size = nl.tile_size.pmax
            trip_count = math.ceil(cols / max_tile_size)
            
            for t in nl.affine_range(trip_count):
                start_col = t * max_tile_size
                actual_cols = min(max_tile_size, cols - start_col)
                
                for col in nl.affine_range(actual_cols):
                    # Load column data
                    column_values = nl.zeros((rows,), dtype=a_tensor.dtype, buffer=nl.sbuf)
                    column_indices = nl.zeros((rows,), dtype=nl.int32, buffer=nl.sbuf)
                    
                    # Initialize indices and load values
                    for i in nl.affine_range(rows):
                        column_indices[i] = i
                        column_values[i] = nl.load(a_tensor[i, start_col + col])
                    
                    # Bubble sort implementation for this column
                    for i in nl.affine_range(rows):
                        for j in nl.affine_range(rows - 1):
                            needs_swap = nl.greater(column_values[j], column_values[j+1])
                            
                            if needs_swap:
                                temp_val = column_values[j]
                                column_values[j] = column_values[j+1]
                                column_values[j+1] = temp_val
                                
                                temp_idx = column_indices[j]
                                column_indices[j] = column_indices[j+1]
                                column_indices[j+1] = temp_idx
                    
                    # Store sorted column back
                    for i in nl.affine_range(rows):
                        nl.store(values[i, start_col + col], column_values[i])
                        nl.store(indices[i, start_col + col], column_indices[i])
        
        else:  # dim == 1, sort along columns
            max_tile_size = nl.tile_size.pmax
            trip_count = math.ceil(rows / max_tile_size)
            
            for t in nl.affine_range(trip_count):
                start_row = t * max_tile_size
                actual_rows = min(max_tile_size, rows - start_row)
                
                for row in nl.affine_range(actual_rows):
                    # Load row data
                    row_values = nl.zeros((cols,), dtype=a_tensor.dtype, buffer=nl.sbuf)
                    row_indices = nl.zeros((cols,), dtype=nl.int32, buffer=nl.sbuf)
                    
                    # Initialize indices and load values
                    for i in nl.affine_range(cols):
                        row_indices[i] = i
                        row_values[i] = nl.load(a_tensor[start_row + row, i])
                    
                    # Bubble sort implementation for this row
                    for i in nl.affine_range(cols):
                        for j in nl.affine_range(cols - 1):
                            needs_swap = nl.greater(row_values[j], row_values[j+1])
                            
                            if needs_swap:
                                temp_val = row_values[j]
                                row_values[j] = row_values[j+1]
                                row_values[j+1] = temp_val
                                
                                temp_idx = row_indices[j]
                                row_indices[j] = row_indices[j+1]
                                row_indices[j+1] = temp_idx
                    
                    # Store sorted row back
                    for i in nl.affine_range(cols):
                        nl.store(values[start_row + row, i], row_values[i])
                        nl.store(indices[start_row + row, i], row_indices[i])
    
    # Return both the sorted values and indices
    return values, indices