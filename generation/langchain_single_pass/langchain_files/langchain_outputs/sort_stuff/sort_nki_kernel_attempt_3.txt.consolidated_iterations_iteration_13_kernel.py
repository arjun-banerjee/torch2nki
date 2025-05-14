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
    ndim = len(shape)
    
    # Initialize result arrays for sorted values and indices
    values = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    indices = nl.ndarray(shape, dtype=nl.int32, buffer=nl.shared_hbm)
    
    # Handle 1D case
    if ndim == 1:
        size = shape[0]
        max_tile_size = 128  # nl.tile_size.pmax
        
        # Process in tiles
        for start_idx in nl.affine_range(math.ceil(size / max_tile_size)):
            # Calculate actual tile size for this iteration
            tile_size = min(max_tile_size, size - start_idx * max_tile_size)
            
            # Load tile data
            indices_tile = nl.zeros((tile_size,), dtype=nl.int32, buffer=nl.sbuf)
            values_tile = nl.zeros((tile_size,), dtype=a_tensor.dtype, buffer=nl.sbuf)
            
            # Initialize indices and load values
            for i in nl.affine_range(tile_size):
                global_idx = start_idx * max_tile_size + i
                if global_idx < size:
                    indices_tile = nl.store(indices_tile, i, indices=(i,))
                    values_tile = nl.store(values_tile, nl.load(a_tensor[global_idx]), indices=(i,))
            
            # Bubble sort algorithm
            for i in nl.affine_range(tile_size):
                for j in nl.affine_range(tile_size - 1):
                    # Load current and next values
                    curr_val = nl.load(values_tile, indices=(j,))
                    next_val = nl.load(values_tile, indices=(j+1,))
                    
                    # Load current and next indices
                    curr_idx = nl.load(indices_tile, indices=(j,))
                    next_idx = nl.load(indices_tile, indices=(j+1,))
                    
                    # Compare and swap if needed
                    swap_needed = nl.greater(curr_val, next_val)
                    
                    # Update values using where
                    values_tile = nl.store(values_tile, 
                                           nl.where(swap_needed, next_val, curr_val), 
                                           indices=(j,))
                    values_tile = nl.store(values_tile, 
                                           nl.where(swap_needed, curr_val, next_val), 
                                           indices=(j+1,))
                    
                    # Update indices using where
                    indices_tile = nl.store(indices_tile, 
                                            nl.where(swap_needed, next_idx, curr_idx), 
                                            indices=(j,))
                    indices_tile = nl.store(indices_tile, 
                                            nl.where(swap_needed, curr_idx, next_idx), 
                                            indices=(j+1,))
            
            # Store sorted values and indices back to shared_hbm
            for i in nl.affine_range(tile_size):
                global_idx = start_idx * max_tile_size + i
                if global_idx < size:
                    val = nl.load(values_tile, indices=(i,))
                    idx = nl.load(indices_tile, indices=(i,))
                    nl.store(values, val, indices=(global_idx,))
                    nl.store(indices, idx + start_idx * max_tile_size, indices=(global_idx,))
                    
    # Handle 2D case
    elif ndim == 2:
        rows, cols = shape
        max_tile_rows = 128  # nl.tile_size.pmax
        
        # Process rows in tiles
        for start_row in nl.affine_range(math.ceil(rows / max_tile_rows)):
            # Calculate actual tile rows for this iteration
            tile_rows = min(max_tile_rows, rows - start_row * max_tile_rows)
            
            # Process each row within this tile
            for row_offset in nl.affine_range(tile_rows):
                row = start_row * max_tile_rows + row_offset
                if row >= rows:
                    continue
                
                # Create temporary arrays for this row
                row_values = nl.zeros((cols,), dtype=a_tensor.dtype, buffer=nl.sbuf)
                row_indices = nl.zeros((cols,), dtype=nl.int32, buffer=nl.sbuf)
                
                # Initialize indices and load values
                for col in nl.affine_range(cols):
                    row_values = nl.store(row_values, nl.load(a_tensor[row, col]), indices=(col,))
                    row_indices = nl.store(row_indices, col, indices=(col,))
                
                # Bubble sort algorithm for this row
                for i in nl.affine_range(cols):
                    for j in nl.affine_range(cols - 1):
                        # Load current and next values
                        curr_val = nl.load(row_values, indices=(j,))
                        next_val = nl.load(row_values, indices=(j+1,))
                        
                        # Load current and next indices
                        curr_idx = nl.load(row_indices, indices=(j,))
                        next_idx = nl.load(row_indices, indices=(j+1,))
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(curr_val, next_val)
                        
                        # Update values using where
                        row_values = nl.store(row_values, 
                                              nl.where(swap_needed, next_val, curr_val), 
                                              indices=(j,))
                        row_values = nl.store(row_values, 
                                              nl.where(swap_needed, curr_val, next_val), 
                                              indices=(j+1,))
                        
                        # Update indices using where
                        row_indices = nl.store(row_indices, 
                                               nl.where(swap_needed, next_idx, curr_idx), 
                                               indices=(j,))
                        row_indices = nl.store(row_indices, 
                                               nl.where(swap_needed, curr_idx, next_idx), 
                                               indices=(j+1,))
                
                # Store sorted values and indices back to shared_hbm
                for col in nl.affine_range(cols):
                    val = nl.load(row_values, indices=(col,))
                    idx = nl.load(row_indices, indices=(col,))
                    nl.store(values, val, indices=(row, col))
                    nl.store(indices, idx, indices=(row, col))
    
    # Return both values and indices
    result = (values, indices)
    return result