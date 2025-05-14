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
    
    # Handle 1D tensor case
    if len(a_tensor.shape) == 1:
        size = a_tensor.shape[0]
        max_tile_size = nl.tile_size.pmax
        num_tiles = math.ceil(size / max_tile_size)
        
        # First, copy input to result
        for i in nl.affine_range(num_tiles):
            start_idx = i * max_tile_size
            # Create index array for current tile
            valid_indices = min(max_tile_size, size - start_idx)
            
            # Load data from input tensor
            input_data = nl.load(a_tensor[start_idx:start_idx + max_tile_size], 
                                mask=(nl.arange(max_tile_size) < valid_indices))
            
            # Store data to result tensor
            nl.store(result[start_idx:start_idx + max_tile_size], value=input_data,
                    mask=(nl.arange(max_tile_size) < valid_indices))
        
        # Perform bubble sort on the entire array
        for i in nl.affine_range(size - 1):
            for j in nl.affine_range(size - 1 - i):
                # We need to handle this in tiles due to hardware limitations
                for tile_idx in nl.affine_range(num_tiles):
                    start_idx = tile_idx * max_tile_size
                    valid_indices = min(max_tile_size, size - start_idx)
                    
                    # Load current tile
                    tile_data = nl.load(result[start_idx:start_idx + max_tile_size],
                                       mask=(nl.arange(max_tile_size) < valid_indices))
                    
                    # Process elements in the tile that correspond to index j
                    for k in nl.affine_range(valid_indices):
                        idx = start_idx + k
                        if idx == j:
                            # Load elements at j and j+1
                            val_j = nl.load(result[j])
                            val_j_plus_1 = nl.load(result[j+1])
                            
                            # Compare and swap if needed
                            cond = nl.greater(val_j, val_j_plus_1)
                            # Use where to select values based on condition
                            new_val_j = nl.where(cond, val_j_plus_1, val_j)
                            new_val_j_plus_1 = nl.where(cond, val_j, val_j_plus_1)
                            
                            # Store swapped values
                            nl.store(result[j], value=new_val_j)
                            nl.store(result[j+1], value=new_val_j_plus_1)
                            break
    
    # Handle multi-dimensional tensor case
    else:
        # Get the size of the dimension to sort along
        sort_dim_size = a_tensor.shape[dim]
        
        # Copy input to result first
        # We need to process in tiles due to hardware limitations
        if dim == 0:
            # Sorting along first dimension
            rows = a_tensor.shape[0]
            cols = a_tensor.shape[1] if len(a_tensor.shape) > 1 else 1
            
            max_tile_rows = nl.tile_size.pmax
            max_tile_cols = 128  # Assuming a reasonable tile size for columns
            
            num_row_tiles = math.ceil(rows / max_tile_rows)
            num_col_tiles = math.ceil(cols / max_tile_cols)
            
            # First, copy input to result
            for row_tile in nl.affine_range(num_row_tiles):
                start_row = row_tile * max_tile_rows
                valid_rows = min(max_tile_rows, rows - start_row)
                
                for col_tile in nl.affine_range(num_col_tiles):
                    start_col = col_tile * max_tile_cols
                    valid_cols = min(max_tile_cols, cols - start_col)
                    
                    # Create index ranges for the current tile
                    row_indices = start_row + nl.arange(max_tile_rows)[:, None]
                    col_indices = start_col + nl.arange(max_tile_cols)[None, :]
                    
                    # Load data from input tensor
                    input_data = nl.load(a_tensor[row_indices, col_indices], 
                                        mask=((row_indices < rows) & (col_indices < cols)))
                    
                    # Store data to result tensor
                    nl.store(result[row_indices, col_indices], value=input_data,
                           mask=((row_indices < rows) & (col_indices < cols)))
            
            # Perform bubble sort on each column
            for i in nl.affine_range(rows - 1):
                for j in nl.affine_range(rows - 1 - i):
                    for col_tile in nl.affine_range(num_col_tiles):
                        start_col = col_tile * max_tile_cols
                        valid_cols = min(max_tile_cols, cols - start_col)
                        
                        # Load elements at j and j+1 for all columns in this tile
                        val_j = nl.load(result[j, start_col:start_col + max_tile_cols],
                                      mask=(nl.arange(max_tile_cols) < valid_cols))
                        val_j_plus_1 = nl.load(result[j+1, start_col:start_col + max_tile_cols],
                                             mask=(nl.arange(max_tile_cols) < valid_cols))
                        
                        # Compare and swap if needed
                        cond = nl.greater(val_j, val_j_plus_1)
                        
                        # Use where to select values based on condition
                        new_val_j = nl.where(cond, val_j_plus_1, val_j)
                        new_val_j_plus_1 = nl.where(cond, val_j, val_j_plus_1)
                        
                        # Store swapped values
                        nl.store(result[j, start_col:start_col + max_tile_cols], value=new_val_j,
                               mask=(nl.arange(max_tile_cols) < valid_cols))
                        nl.store(result[j+1, start_col:start_col + max_tile_cols], value=new_val_j_plus_1,
                               mask=(nl.arange(max_tile_cols) < valid_cols))
        else:
            # Sorting along non-first dimension (typically last dimension)
            # For simplicity, we'll handle the 2D case where dim=1
            if len(a_tensor.shape) == 2 and dim == 1:
                rows = a_tensor.shape[0]
                cols = a_tensor.shape[1]
                
                max_tile_rows = nl.tile_size.pmax
                max_tile_cols = 128  # Assuming a reasonable tile size for columns
                
                num_row_tiles = math.ceil(rows / max_tile_rows)
                num_col_tiles = math.ceil(cols / max_tile_cols)
                
                # First, copy input to result
                for row_tile in nl.affine_range(num_row_tiles):
                    start_row = row_tile * max_tile_rows
                    valid_rows = min(max_tile_rows, rows - start_row)
                    
                    for col_tile in nl.affine_range(num_col_tiles):
                        start_col = col_tile * max_tile_cols
                        valid_cols = min(max_tile_cols, cols - start_col)
                        
                        # Create index ranges for the current tile
                        row_indices = start_row + nl.arange(max_tile_rows)[:, None]
                        col_indices = start_col + nl.arange(max_tile_cols)[None, :]
                        
                        # Load data from input tensor
                        input_data = nl.load(a_tensor[row_indices, col_indices], 
                                            mask=((row_indices < rows) & (col_indices < cols)))
                        
                        # Store data to result tensor
                        nl.store(result[row_indices, col_indices], value=input_data,
                               mask=((row_indices < rows) & (col_indices < cols)))
                
                # Perform bubble sort on each row
                for row_tile in nl.affine_range(num_row_tiles):
                    start_row = row_tile * max_tile_rows
                    valid_rows = min(max_tile_rows, rows - start_row)
                    
                    for i in nl.affine_range(valid_rows):
                        row_idx = start_row + i
                        
                        # Sort this row using bubble sort
                        for j in nl.affine_range(cols - 1):
                            for k in nl.affine_range(cols - 1 - j):
                                # Load elements at k and k+1
                                val_k = nl.load(result[row_idx, k])
                                val_k_plus_1 = nl.load(result[row_idx, k+1])
                                
                                # Compare and swap if needed
                                cond = nl.greater(val_k, val_k_plus_1)
                                
                                # Use where to select values based on condition
                                new_val_k = nl.where(cond, val_k_plus_1, val_k)
                                new_val_k_plus_1 = nl.where(cond, val_k, val_k_plus_1)
                                
                                # Store swapped values
                                nl.store(result[row_idx, k], value=new_val_k)
                                nl.store(result[row_idx, k+1], value=new_val_k_plus_1)
    
    return result