from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Handle negative dimension indexing
    if dim < 0:
        dim = dim + len(a_tensor.shape)
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Get tensor shape
    tensor_shape = a_tensor.shape
    ndims = len(tensor_shape)
    
    # First, copy input to result
    if ndims == 1:
        # For 1D tensors, we need to handle the sorting directly
        sort_dim_size = tensor_shape[0]
        
        # Process in tiles to respect hardware limitations
        max_tile_size = nl.tile_size.pmax
        trip_count = math.ceil(sort_dim_size / max_tile_size)
        
        # First, copy the input to result
        for p in nl.affine_range(trip_count):
            # Generate tensor indices for the current tile
            i_p = p * max_tile_size + nl.arange(max_tile_size)
            
            # Load input data from external memory to on-chip memory
            x_tile = nl.load(a_tensor[i_p], mask=(i_p < sort_dim_size))
            
            # Store to result
            nl.store(result[i_p], value=x_tile, mask=(i_p < sort_dim_size))
        
        # Bubble sort implementation
        for i in range(sort_dim_size):
            for p in nl.affine_range(trip_count):
                i_p = p * max_tile_size + nl.arange(max_tile_size)
                
                # Load current segment
                current_tile = nl.load(result[i_p], mask=(i_p < sort_dim_size))
                
                # For each position in the tile
                for j in range(max_tile_size-1):
                    # Skip if out of bounds
                    if p * max_tile_size + j + 1 >= sort_dim_size:
                        continue
                    
                    # Compare adjacent elements
                    if j + 1 < max_tile_size:
                        # Both elements are in the same tile
                        left = current_tile[j]
                        right = current_tile[j+1]
                        
                        # If left > right, swap them
                        condition = nl.greater(left, right)
                        current_tile = nl.load(current_tile)  # Make a copy to modify
                        
                        # Perform swap using where
                        temp_left = nl.where(condition, right, left)
                        temp_right = nl.where(condition, left, right)
                        
                        current_tile[j] = temp_left
                        current_tile[j+1] = temp_right
                        
                # Store the updated tile back to result
                nl.store(result[i_p], value=current_tile, mask=(i_p < sort_dim_size))
                
                # Handle boundary between tiles
                if p < trip_count - 1:
                    # Load current tile's last element and next tile's first element
                    last_idx = (p+1) * max_tile_size - 1
                    next_idx = (p+1) * max_tile_size
                    
                    if next_idx < sort_dim_size:
                        last = nl.load(result[last_idx])
                        next_first = nl.load(result[next_idx])
                        
                        # If last > next_first, swap them
                        condition = nl.greater(last, next_first)
                        temp_last = nl.where(condition, next_first, last)
                        temp_next = nl.where(condition, last, next_first)
                        
                        nl.store(result[last_idx], value=temp_last)
                        nl.store(result[next_idx], value=temp_next)
    else:
        # For multi-dimensional tensors, we need to handle the sorting dimension
        # First, copy the input to result
        
        # Calculate total elements
        total_elements = 1
        for i in range(ndims):
            total_elements *= tensor_shape[i]
        
        # Determine max tile size for batch processing
        max_tile_size = nl.tile_size.pmax
        sort_dim_size = tensor_shape[dim]
        
        # For simplicity, we'll handle 2D tensors with sorting along different dimensions
        if ndims == 2:
            if dim == 0:
                # Sort along first dimension (rows)
                rows, cols = tensor_shape
                
                # First copy input to result
                for r in nl.affine_range(math.ceil(rows / max_tile_size)):
                    r_indices = r * max_tile_size + nl.arange(max_tile_size)[:, None]
                    c_indices = nl.arange(cols)[None, :]
                    
                    # Load data
                    data_tile = nl.load(a_tensor[r_indices, c_indices], mask=(r_indices < rows))
                    
                    # Store to result
                    nl.store(result[r_indices, c_indices], value=data_tile, mask=(r_indices < rows))
                
                # For each column, sort elements in that column
                for c in range(cols):
                    # Bubble sort for this column
                    for i in range(rows):
                        for r in nl.affine_range(math.ceil(rows / max_tile_size)):
                            r_indices = r * max_tile_size + nl.arange(max_tile_size)
                            
                            # Load current segment of the column
                            col_data = nl.load(result[r_indices, c], mask=(r_indices < rows))
                            
                            # For each position in the tile
                            for j in range(max_tile_size-1):
                                actual_j = r * max_tile_size + j
                                # Skip if out of bounds
                                if actual_j + 1 >= rows:
                                    continue
                                
                                # Compare adjacent elements
                                if j + 1 < max_tile_size:
                                    # Both elements are in the same tile
                                    left = col_data[j]
                                    right = col_data[j+1]
                                    
                                    # If left > right, swap them
                                    condition = nl.greater(left, right)
                                    
                                    # Perform swap using where
                                    temp_left = nl.where(condition, right, left)
                                    temp_right = nl.where(condition, left, right)
                                    
                                    col_data[j] = temp_left
                                    col_data[j+1] = temp_right
                                    
                            # Store the updated column segment back to result
                            nl.store(result[r_indices, c], value=col_data, mask=(r_indices < rows))
                            
                            # Handle boundary between tiles
                            if r < math.ceil(rows / max_tile_size) - 1:
                                last_idx = (r+1) * max_tile_size - 1
                                next_idx = (r+1) * max_tile_size
                                
                                if next_idx < rows:
                                    last = nl.load(result[last_idx, c])
                                    next_first = nl.load(result[next_idx, c])
                                    
                                    # If last > next_first, swap them
                                    condition = nl.greater(last, next_first)
                                    temp_last = nl.where(condition, next_first, last)
                                    temp_next = nl.where(condition, last, next_first)
                                    
                                    nl.store(result[last_idx, c], value=temp_last)
                                    nl.store(result[next_idx, c], value=temp_next)
            else:
                # Sort along second dimension (columns)
                rows, cols = tensor_shape
                
                # First copy input to result
                for r in nl.affine_range(math.ceil(rows / max_tile_size)):
                    r_indices = r * max_tile_size + nl.arange(max_tile_size)[:, None]
                    c_indices = nl.arange(cols)[None, :]
                    
                    # Load data
                    data_tile = nl.load(a_tensor[r_indices, c_indices], mask=(r_indices < rows))
                    
                    # Store to result
                    nl.store(result[r_indices, c_indices], value=data_tile, mask=(r_indices < rows))
                
                # For each row, sort elements in that row
                for r in range(rows):
                    # Bubble sort for this row
                    for i in range(cols):
                        for c in nl.affine_range(math.ceil(cols / max_tile_size)):
                            c_indices = c * max_tile_size + nl.arange(max_tile_size)
                            
                            # Load current segment of the row
                            row_data = nl.load(result[r, c_indices], mask=(c_indices < cols))
                            
                            # For each position in the tile
                            for j in range(max_tile_size-1):
                                actual_j = c * max_tile_size + j
                                # Skip if out of bounds
                                if actual_j + 1 >= cols:
                                    continue
                                
                                # Compare adjacent elements
                                if j + 1 < max_tile_size:
                                    # Both elements are in the same tile
                                    left = row_data[j]
                                    right = row_data[j+1]
                                    
                                    # If left > right, swap them
                                    condition = nl.greater(left, right)
                                    
                                    # Perform swap using where
                                    temp_left = nl.where(condition, right, left)
                                    temp_right = nl.where(condition, left, right)
                                    
                                    row_data[j] = temp_left
                                    row_data[j+1] = temp_right
                                    
                            # Store the updated row segment back to result
                            nl.store(result[r, c_indices], value=row_data, mask=(c_indices < cols))
                            
                            # Handle boundary between tiles
                            if c < math.ceil(cols / max_tile_size) - 1:
                                last_idx = (c+1) * max_tile_size - 1
                                next_idx = (c+1) * max_tile_size
                                
                                if next_idx < cols:
                                    last = nl.load(result[r, last_idx])
                                    next_first = nl.load(result[r, next_idx])
                                    
                                    # If last > next_first, swap them
                                    condition = nl.greater(last, next_first)
                                    temp_last = nl.where(condition, next_first, last)
                                    temp_next = nl.where(condition, last, next_first)
                                    
                                    nl.store(result[r, last_idx], value=temp_last)
                                    nl.store(result[r, next_idx], value=temp_next)
    
    return result