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
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Handle 1D tensor case
    if ndim == 1:
        size = shape[0]
        # Copy input to result first
        for p in nl.affine_range(math.ceil(size / nl.tile_size.pmax)):
            start_idx = p * nl.tile_size.pmax
            tile_size = min(nl.tile_size.pmax, size - start_idx)
            
            i_p = nl.arange(nl.tile_size.pmax)
            input_tile = nl.load(a_tensor[start_idx + i_p], mask=(i_p < tile_size))
            nl.store(result[start_idx + i_p], value=input_tile, mask=(i_p < tile_size))
        
        # Bubble sort implementation
        for i in range(size - 1):
            for p in nl.affine_range(math.ceil(size / nl.tile_size.pmax)):
                start_idx = p * nl.tile_size.pmax
                tile_size = min(nl.tile_size.pmax, size - start_idx)
                
                i_p = nl.arange(nl.tile_size.pmax)
                curr_tile = nl.load(result[start_idx + i_p], mask=(i_p < tile_size))
                
                # Create shifted indices for comparison
                i_p_shifted = nl.arange(1, nl.tile_size.pmax + 1)
                next_idx = start_idx + i_p_shifted
                
                # Load next values, being careful at tile boundaries
                next_tile = nl.load(result[next_idx - 1], mask=((i_p < tile_size - 1) & (next_idx - 1 < size)))
                
                # Compare and swap if needed
                swap_condition = nl.greater(curr_tile, next_tile)
                
                # Store the smaller values in current positions
                smaller_values = nl.where(swap_condition, next_tile, curr_tile)
                nl.store(result[start_idx + i_p], value=smaller_values, mask=((i_p < tile_size - 1) & (next_idx - 1 < size)))
                
                # Store the larger values in next positions
                larger_values = nl.where(swap_condition, curr_tile, next_tile)
                nl.store(result[next_idx - 1], value=larger_values, mask=((i_p < tile_size - 1) & (next_idx - 1 < size)))
    
    # Handle 2D tensor case
    elif ndim == 2:
        rows, cols = shape
        
        # Sort along rows (dim=0)
        if dim == 0:
            # Initialize the result tensor by copying input
            for p in nl.affine_range(math.ceil(rows / nl.tile_size.pmax)):
                p_start = p * nl.tile_size.pmax
                p_size = min(nl.tile_size.pmax, rows - p_start)
                
                i_p = nl.arange(nl.tile_size.pmax)[:, None]
                i_f = nl.arange(cols)[None, :]
                
                input_tile = nl.load(a_tensor[p_start + i_p, i_f], mask=(i_p < p_size))
                nl.store(result[p_start + i_p, i_f], value=input_tile, mask=(i_p < p_size))
            
            # Bubble sort along dimension 0
            for i in range(rows - 1):
                for p in nl.affine_range(math.ceil(rows / nl.tile_size.pmax)):
                    p_start = p * nl.tile_size.pmax
                    p_size = min(nl.tile_size.pmax, rows - p_start)
                    
                    i_p = nl.arange(nl.tile_size.pmax)[:, None]
                    i_f = nl.arange(cols)[None, :]
                    
                    curr_tile = nl.load(result[p_start + i_p, i_f], mask=(i_p < p_size))
                    
                    # Create shifted indices for comparison
                    i_p_shifted = nl.arange(1, nl.tile_size.pmax + 1)[:, None]
                    next_idx = p_start + i_p_shifted
                    
                    # Load next values, being careful at boundaries
                    next_tile = nl.load(result[next_idx - 1, i_f], mask=((i_p < p_size - 1) & (next_idx - 1 < rows)))
                    
                    # Compare column-wise and swap if needed
                    for c in range(cols):
                        curr_col = curr_tile[:, c:c+1]
                        next_col = next_tile[:, c:c+1]
                        
                        swap_condition = nl.greater(curr_col, next_col)
                        
                        # Store smaller values
                        smaller_values = nl.where(swap_condition, next_col, curr_col)
                        nl.store(result[p_start + i_p, c], value=smaller_values, 
                                mask=((i_p < p_size - 1) & (next_idx - 1 < rows)))
                        
                        # Store larger values
                        larger_values = nl.where(swap_condition, curr_col, next_col)
                        nl.store(result[next_idx - 1, c], value=larger_values, 
                                mask=((i_p < p_size - 1) & (next_idx - 1 < rows)))
        
        # Sort along columns (dim=1)
        else:
            # Initialize the result tensor by copying input
            for p in nl.affine_range(math.ceil(rows / nl.tile_size.pmax)):
                p_start = p * nl.tile_size.pmax
                p_size = min(nl.tile_size.pmax, rows - p_start)
                
                i_p = nl.arange(nl.tile_size.pmax)[:, None]
                i_f = nl.arange(cols)[None, :]
                
                input_tile = nl.load(a_tensor[p_start + i_p, i_f], mask=(i_p < p_size))
                nl.store(result[p_start + i_p, i_f], value=input_tile, mask=(i_p < p_size))
            
            # Bubble sort along dimension 1
            for i in range(cols - 1):
                for p in nl.affine_range(math.ceil(rows / nl.tile_size.pmax)):
                    p_start = p * nl.tile_size.pmax
                    p_size = min(nl.tile_size.pmax, rows - p_start)
                    
                    i_p = nl.arange(nl.tile_size.pmax)[:, None]
                    i_f = nl.arange(cols)[None, :]
                    
                    curr_tile = nl.load(result[p_start + i_p, i_f], mask=(i_p < p_size))
                    
                    # Sort each row within the tile
                    for r in range(min(nl.tile_size.pmax, p_size)):
                        row_idx = p_start + r
                        if row_idx < rows:
                            row_data = curr_tile[r:r+1, :]
                            
                            # Compare and swap adjacent elements in the row
                            for j in range(cols - 1):
                                curr_val = row_data[0, j:j+1]
                                next_val = row_data[0, j+1:j+2]
                                
                                swap_condition = nl.greater(curr_val, next_val)
                                
                                # Perform swap if needed
                                if j < cols - 1:
                                    smaller_value = nl.where(swap_condition, next_val, curr_val)
                                    larger_value = nl.where(swap_condition, curr_val, next_val)
                                    
                                    row_data = nl.store(row_data[0, j:j+1], value=smaller_value)
                                    row_data = nl.store(row_data[0, j+1:j+2], value=larger_value)
                            
                            # Store back the sorted row
                            nl.store(result[row_idx, :], value=row_data)
    
    # Higher dimension tensors not supported in this implementation
    else:
        # For higher dimensions, we'd need to implement a more complex approach
        # This is a placeholder for potential future implementation
        pass
    
    return result