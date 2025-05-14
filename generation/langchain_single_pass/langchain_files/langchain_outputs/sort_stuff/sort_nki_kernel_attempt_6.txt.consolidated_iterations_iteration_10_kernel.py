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
    ndim = len(shape)
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # We need to handle the case where we sort along different dimensions
    # First, determine the size of the dimension to sort
    sort_dim_size = shape[dim]
    
    # Calculate the number of independent sort operations we need to perform
    # This is the product of all dimensions except the sort dimension
    num_sorts = 1
    for i in range(ndim):
        if i != dim:
            num_sorts *= shape[i]
    
    # Maximum partition size for tiling
    tile_size = min(nl.tile_size.pmax, sort_dim_size)
    
    # First, copy input to result tensor
    # We process in tiles to handle tensors of any size
    for i in nl.affine_range(math.ceil(num_sorts / tile_size)):
        start_idx = i * tile_size
        end_idx = min((i + 1) * tile_size, num_sorts)
        actual_size = end_idx - start_idx
        
        # Create indices for loading the current tile
        if dim == 0:
            # Sort along first dimension
            indices_p = nl.arange(tile_size)[:, None]
            indices_f = nl.arange(shape[1])[None, :]
            
            # Load current tile
            tile_data = nl.load(a_tensor[indices_p, indices_f], mask=(indices_p < sort_dim_size))
            
            # Store to result
            nl.store(result[indices_p, indices_f], value=tile_data, mask=(indices_p < sort_dim_size))
        else:
            # Sort along last dimension (dim == 1 for 2D tensor)
            indices_p = nl.arange(tile_size)[:, None]
            indices_f = nl.arange(shape[0])[None, :]
            
            # Load current tile
            tile_data = nl.load(a_tensor[indices_f, indices_p], mask=(indices_p < sort_dim_size))
            
            # Store to result
            nl.store(result[indices_f, indices_p], value=tile_data, mask=(indices_p < sort_dim_size))
    
    # Now perform bubble sort on result tensor
    # For simplicity, assuming a 2D tensor with sort_dim either 0 or 1
    if ndim == 2:
        if dim == 0:
            # Sort along first dimension
            for i in nl.affine_range(sort_dim_size):
                for j in nl.affine_range(sort_dim_size - 1):
                    # Process in tiles
                    for tile_idx in nl.affine_range(math.ceil(shape[1] / tile_size)):
                        start_col = tile_idx * tile_size
                        end_col = min((tile_idx + 1) * tile_size, shape[1])
                        actual_col_size = end_col - start_col
                        
                        # Create indices for the current tile
                        col_indices = start_col + nl.arange(tile_size)[None, :]
                        
                        # Load current and next row elements
                        current_row_idx = nl.arange(1)[:, None] * 0 + j
                        next_row_idx = nl.arange(1)[:, None] * 0 + (j + 1)
                        
                        current = nl.load(result[current_row_idx, col_indices], mask=(col_indices < shape[1]))
                        next_val = nl.load(result[next_row_idx, col_indices], mask=(col_indices < shape[1]))
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(current, next_val)
                        
                        # Conditionally swap elements
                        new_current = nl.where(swap_needed, next_val, current)
                        new_next = nl.where(swap_needed, current, next_val)
                        
                        # Store back
                        nl.store(result[current_row_idx, col_indices], value=new_current, mask=(col_indices < shape[1]))
                        nl.store(result[next_row_idx, col_indices], value=new_next, mask=(col_indices < shape[1]))
        else:
            # Sort along last dimension (dim == 1)
            for i in nl.affine_range(shape[0]):
                for j in nl.affine_range(sort_dim_size - 1):
                    # Process in tiles
                    for tile_idx in nl.affine_range(math.ceil(sort_dim_size / tile_size)):
                        start_col = tile_idx * tile_size
                        end_col = min((tile_idx + 1) * tile_size, sort_dim_size)
                        
                        # Create indices for the current tile
                        row_idx = nl.arange(1)[:, None] * 0 + i
                        col_indices = start_col + nl.arange(tile_size)[None, :]
                        
                        # We only need to compare elements at position j and j+1
                        # Calculate the actual indices in the tile
                        j_in_tile = j - start_col
                        
                        # Only process if j is in the current tile and j+1 is also in the tile
                        if j >= start_col and j < end_col - 1:
                            j_idx = nl.arange(1)[None, :] * 0 + j
                            j_next_idx = nl.arange(1)[None, :] * 0 + (j + 1)
                            
                            # Load elements at j and j+1
                            current = nl.load(result[row_idx, j_idx])
                            next_val = nl.load(result[row_idx, j_next_idx])
                            
                            # Compare and swap if needed
                            if nl.greater(current, next_val).item():
                                # Swap elements
                                nl.store(result[row_idx, j_idx], value=next_val)
                                nl.store(result[row_idx, j_next_idx], value=current)
    
    # For 1D tensor
    elif ndim == 1:
        for i in nl.affine_range(sort_dim_size):
            for j in nl.affine_range(sort_dim_size - 1):
                # Process in tiles
                for tile_idx in nl.affine_range(math.ceil(sort_dim_size / tile_size)):
                    start_idx = tile_idx * tile_size
                    end_idx = min((tile_idx + 1) * tile_size, sort_dim_size)
                    
                    # Only process if j is in the current tile and j+1 is also in the tile
                    if j >= start_idx and j < end_idx - 1:
                        j_idx = nl.arange(1)[:, None] * 0 + j
                        j_next_idx = nl.arange(1)[:, None] * 0 + (j + 1)
                        
                        # Load elements at j and j+1
                        current = nl.load(result[j_idx])
                        next_val = nl.load(result[j_next_idx])
                        
                        # Compare and swap if needed
                        if nl.greater(current, next_val).item():
                            # Swap elements
                            nl.store(result[j_idx], value=next_val)
                            nl.store(result[j_next_idx], value=current)
    
    return result