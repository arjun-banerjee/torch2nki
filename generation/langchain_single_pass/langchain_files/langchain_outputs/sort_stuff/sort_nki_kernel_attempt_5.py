from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Handle negative dimension
    ndim = len(a_tensor.shape)
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First copy input to result
    if ndim == 1:
        # 1D tensor case
        size = a_tensor.shape[0]
        tile_size = min(128, size)  # Respect hardware limitation
        
        # Copy input to result in tiles
        for i in nl.affine_range((size + tile_size - 1) // tile_size):
            start_idx = i * tile_size
            # Load from input
            indices = nl.arange(tile_size)[:, None]
            actual_indices = start_idx + indices
            mask = (actual_indices < size)
            tile_data = nl.load(a_tensor[actual_indices], mask=mask)
            
            # Store to result
            nl.store(result[actual_indices], value=tile_data, mask=mask)
        
        # Perform bubble sort on the entire array in chunks
        n = size
        for i in nl.affine_range(n):
            # Last i elements are already in place
            for j in nl.affine_range(n - 1):
                # Process in tiles
                for k in nl.affine_range((n - 1 + tile_size - 1) // tile_size):
                    start_idx = k * tile_size
                    # Load current elements
                    indices = nl.arange(tile_size)[:, None]
                    actual_indices = start_idx + indices
                    next_indices = actual_indices + 1
                    
                    # Make sure we're within bounds and not at the last element
                    mask = (actual_indices < n - 1) & (actual_indices < (n - i - 1))
                    
                    # Load current and next elements
                    current = nl.load(result[actual_indices], mask=mask)
                    next_elem = nl.load(result[next_indices], mask=mask)
                    
                    # Compare and swap if needed
                    swap_needed = nl.greater(current, next_elem, mask=mask)
                    new_current = nl.where(swap_needed, next_elem, current, mask=mask)
                    new_next = nl.where(swap_needed, current, next_elem, mask=mask)
                    
                    # Store results back
                    nl.store(result[actual_indices], value=new_current, mask=mask)
                    nl.store(result[next_indices], value=new_next, mask=mask)
    
    elif ndim == 2:
        # 2D tensor case - sort along specified dimension
        rows, cols = a_tensor.shape
        
        # Copy input to result in tiles
        if dim == 0:
            # Sort along rows
            max_tile_size = min(128, rows)
            for c in nl.affine_range(cols):
                # Copy column by column
                for r in nl.affine_range((rows + max_tile_size - 1) // max_tile_size):
                    start_idx = r * max_tile_size
                    indices = nl.arange(max_tile_size)[:, None]
                    actual_indices = start_idx + indices
                    mask = (actual_indices < rows)
                    
                    # Load from input
                    tile_data = nl.load(a_tensor[actual_indices, c], mask=mask)
                    
                    # Store to result
                    nl.store(result[actual_indices, c], value=tile_data, mask=mask)
                
                # Sort this column
                for i in nl.affine_range(rows):
                    for j in nl.affine_range(rows - 1):
                        for k in nl.affine_range((rows - 1 + max_tile_size - 1) // max_tile_size):
                            start_idx = k * max_tile_size
                            indices = nl.arange(max_tile_size)[:, None]
                            actual_indices = start_idx + indices
                            next_indices = actual_indices + 1
                            
                            mask = (actual_indices < rows - 1) & (actual_indices < (rows - i - 1))
                            
                            # Load current and next elements
                            current = nl.load(result[actual_indices, c], mask=mask)
                            next_elem = nl.load(result[next_indices, c], mask=mask)
                            
                            # Compare and swap if needed
                            swap_needed = nl.greater(current, next_elem, mask=mask)
                            new_current = nl.where(swap_needed, next_elem, current, mask=mask)
                            new_next = nl.where(swap_needed, current, next_elem, mask=mask)
                            
                            # Store results back
                            nl.store(result[actual_indices, c], value=new_current, mask=mask)
                            nl.store(result[next_indices, c], value=new_next, mask=mask)
        else:
            # Sort along columns (dim=1)
            max_tile_size = min(128, cols)
            for r in nl.affine_range(rows):
                # Copy row by row
                for c in nl.affine_range((cols + max_tile_size - 1) // max_tile_size):
                    start_idx = c * max_tile_size
                    indices = nl.arange(max_tile_size)[None, :]
                    actual_indices = start_idx + indices
                    mask = (actual_indices < cols)
                    
                    # Load from input
                    tile_data = nl.load(a_tensor[r, actual_indices], mask=mask)
                    
                    # Store to result
                    nl.store(result[r, actual_indices], value=tile_data, mask=mask)
                
                # Sort this row
                for i in nl.affine_range(cols):
                    for j in nl.affine_range(cols - 1):
                        for k in nl.affine_range((cols - 1 + max_tile_size - 1) // max_tile_size):
                            start_idx = k * max_tile_size
                            indices = nl.arange(max_tile_size)[None, :]
                            actual_indices = start_idx + indices
                            next_indices = actual_indices + 1
                            
                            mask = (actual_indices < cols - 1) & (actual_indices < (cols - i - 1))
                            
                            # Load current and next elements
                            current = nl.load(result[r, actual_indices], mask=mask)
                            next_elem = nl.load(result[r, next_indices], mask=mask)
                            
                            # Compare and swap if needed
                            swap_needed = nl.greater(current, next_elem, mask=mask)
                            new_current = nl.where(swap_needed, next_elem, current, mask=mask)
                            new_next = nl.where(swap_needed, current, next_elem, mask=mask)
                            
                            # Store results back
                            nl.store(result[r, actual_indices], value=new_current, mask=mask)
                            nl.store(result[r, next_indices], value=new_next, mask=mask)
    
    return result