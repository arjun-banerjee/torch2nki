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
    
    # First copy the input tensor to result
    if ndim == 1:
        # Handle 1D case specifically
        size = shape[0]
        # Calculate number of tiles needed
        tile_size = min(nl.tile_size.pmax, size)
        trip_count = math.ceil(size / tile_size)
        
        # Copy data to result
        for p in nl.affine_range(trip_count):
            i_p = p * tile_size + nl.arange(tile_size)[:, None]
            in_tile = nl.load(a_tensor[i_p], mask=(i_p < size))
            nl.store(result[i_p], value=in_tile, mask=(i_p < size))
        
        # Bubble sort
        for i in nl.affine_range(size):
            for j in nl.affine_range(size - 1):
                for p in nl.affine_range(trip_count):
                    i_p = p * tile_size + nl.arange(tile_size)[:, None]
                    
                    # Load current elements
                    curr = nl.load(result[i_p], mask=(i_p < size))
                    
                    # Load next elements (shifted by 1)
                    next_idx = i_p + 1
                    next_val = nl.load(result[next_idx], mask=((i_p < size - 1) & (next_idx < size)))
                    
                    # Compare and swap if needed
                    swap_needed = nl.greater(curr, next_val)
                    
                    # Only swap if index is within range and swap is needed
                    mask = (i_p < size - 1) & swap_needed
                    
                    # Store swapped values
                    nl.store(result[i_p], value=next_val, mask=mask)
                    nl.store(result[next_idx], value=curr, mask=mask)
                    
    elif ndim == 2:
        # Handle 2D case
        rows, cols = shape
        
        # If sorting along dimension 0 (rows)
        if dim == 0:
            # Copy data to result
            for r in nl.affine_range(math.ceil(rows / nl.tile_size.pmax)):
                row_idx = r * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
                col_idx = nl.arange(cols)[None, :]
                
                in_tile = nl.load(a_tensor[row_idx, col_idx], mask=(row_idx < rows))
                nl.store(result[row_idx, col_idx], value=in_tile, mask=(row_idx < rows))
            
            # Sort each column independently
            for c in nl.affine_range(cols):
                # Bubble sort
                for i in nl.affine_range(rows):
                    for j in nl.affine_range(rows - 1):
                        for r in nl.affine_range(math.ceil(rows / nl.tile_size.pmax)):
                            row_idx = r * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
                            col_val = nl.full((nl.tile_size.pmax, 1), c, dtype=nl.int32)
                            
                            # Load current elements
                            curr = nl.load(result[row_idx, col_val], mask=(row_idx < rows))
                            
                            # Load next elements (shifted by 1)
                            next_row = row_idx + 1
                            next_val = nl.load(result[next_row, col_val], mask=((row_idx < rows - 1) & (next_row < rows)))
                            
                            # Compare and swap if needed
                            swap_needed = nl.greater(curr, next_val)
                            
                            # Only swap if index is within range and swap is needed
                            mask = (row_idx < rows - 1) & swap_needed
                            
                            # Store swapped values
                            nl.store(result[row_idx, col_val], value=next_val, mask=mask)
                            nl.store(result[next_row, col_val], value=curr, mask=mask)
                            
        # If sorting along dimension 1 (columns)
        else:
            # Copy data to result
            for r in nl.affine_range(math.ceil(rows / nl.tile_size.pmax)):
                row_idx = r * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
                col_idx = nl.arange(cols)[None, :]
                
                in_tile = nl.load(a_tensor[row_idx, col_idx], mask=(row_idx < rows))
                nl.store(result[row_idx, col_idx], value=in_tile, mask=(row_idx < rows))
            
            # Sort each row independently
            for r in nl.affine_range(rows):
                # Bubble sort
                for i in nl.affine_range(cols):
                    for j in nl.affine_range(cols - 1):
                        row_val = nl.full((1, 1), r, dtype=nl.int32)
                        
                        for c in nl.affine_range(math.ceil(cols / nl.tile_size.pmax)):
                            col_idx = c * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[None, :]
                            
                            # Load current elements
                            curr = nl.load(result[row_val, col_idx], mask=(col_idx < cols))
                            
                            # Load next elements (shifted by 1)
                            next_col = col_idx + 1
                            next_val = nl.load(result[row_val, next_col], mask=((col_idx < cols - 1) & (next_col < cols)))
                            
                            # Compare and swap if needed
                            swap_needed = nl.greater(curr, next_val)
                            
                            # Only swap if index is within range and swap is needed
                            mask = (col_idx < cols - 1) & swap_needed
                            
                            # Store swapped values
                            nl.store(result[row_val, col_idx], value=next_val, mask=mask)
                            nl.store(result[row_val, next_col], value=curr, mask=mask)
    else:
        # For higher dimensions, we can implement similar logic but this is a simple version for 1D and 2D
        pass
                            
    return result