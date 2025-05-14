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
    
    # Copy input to result first (we'll sort in-place in the result tensor)
    if len(a_tensor.shape) == 1:
        # 1D tensor case
        size = a_tensor.shape[0]
        trip_count = math.ceil(size / nl.tile_size.pmax)
        
        # Copy input to result in tiles
        for p in nl.affine_range(trip_count):
            i = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            in_tile = nl.load(a_tensor[i], mask=(i < size))
            nl.store(result[i], value=in_tile, mask=(i < size))
        
        # Bubble sort - always do size iterations to avoid dynamic control flow
        for i in nl.affine_range(size):
            # Process in tiles to respect architecture limitations
            for p in nl.affine_range(trip_count - 1):
                # Load current tile
                idx_curr = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                curr_tile = nl.load(result[idx_curr], mask=(idx_curr < size))
                
                # Load next element (could be in same tile or next tile)
                idx_next = idx_curr + 1
                next_tile = nl.load(result[idx_next], mask=(idx_next < size))
                
                # Determine if swap is needed
                swap_needed = nl.greater(curr_tile, next_tile)
                
                # Perform swap using where
                new_curr = nl.where(swap_needed, next_tile, curr_tile)
                new_next = nl.where(swap_needed, curr_tile, next_tile)
                
                # Store back
                nl.store(result[idx_curr], value=new_curr, mask=(idx_curr < size))
                nl.store(result[idx_next], value=new_next, mask=(idx_next < size))
                
    elif len(a_tensor.shape) == 2:
        # 2D tensor case
        rows, cols = a_tensor.shape
        
        if dim == 0:
            # Sort along rows
            trip_count_rows = math.ceil(rows / nl.tile_size.pmax)
            
            # Copy input to result in tiles
            for p in nl.affine_range(trip_count_rows):
                i_r = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
                i_c = nl.arange(cols)[None, :]
                in_tile = nl.load(a_tensor[i_r, i_c], mask=(i_r < rows))
                nl.store(result[i_r, i_c], value=in_tile, mask=(i_r < rows))
            
            # Sort each column independently
            for c in nl.affine_range(cols):
                # Bubble sort for this column
                for i in nl.affine_range(rows):
                    for p in nl.affine_range(trip_count_rows - 1):
                        # Generate indices
                        idx_curr = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                        idx_next = idx_curr + 1
                        
                        # Load current and next elements
                        curr_vals = nl.load(result[idx_curr, c], mask=(idx_curr < rows))
                        next_vals = nl.load(result[idx_next, c], mask=(idx_next < rows))
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(curr_vals, next_vals)
                        new_curr = nl.where(swap_needed, next_vals, curr_vals)
                        new_next = nl.where(swap_needed, curr_vals, next_vals)
                        
                        # Store back
                        nl.store(result[idx_curr, c], value=new_curr, mask=(idx_curr < rows))
                        nl.store(result[idx_next, c], value=new_next, mask=(idx_next < rows))
                        
        else:  # dim == 1, sort along columns
            trip_count_cols = math.ceil(cols / nl.tile_size.pmax)
            
            # Copy input to result in tiles
            for p in nl.affine_range(trip_count_cols):
                i_c = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[None, :]
                i_r = nl.arange(rows)[:, None]
                in_tile = nl.load(a_tensor[i_r, i_c], mask=(i_c < cols))
                nl.store(result[i_r, i_c], value=in_tile, mask=(i_c < cols))
            
            # Sort each row independently
            for r in nl.affine_range(rows):
                # Bubble sort for this row
                for i in nl.affine_range(cols):
                    for p in nl.affine_range(trip_count_cols - 1):
                        # Generate indices
                        idx_curr = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                        idx_next = idx_curr + 1
                        
                        # Load current and next elements
                        curr_vals = nl.load(result[r, idx_curr], mask=(idx_curr < cols))
                        next_vals = nl.load(result[r, idx_next], mask=(idx_next < cols))
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(curr_vals, next_vals)
                        new_curr = nl.where(swap_needed, next_vals, curr_vals)
                        new_next = nl.where(swap_needed, curr_vals, next_vals)
                        
                        # Store back
                        nl.store(result[r, idx_curr], value=new_curr, mask=(idx_curr < cols))
                        nl.store(result[r, idx_next], value=new_next, mask=(idx_next < cols))
    
    return result