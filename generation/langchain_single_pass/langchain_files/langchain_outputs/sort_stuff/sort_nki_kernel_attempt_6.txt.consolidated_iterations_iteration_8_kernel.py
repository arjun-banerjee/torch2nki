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
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First copy the input tensor to result
    # We need to process in tiles to handle large tensors
    if len(shape) == 1:
        # Handle 1D tensor
        size = shape[0]
        tile_size = min(size, nl.tile_size.pmax)
        trip_count = math.ceil(size / tile_size)
        
        # Copy input to result
        for i in nl.affine_range(trip_count):
            i_p = i * tile_size + nl.arange(tile_size)[:, None]
            val_tile = nl.load(a_tensor[i_p], mask=(i_p < size))
            nl.store(result[i_p], val_tile, mask=(i_p < size))
        
        # Bubble sort
        for i in nl.affine_range(size):
            for j in nl.affine_range(size - 1):
                # Process in tiles
                for k in nl.affine_range(trip_count):
                    k_p = k * tile_size + nl.arange(tile_size)[:, None]
                    
                    # Only process elements at position j
                    mask = (k_p == j) & (k_p < size - 1)
                    if mask.any():
                        # Load current and next element
                        curr = nl.load(result[j], mask=mask)
                        next_val = nl.load(result[j + 1], mask=mask)
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(curr, next_val)
                        if swap_needed.any():
                            nl.store(result[j], next_val, mask=mask & swap_needed)
                            nl.store(result[j + 1], curr, mask=mask & swap_needed)
    
    elif len(shape) == 2:
        # Handle 2D tensor
        rows, cols = shape
        
        if dim == 0:
            # Sort each column
            for col in nl.affine_range(cols):
                # Process each column in tiles
                tile_size = min(rows, nl.tile_size.pmax)
                trip_count = math.ceil(rows / tile_size)
                
                # Copy input to result for this column
                for i in nl.affine_range(trip_count):
                    i_p = i * tile_size + nl.arange(tile_size)[:, None]
                    col_idx = nl.full((tile_size, 1), col, dtype=nl.int32)
                    val_tile = nl.load(a_tensor[i_p, col_idx], mask=(i_p < rows))
                    nl.store(result[i_p, col_idx], val_tile, mask=(i_p < rows))
                
                # Bubble sort for this column
                for i in nl.affine_range(rows):
                    for j in nl.affine_range(rows - 1):
                        # Load current and next element
                        curr = nl.load(result[j, col])
                        next_val = nl.load(result[j + 1, col])
                        
                        # Compare and swap if needed
                        if nl.greater(curr, next_val).item():
                            nl.store(result[j, col], next_val)
                            nl.store(result[j + 1, col], curr)
        
        else:  # dim == 1
            # Sort each row
            tile_size = min(cols, nl.tile_size.pmax)
            
            # Process each row
            for row in nl.affine_range(rows):
                # Copy input to result for this row
                trip_count = math.ceil(cols / tile_size)
                for i in nl.affine_range(trip_count):
                    i_p = i * tile_size + nl.arange(tile_size)[None, :]
                    row_idx = nl.full((1, tile_size), row, dtype=nl.int32)
                    val_tile = nl.load(a_tensor[row_idx, i_p], mask=(i_p < cols))
                    nl.store(result[row_idx, i_p], val_tile, mask=(i_p < cols))
                
                # Bubble sort for this row
                for i in nl.affine_range(cols):
                    for j in nl.affine_range(cols - 1):
                        # Process in tiles if needed
                        if j < cols - 1:
                            # Load current and next element
                            curr = nl.load(result[row, j])
                            next_val = nl.load(result[row, j + 1])
                            
                            # Compare and swap if needed
                            if nl.greater(curr, next_val).item():
                                nl.store(result[row, j], next_val)
                                nl.store(result[row, j + 1], curr)
    
    else:
        # For higher dimensions, we only handle sorting along the last dimension for now
        # This could be extended to handle other dimensions with more complex indexing
        if dim != len(shape) - 1:
            # For simplicity, only supporting last dimension sorting for now
            pass
        
        # Get the size of the last dimension
        last_dim_size = shape[-1]
        
        # Calculate the total number of slices to sort
        total_slices = 1
        for i in range(len(shape) - 1):
            total_slices *= shape[i]
        
        # Process each slice
        for slice_idx in nl.affine_range(total_slices):
            # This is a simple bubble sort for the last dimension
            for i in nl.affine_range(last_dim_size):
                for j in nl.affine_range(last_dim_size - 1):
                    # For higher dimensions, we would need more complex indexing here
                    # This is left as a placeholder
                    pass
    
    return result