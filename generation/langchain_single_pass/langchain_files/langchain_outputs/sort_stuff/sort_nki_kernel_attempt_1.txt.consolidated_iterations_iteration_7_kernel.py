from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get tensor shape and normalize dimension
    shape = a_tensor.shape
    ndim = len(shape)
    
    # Convert negative dim to positive
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Handle 1D case
    if ndim == 1:
        size = shape[0]
        # Copy the input data to result first
        for i in nl.affine_range(math.ceil(size / nl.tile_size.pmax)):
            start_idx = i * nl.tile_size.pmax
            # Create indices for current tile
            idx = start_idx + nl.arange(nl.tile_size.pmax)
            # Load data with mask to handle boundary
            tile = nl.load(a_tensor[idx], mask=(idx < size))
            # Store data to result
            nl.store(result[idx], value=tile, mask=(idx < size))
        
        # Bubble sort implementation
        for i in nl.affine_range(size):
            for j in nl.affine_range(size - 1):
                # For each pair, load the values
                j_idx = nl.full((1, 1), j, dtype=nl.int32)[0, 0]
                j_next_idx = nl.full((1, 1), j + 1, dtype=nl.int32)[0, 0]
                
                # Load current and next value
                val_j = nl.load(result[j_idx])
                val_next = nl.load(result[j_next_idx])
                
                # Compare and swap if needed
                swap_needed = nl.greater(val_j, val_next)
                
                # Conditional swap
                if swap_needed:
                    nl.store(result[j_idx], value=val_next)
                    nl.store(result[j_next_idx], value=val_j)
    
    # Handle 2D case
    elif ndim == 2:
        rows, cols = shape
        
        if dim == 0:  # Sort along rows
            # Process each column
            for col in nl.affine_range(cols):
                col_idx = nl.full((1, 1), col, dtype=nl.int32)[0, 0]
                
                # Copy column to result first
                for i in nl.affine_range(math.ceil(rows / nl.tile_size.pmax)):
                    start_idx = i * nl.tile_size.pmax
                    # Create indices for current tile
                    idx = start_idx + nl.arange(nl.tile_size.pmax)
                    # Load data with mask to handle boundary
                    tile = nl.load(a_tensor[idx, col_idx], mask=(idx < rows))
                    # Store data to result
                    nl.store(result[idx, col_idx], value=tile, mask=(idx < rows))
                
                # Bubble sort the column
                for i in nl.affine_range(rows):
                    for j in nl.affine_range(rows - 1):
                        j_idx = nl.full((1, 1), j, dtype=nl.int32)[0, 0]
                        j_next_idx = nl.full((1, 1), j + 1, dtype=nl.int32)[0, 0]
                        
                        # Load current and next value
                        val_j = nl.load(result[j_idx, col_idx])
                        val_next = nl.load(result[j_next_idx, col_idx])
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(val_j, val_next)
                        
                        # Conditional swap
                        if swap_needed:
                            nl.store(result[j_idx, col_idx], value=val_next)
                            nl.store(result[j_next_idx, col_idx], value=val_j)
        
        else:  # Sort along columns (dim == 1)
            # Process each row
            for row in nl.affine_range(rows):
                row_idx = nl.full((1, 1), row, dtype=nl.int32)[0, 0]
                
                # Copy row to result first
                for i in nl.affine_range(math.ceil(cols / nl.tile_size.pmax)):
                    start_idx = i * nl.tile_size.pmax
                    # Create indices for current tile
                    idx = start_idx + nl.arange(nl.tile_size.pmax)
                    # Load data with mask to handle boundary
                    tile = nl.load(a_tensor[row_idx, idx], mask=(idx < cols))
                    # Store data to result
                    nl.store(result[row_idx, idx], value=tile, mask=(idx < cols))
                
                # Bubble sort the row
                for i in nl.affine_range(cols):
                    for j in nl.affine_range(cols - 1):
                        j_idx = nl.full((1, 1), j, dtype=nl.int32)[0, 0]
                        j_next_idx = nl.full((1, 1), j + 1, dtype=nl.int32)[0, 0]
                        
                        # Load current and next value
                        val_j = nl.load(result[row_idx, j_idx])
                        val_next = nl.load(result[row_idx, j_next_idx])
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(val_j, val_next)
                        
                        # Conditional swap
                        if swap_needed:
                            nl.store(result[row_idx, j_idx], value=val_next)
                            nl.store(result[row_idx, j_next_idx], value=val_j)
    
    # Handle higher dimensional case
    else:
        # Copy the entire tensor first
        flat_size = 1
        for d in range(ndim):
            flat_size *= shape[d]
        
        # Flatten the tensor for copying
        flattened_size = flat_size
        trip_count = math.ceil(flattened_size / nl.tile_size.pmax)
        
        # Copy input to result in tiles
        for i in nl.affine_range(trip_count):
            start_idx = i * nl.tile_size.pmax
            idx_range = start_idx + nl.arange(nl.tile_size.pmax)
            
            # Calculate multi-dimensional indices
            # For simplicity, we'll treat the tensor as flattened for copying
            # and then sort along the specified dimension
            flat_indices = idx_range
            
            # Load and store with boundary check
            input_tile = nl.load(a_tensor.reshape((-1,))[flat_indices], 
                                mask=(flat_indices < flattened_size))
            nl.store(result.reshape((-1,))[flat_indices], 
                     value=input_tile, mask=(flat_indices < flattened_size))
        
        # Now sort along the specified dimension
        # For higher dimensions, we need to iterate over all other dimensions
        # and sort each "line" along the sort dimension
        
        # Calculate the size of each dimension
        dim_sizes = []
        for d in range(ndim):
            dim_sizes.append(shape[d])
        
        # Calculate stride for each dimension
        strides = [1]
        for d in range(ndim-1, 0, -1):
            strides.insert(0, strides[0] * dim_sizes[d])
        
        # Size of the dimension to sort
        sort_size = dim_sizes[dim]
        
        # Number of "lines" to sort
        num_lines = flattened_size // sort_size
        
        # For each line
        for line_idx in nl.affine_range(num_lines):
            line_base = nl.full((1, 1), line_idx, dtype=nl.int32)[0, 0]
            
            # Calculate the base index for this line
            base_idx = 0
            remaining = line_base
            
            for d in range(ndim):
                if d != dim:
                    dim_idx = remaining // strides[d]
                    remaining = remaining % strides[d]
                    base_idx += dim_idx * strides[d]
            
            # Sort this line using bubble sort
            for i in nl.affine_range(sort_size):
                for j in nl.affine_range(sort_size - 1):
                    j_idx = nl.full((1, 1), j, dtype=nl.int32)[0, 0]
                    j_next_idx = nl.full((1, 1), j + 1, dtype=nl.int32)[0, 0]
                    
                    # Calculate the actual indices in the result tensor
                    idx1 = base_idx + j_idx * strides[dim]
                    idx2 = base_idx + j_next_idx * strides[dim]
                    
                    # Load values
                    val1 = nl.load(result.reshape((-1,))[idx1])
                    val2 = nl.load(result.reshape((-1,))[idx2])
                    
                    # Compare and swap if needed
                    swap_needed = nl.greater(val1, val2)
                    
                    # Conditional swap
                    if swap_needed:
                        nl.store(result.reshape((-1,))[idx1], value=val2)
                        nl.store(result.reshape((-1,))[idx2], value=val1)
    
    return result