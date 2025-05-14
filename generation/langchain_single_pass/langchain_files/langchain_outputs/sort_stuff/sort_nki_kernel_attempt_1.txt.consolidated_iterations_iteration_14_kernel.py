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
    
    # First copy the input tensor to result
    if ndim == 1:
        # Handle 1D case
        size = shape[0]
        max_tile_size = nl.tile_size.pmax
        
        # Process in tiles to respect hardware limitations
        for p in nl.affine_range(math.ceil(size / max_tile_size)):
            # Calculate indices for this tile
            start_idx = p * max_tile_size
            # Use arange with a column dimension to make it 2D
            i_p = nl.arange(max_tile_size)[:, None]
            
            # Load data with masking to handle boundary
            in_tile = nl.load(a_tensor[start_idx + i_p], mask=(start_idx + i_p < size))
            
            # Store to result
            nl.store(result[start_idx + i_p], value=in_tile, mask=(start_idx + i_p < size))
            
        # Now perform bubble sort
        for i in range(size):
            for j in range(0, size - i - 1):
                # Process in tiles for both j and j+1 indices
                j_tile = nl.load(result[j])
                j_next_tile = nl.load(result[j+1])
                
                # Compare and swap if needed
                swap_needed = nl.greater(j_tile, j_next_tile)
                temp = nl.copy(j_tile)
                j_tile = nl.where(swap_needed, j_next_tile, j_tile)
                j_next_tile = nl.where(swap_needed, temp, j_next_tile)
                
                # Store back
                nl.store(result[j], value=j_tile)
                nl.store(result[j+1], value=j_next_tile)
    else:
        # Handle multi-dimensional case
        sort_dim_size = shape[dim]
        
        # For each "row" (fixing all dimensions except the sort dimension)
        # We'll need to create indices for all dimensions
        
        # First copy input to result
        # Create a list to track the current indices for each dimension
        indices = [0] * ndim
        
        # Calculate the total number of "rows" to process
        total_rows = 1
        for d in range(ndim):
            if d != dim:
                total_rows *= shape[d]
        
        # Process each row
        for row_idx in range(total_rows):
            # Calculate the multi-dimensional indices for this row
            remaining_idx = row_idx
            for d in range(ndim):
                if d != dim:
                    dim_size = shape[d]
                    indices[d] = remaining_idx % dim_size
                    remaining_idx //= dim_size
            
            # Create indices for the sort dimension
            for i in range(sort_dim_size):
                indices[dim] = i
                
                # Load element from this position
                # We need to handle the indices properly
                if ndim == 2:
                    if dim == 0:
                        src_val = nl.load(a_tensor[i, indices[1]])
                        nl.store(result[i, indices[1]], value=src_val)
                    else:
                        src_val = nl.load(a_tensor[indices[0], i])
                        nl.store(result[indices[0], i], value=src_val)
                elif ndim == 3:
                    if dim == 0:
                        src_val = nl.load(a_tensor[i, indices[1], indices[2]])
                        nl.store(result[i, indices[1], indices[2]], value=src_val)
                    elif dim == 1:
                        src_val = nl.load(a_tensor[indices[0], i, indices[2]])
                        nl.store(result[indices[0], i, indices[2]], value=src_val)
                    else:
                        src_val = nl.load(a_tensor[indices[0], indices[1], i])
                        nl.store(result[indices[0], indices[1], i], value=src_val)
            
            # Now bubble sort this row
            for i in range(sort_dim_size):
                for j in range(sort_dim_size - i - 1):
                    # Load the two elements to compare
                    indices[dim] = j
                    if ndim == 2:
                        if dim == 0:
                            val1 = nl.load(result[j, indices[1]])
                            val2 = nl.load(result[j+1, indices[1]])
                        else:
                            val1 = nl.load(result[indices[0], j])
                            val2 = nl.load(result[indices[0], j+1])
                    elif ndim == 3:
                        if dim == 0:
                            val1 = nl.load(result[j, indices[1], indices[2]])
                            val2 = nl.load(result[j+1, indices[1], indices[2]])
                        elif dim == 1:
                            val1 = nl.load(result[indices[0], j, indices[2]])
                            val2 = nl.load(result[indices[0], j+1, indices[2]])
                        else:
                            val1 = nl.load(result[indices[0], indices[1], j])
                            val2 = nl.load(result[indices[0], indices[1], j+1])
                    
                    # Compare and swap if needed
                    swap_needed = nl.greater(val1, val2)
                    temp = nl.copy(val1)
                    val1 = nl.where(swap_needed, val2, val1)
                    val2 = nl.where(swap_needed, temp, val2)
                    
                    # Store back
                    indices[dim] = j
                    if ndim == 2:
                        if dim == 0:
                            nl.store(result[j, indices[1]], value=val1)
                            nl.store(result[j+1, indices[1]], value=val2)
                        else:
                            nl.store(result[indices[0], j], value=val1)
                            nl.store(result[indices[0], j+1], value=val2)
                    elif ndim == 3:
                        if dim == 0:
                            nl.store(result[j, indices[1], indices[2]], value=val1)
                            nl.store(result[j+1, indices[1], indices[2]], value=val2)
                        elif dim == 1:
                            nl.store(result[indices[0], j, indices[2]], value=val1)
                            nl.store(result[indices[0], j+1, indices[2]], value=val2)
                        else:
                            nl.store(result[indices[0], indices[1], j], value=val1)
                            nl.store(result[indices[0], indices[1], j+1], value=val2)
    
    return result