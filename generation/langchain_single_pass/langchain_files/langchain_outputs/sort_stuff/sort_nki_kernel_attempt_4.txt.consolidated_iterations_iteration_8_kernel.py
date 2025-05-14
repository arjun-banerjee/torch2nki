from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get shape information
    shape = a_tensor.shape
    ndim = len(shape)
    
    # If dim is negative, convert to positive index
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First, copy input to result
    if ndim == 1:
        # Handle 1D tensor case
        size = shape[0]
        trip_count = math.ceil(size / nl.tile_size.pmax)
        
        for p in nl.affine_range(trip_count):
            # Generate tensor indices for the current tile
            idx = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            
            # Load input data from external memory to on-chip memory
            src_tile = nl.load(a_tensor[idx], mask=(idx < size))
            
            # Store to result
            nl.store(result[idx], value=src_tile, mask=(idx < size))
            
        # Bubble sort the 1D array
        for i in nl.affine_range(size):
            for j in nl.affine_range(size - 1):
                # Load adjacent elements
                idx_j = nl.arange(1)
                idx_j1 = nl.arange(1) + 1
                
                val_j = nl.load(result[j])
                val_j1 = nl.load(result[j+1])
                
                # Compare and swap if necessary
                swap_needed = nl.greater(val_j, val_j1)
                
                # Prepare values after potential swap
                new_val_j = nl.where(swap_needed, val_j1, val_j)
                new_val_j1 = nl.where(swap_needed, val_j, val_j1)
                
                # Store back
                nl.store(result[j], value=new_val_j)
                nl.store(result[j+1], value=new_val_j1)
    
    elif ndim == 2:
        rows, cols = shape[0], shape[1]
        
        # Determine which dimension to sort along
        if dim == 0:  # Sort along rows
            for col in nl.affine_range(cols):
                # Extract this column
                column_size = rows
                trip_count = math.ceil(column_size / nl.tile_size.pmax)
                
                # Copy column to result
                for p in nl.affine_range(trip_count):
                    idx_r = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                    
                    # Load input data
                    src_tile = nl.load(a_tensor[idx_r, col], mask=(idx_r < rows))
                    
                    # Store to result
                    nl.store(result[idx_r, col], value=src_tile, mask=(idx_r < rows))
                
                # Bubble sort this column
                for i in nl.affine_range(rows):
                    for j in nl.affine_range(rows - 1):
                        # Load adjacent elements
                        val_j = nl.load(result[j, col])
                        val_j1 = nl.load(result[j+1, col])
                        
                        # Compare and swap if necessary
                        swap_needed = nl.greater(val_j, val_j1)
                        
                        # Prepare values after potential swap
                        new_val_j = nl.where(swap_needed, val_j1, val_j)
                        new_val_j1 = nl.where(swap_needed, val_j, val_j1)
                        
                        # Store back
                        nl.store(result[j, col], value=new_val_j)
                        nl.store(result[j+1, col], value=new_val_j1)
                
        else:  # Sort along columns (dim == 1)
            for row in nl.affine_range(rows):
                # Extract this row
                row_size = cols
                trip_count = math.ceil(row_size / nl.tile_size.pmax)
                
                # Copy row to result
                for p in nl.affine_range(trip_count):
                    idx_c = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                    
                    # Load input data
                    src_tile = nl.load(a_tensor[row, idx_c], mask=(idx_c < cols))
                    
                    # Store to result
                    nl.store(result[row, idx_c], value=src_tile, mask=(idx_c < cols))
                
                # Bubble sort this row
                for i in nl.affine_range(cols):
                    for j in nl.affine_range(cols - 1):
                        # Load adjacent elements
                        val_j = nl.load(result[row, j])
                        val_j1 = nl.load(result[row, j+1])
                        
                        # Compare and swap if necessary
                        swap_needed = nl.greater(val_j, val_j1)
                        
                        # Prepare values after potential swap
                        new_val_j = nl.where(swap_needed, val_j1, val_j)
                        new_val_j1 = nl.where(swap_needed, val_j, val_j1)
                        
                        # Store back
                        nl.store(result[row, j], value=new_val_j)
                        nl.store(result[row, j+1], value=new_val_j1)
    
    else:  # ndim > 2
        # For higher dimensions, we need to iterate over all but the dim to sort
        # This is a simplified implementation for common cases
        
        if dim == ndim - 1:  # Sort along the last dimension
            # Calculate the number of slices to process
            slice_size = 1
            for i in range(ndim - 1):
                slice_size *= shape[i]
            
            # Process each slice
            for slice_idx in nl.affine_range(slice_size):
                # Calculate multi-dimensional indices for this slice
                indices = []
                remaining = slice_idx
                for i in range(ndim - 1):
                    dim_size = shape[i]
                    idx = remaining // (slice_size // dim_size // (1 if i == 0 else shape[i-1]))
                    remaining = remaining % (slice_size // dim_size // (1 if i == 0 else shape[i-1]))
                    indices.append(idx)
                
                # Sort this slice (which is a 1D array)
                last_dim_size = shape[ndim-1]
                trip_count = math.ceil(last_dim_size / nl.tile_size.pmax)
                
                # Copy slice to result
                for p in nl.affine_range(trip_count):
                    idx_last = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
                    
                    # Create proper indexing for this slice
                    # For simplicity, we'll only handle up to 3D tensors in detail
                    if ndim == 3:
                        # Load input data
                        src_tile = nl.load(a_tensor[indices[0], indices[1], idx_last], 
                                          mask=(idx_last < last_dim_size))
                        
                        # Store to result
                        nl.store(result[indices[0], indices[1], idx_last], 
                                value=src_tile, mask=(idx_last < last_dim_size))
                    else:
                        # For higher dimensions, we'd need more complex indexing
                        # This is a placeholder for the general case
                        pass
                
                # Bubble sort this slice
                # For simplicity, we'll only handle up to 3D tensors in detail
                if ndim == 3:
                    for i in nl.affine_range(last_dim_size):
                        for j in nl.affine_range(last_dim_size - 1):
                            # Load adjacent elements
                            val_j = nl.load(result[indices[0], indices[1], j])
                            val_j1 = nl.load(result[indices[0], indices[1], j+1])
                            
                            # Compare and swap if necessary
                            swap_needed = nl.greater(val_j, val_j1)
                            
                            # Prepare values after potential swap
                            new_val_j = nl.where(swap_needed, val_j1, val_j)
                            new_val_j1 = nl.where(swap_needed, val_j, val_j1)
                            
                            # Store back
                            nl.store(result[indices[0], indices[1], j], value=new_val_j)
                            nl.store(result[indices[0], indices[1], j+1], value=new_val_j1)
        
        # For sorting along other dimensions in higher-dimensional tensors,
        # a more complex implementation would be needed
    
    return result