from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Handle negative dim index
    if dim < 0:
        dim = len(a_tensor.shape) + dim
    
    # Get shape information
    shape = a_tensor.shape
    ndim = len(shape)
    
    # Initialize result arrays for sorted values and indices
    values = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    indices = nl.ndarray(shape, dtype=nl.int32, buffer=nl.shared_hbm)
    
    # Special case for 1D tensor
    if ndim == 1:
        # Copy input data to values
        size = shape[0]
        max_tile_size = 128  # nl.tile_size.pmax
        
        # Process in tiles to respect hardware limitations
        for i in nl.affine_range(math.ceil(size / max_tile_size)):
            # Calculate start and end indices for this tile
            start_idx = i * max_tile_size
            
            # Define ranges for loading/storing
            load_range = nl.arange(max_tile_size)
            
            # Load a tile of data
            data_tile = nl.load(a_tensor[start_idx + load_range], mask=(start_idx + load_range < size))
            
            # Initialize indices for this tile
            idx_tile = nl.zeros((max_tile_size,), dtype=nl.int32, buffer=nl.sbuf)
            for j in nl.affine_range(max_tile_size):
                idx_tile = nl.where(start_idx + j < size, start_idx + j, idx_tile, mask=(start_idx + j < size))
            
            # Sort using bubble sort
            tile_size = min(max_tile_size, size - start_idx)
            if tile_size > 0:
                for j in nl.affine_range(tile_size):
                    for k in nl.affine_range(tile_size - 1):
                        # Compare adjacent elements
                        cond = nl.greater(data_tile[k], data_tile[k+1])
                        
                        # Swap values if needed
                        temp_val = nl.where(cond, data_tile[k+1], data_tile[k])
                        data_tile = nl.where(cond, data_tile[k], temp_val, mask=(k+1 < tile_size))
                        
                        # Swap indices if needed
                        temp_idx = nl.where(cond, idx_tile[k+1], idx_tile[k])
                        idx_tile = nl.where(cond, idx_tile[k], temp_idx, mask=(k+1 < tile_size))
            
            # Store sorted values and indices
            nl.store(values[start_idx + load_range], value=data_tile, mask=(start_idx + load_range < size))
            nl.store(indices[start_idx + load_range], value=idx_tile, mask=(start_idx + load_range < size))
    
    else:
        # For multi-dimensional tensors, we need to sort along the specified dimension
        # First, copy the input tensor to values and initialize indices
        for i in nl.affine_range(math.prod(shape)):
            # Convert flat index to multi-dimensional indices
            multi_idx = []
            temp_idx = i
            for d in range(ndim-1, -1, -1):
                dim_size = shape[d]
                idx_d = temp_idx % dim_size
                temp_idx = temp_idx // dim_size
                multi_idx.insert(0, idx_d)
            
            # Load the value and store it in the values tensor
            val = nl.load(a_tensor[tuple(multi_idx)])
            nl.store(values[tuple(multi_idx)], value=val)
            
            # Initialize indices with their positions along the sort dimension
            idx_val = multi_idx[dim]
            nl.store(indices[tuple(multi_idx)], value=idx_val)
        
        # Sort along the specified dimension
        dim_size = shape[dim]
        
        # Process each "line" along the sort dimension
        for i in nl.affine_range(math.prod(shape) // dim_size):
            # Convert flat index to multi-dimensional indices (excluding sort dimension)
            multi_idx = []
            temp_idx = i
            for d in range(ndim):
                if d == dim:
                    multi_idx.append(0)  # Placeholder for sort dimension
                    continue
                
                idx_d = temp_idx % shape[d]
                temp_idx = temp_idx // shape[d]
                multi_idx.append(idx_d)
            
            # Bubble sort along the specified dimension
            for j in nl.affine_range(dim_size):
                for k in nl.affine_range(dim_size - 1):
                    # Set indices for comparing adjacent elements
                    idx1 = list(multi_idx)
                    idx2 = list(multi_idx)
                    idx1[dim] = k
                    idx2[dim] = k + 1
                    
                    # Load values to compare
                    val1 = nl.load(values[tuple(idx1)])
                    val2 = nl.load(values[tuple(idx2)])
                    
                    # Load indices
                    idx_val1 = nl.load(indices[tuple(idx1)])
                    idx_val2 = nl.load(indices[tuple(idx2)])
                    
                    # Compare and swap if needed
                    if nl.greater(val1, val2):
                        # Swap values
                        nl.store(values[tuple(idx1)], value=val2)
                        nl.store(values[tuple(idx2)], value=val1)
                        
                        # Swap indices
                        nl.store(indices[tuple(idx1)], value=idx_val2)
                        nl.store(indices[tuple(idx2)], value=idx_val1)
    
    return values, indices