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
    
    # Copy input to result first
    if ndim == 1:
        # Simple 1D case
        size = shape[0]
        trip_count = math.ceil(size / nl.tile_size.pmax)
        
        for p in nl.affine_range(trip_count):
            # Generate indices for current tile
            idx = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            
            # Load and store data
            data_tile = nl.load(a_tensor[idx], mask=(idx < size))
            nl.store(result[idx], value=data_tile, mask=(idx < size))
            
        # Bubble sort on result
        for i in range(size):
            for j in range(size - i - 1):
                # Create indices for the current pair
                idx_j = nl.full((), j, dtype=nl.int32)
                idx_j_plus_1 = nl.full((), j+1, dtype=nl.int32)
                
                # Load values
                val_j = nl.load(result[idx_j])
                val_j_plus_1 = nl.load(result[idx_j_plus_1])
                
                # Compare and swap if needed
                is_greater = nl.greater(val_j, val_j_plus_1)
                
                # Create temporary buffers for swapped values
                temp_j = nl.zeros((), dtype=a_tensor.dtype, buffer=nl.sbuf)
                temp_j_plus_1 = nl.zeros((), dtype=a_tensor.dtype, buffer=nl.sbuf)
                
                # Conditional swap
                temp_j = nl.where(is_greater, val_j_plus_1, val_j)
                temp_j_plus_1 = nl.where(is_greater, val_j, val_j_plus_1)
                
                # Store back to result
                nl.store(result[idx_j], value=temp_j)
                nl.store(result[idx_j_plus_1], value=temp_j_plus_1)
                
    elif ndim == 2:
        # For 2D tensor
        rows, cols = shape
        trip_count_p = math.ceil(rows / nl.tile_size.pmax)
        
        # First copy the input to result
        for p in nl.affine_range(trip_count_p):
            # Generate indices for current tile
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
            i_f = nl.arange(cols)[None, :]
            
            # Load and store data
            data_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < rows))
            nl.store(result[i_p, i_f], value=data_tile, mask=(i_p < rows))
        
        # Sort each row or column based on dim
        if dim == 0:  # Sort along rows
            for j in range(cols):
                for i in range(rows):
                    for k in range(rows - i - 1):
                        # Create indices for the current pair
                        idx_k = nl.full((), k, dtype=nl.int32)
                        idx_k_plus_1 = nl.full((), k+1, dtype=nl.int32)
                        idx_j = nl.full((), j, dtype=nl.int32)
                        
                        # Load values
                        val_k = nl.load(result[idx_k, idx_j])
                        val_k_plus_1 = nl.load(result[idx_k_plus_1, idx_j])
                        
                        # Compare and swap if needed
                        is_greater = nl.greater(val_k, val_k_plus_1)
                        
                        # Create temporary buffers for swapped values
                        temp_k = nl.zeros((), dtype=a_tensor.dtype, buffer=nl.sbuf)
                        temp_k_plus_1 = nl.zeros((), dtype=a_tensor.dtype, buffer=nl.sbuf)
                        
                        # Conditional swap
                        temp_k = nl.where(is_greater, val_k_plus_1, val_k)
                        temp_k_plus_1 = nl.where(is_greater, val_k, val_k_plus_1)
                        
                        # Store back to result
                        nl.store(result[idx_k, idx_j], value=temp_k)
                        nl.store(result[idx_k_plus_1, idx_j], value=temp_k_plus_1)
        else:  # Sort along columns (dim=1)
            for i in range(rows):
                for j in range(cols):
                    for k in range(cols - j - 1):
                        # Create indices for the current pair
                        idx_i = nl.full((), i, dtype=nl.int32)
                        idx_k = nl.full((), k, dtype=nl.int32)
                        idx_k_plus_1 = nl.full((), k+1, dtype=nl.int32)
                        
                        # Load values
                        val_k = nl.load(result[idx_i, idx_k])
                        val_k_plus_1 = nl.load(result[idx_i, idx_k_plus_1])
                        
                        # Compare and swap if needed
                        is_greater = nl.greater(val_k, val_k_plus_1)
                        
                        # Create temporary buffers for swapped values
                        temp_k = nl.zeros((), dtype=a_tensor.dtype, buffer=nl.sbuf)
                        temp_k_plus_1 = nl.zeros((), dtype=a_tensor.dtype, buffer=nl.sbuf)
                        
                        # Conditional swap
                        temp_k = nl.where(is_greater, val_k_plus_1, val_k)
                        temp_k_plus_1 = nl.where(is_greater, val_k, val_k_plus_1)
                        
                        # Store back to result
                        nl.store(result[idx_i, idx_k], value=temp_k)
                        nl.store(result[idx_i, idx_k_plus_1], value=temp_k_plus_1)
    else:
        # For higher dimension tensors, we implement a generalized approach
        # First copy the input to result with tiling
        flat_size = 1
        for s in shape:
            flat_size *= s
        
        # Calculate the number of tiles needed for copying
        trip_count = math.ceil(flat_size / nl.tile_size.pmax)
        
        # Copy input to result first
        for p in nl.affine_range(trip_count):
            idx = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            flat_idx = idx % flat_size
            
            # Compute multi-dimensional indices
            multi_idx = []
            remaining = flat_idx
            for i in range(ndim-1, -1, -1):
                divisor = 1
                for j in range(i+1, ndim):
                    divisor *= shape[j]
                idx_i = remaining // divisor
                remaining = remaining % divisor
                multi_idx.append(idx_i)
            
            # Load and store with proper indexing
            # For simplicity in higher dimensions, we'll use a less efficient approach
            # that processes one element at a time
            for i in range(min(nl.tile_size.pmax, flat_size - p * nl.tile_size.pmax)):
                elem_idx = p * nl.tile_size.pmax + i
                if elem_idx < flat_size:
                    # Compute multi-dimensional indices for this element
                    indices = []
                    remaining = elem_idx
                    for d in range(ndim-1, -1, -1):
                        divisor = 1
                        for j in range(d+1, ndim):
                            divisor *= shape[j]
                        idx_d = remaining // divisor
                        remaining = remaining % divisor
                        indices.insert(0, idx_d)
                    
                    # Create index arrays for this element
                    idx_arrays = []
                    for d in range(ndim):
                        idx_arrays.append(nl.full((), indices[d], dtype=nl.int32))
                    
                    # Load from input and store to result
                    if ndim == 3:
                        val = nl.load(a_tensor[idx_arrays[0], idx_arrays[1], idx_arrays[2]])
                        nl.store(result[idx_arrays[0], idx_arrays[1], idx_arrays[2]], value=val)
                    else:  # Handle other dimensions as needed
                        # This is a simplified approach - in practice, extend this for more dimensions
                        pass
        
    return result