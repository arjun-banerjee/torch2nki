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
        # Handle 1D case - sort the entire array
        size = shape[0]
        
        # Process in tiles if needed
        max_tile_size = 128
        num_tiles = math.ceil(size / max_tile_size)
        
        for tile_idx in range(num_tiles):
            start_idx = tile_idx * max_tile_size
            end_idx = min(start_idx + max_tile_size, size)
            tile_size = end_idx - start_idx
            
            # Create indices for loading and storing
            indices = nl.arange(tile_size)
            
            # Load input data
            input_data = nl.load(a_tensor[start_idx + indices], mask=(indices < tile_size))
            
            # Store into result
            nl.store(result[start_idx + indices], input_data, mask=(indices < tile_size))
        
        # Bubble sort on the entire array
        for i in range(size):
            for j in range(0, size - i - 1):
                # Load adjacent elements
                j_idx = nl.full((), j, dtype=nl.int32)
                j_next_idx = nl.full((), j + 1, dtype=nl.int32)
                
                elem1 = nl.load(result[j_idx])
                elem2 = nl.load(result[j_next_idx])
                
                # Compare and swap if needed
                is_greater = nl.greater(elem1, elem2)
                
                # If elem1 > elem2, swap them
                temp1 = nl.copy(elem1)
                temp2 = nl.copy(elem2)
                
                # Use conditional store based on comparison
                nl.store(result[j_idx], nl.multiply(is_greater, temp2) + nl.multiply(1 - is_greater, temp1))
                nl.store(result[j_next_idx], nl.multiply(is_greater, temp1) + nl.multiply(1 - is_greater, temp2))
    
    else:
        # For multi-dimensional tensors
        # Reshape the tensor to treat it as a 2D tensor with the sort dimension as the second dimension
        
        # Calculate sizes before and after the dimension to sort
        before_dims_size = 1
        for i in range(dim):
            before_dims_size *= shape[i]
        
        sort_dim_size = shape[dim]
        
        after_dims_size = 1
        for i in range(dim + 1, ndim):
            after_dims_size *= shape[i]
        
        # Copy input to result first
        max_tile_size = 128
        
        # Process before_dims in chunks
        for b_idx in range(0, before_dims_size, max_tile_size):
            b_end = min(b_idx + max_tile_size, before_dims_size)
            b_size = b_end - b_idx
            
            # Process sort_dim in chunks
            for s_idx in range(0, sort_dim_size, max_tile_size):
                s_end = min(s_idx + max_tile_size, sort_dim_size)
                s_size = s_end - s_idx
                
                # Process after_dims in chunks
                for a_idx in range(0, after_dims_size, max_tile_size):
                    a_end = min(a_idx + max_tile_size, after_dims_size)
                    a_size = a_end - a_idx
                    
                    # Create indices for this tile
                    b_indices = nl.arange(b_size)[:, None, None]
                    s_indices = nl.arange(s_size)[None, :, None]
                    a_indices = nl.arange(a_size)[None, None, :]
                    
                    # Calculate full indices for each dimension
                    full_indices = []
                    dim_idx = 0
                    
                    for i in range(ndim):
                        if i < dim:
                            # This is a "before" dimension
                            idx = (b_idx + b_indices) // (before_dims_size // shape[i]) % shape[i]
                            full_indices.append(idx)
                        elif i == dim:
                            # This is the sort dimension
                            full_indices.append(s_idx + s_indices)
                        else:
                            # This is an "after" dimension
                            idx = (a_idx + a_indices) // (after_dims_size // shape[i]) % shape[i]
                            full_indices.append(idx)
                    
                    # Load input data
                    input_data = nl.load(a_tensor[tuple(full_indices)])
                    
                    # Store into result
                    nl.store(result[tuple(full_indices)], input_data)
        
        # Sort each "row" along the sort dimension
        for b_idx in range(before_dims_size):
            for a_idx in range(after_dims_size):
                # Bubble sort along sort_dim
                for i in range(sort_dim_size):
                    for j in range(sort_dim_size - i - 1):
                        # Calculate full indices for the current elements
                        idx1_full = []
                        idx2_full = []
                        
                        for d in range(ndim):
                            if d < dim:
                                # Before dimensions
                                idx = (b_idx // (before_dims_size // shape[d])) % shape[d]
                                idx1_full.append(nl.full((), idx, dtype=nl.int32))
                                idx2_full.append(nl.full((), idx, dtype=nl.int32))
                            elif d == dim:
                                # Sort dimension
                                idx1_full.append(nl.full((), j, dtype=nl.int32))
                                idx2_full.append(nl.full((), j + 1, dtype=nl.int32))
                            else:
                                # After dimensions
                                idx = (a_idx // (after_dims_size // shape[d])) % shape[d]
                                idx1_full.append(nl.full((), idx, dtype=nl.int32))
                                idx2_full.append(nl.full((), idx, dtype=nl.int32))
                        
                        # Load adjacent elements
                        elem1 = nl.load(result[tuple(idx1_full)])
                        elem2 = nl.load(result[tuple(idx2_full)])
                        
                        # Compare and swap if needed
                        is_greater = nl.greater(elem1, elem2)
                        
                        # If elem1 > elem2, swap them
                        temp1 = nl.copy(elem1)
                        temp2 = nl.copy(elem2)
                        
                        # Use conditional store based on comparison
                        nl.store(result[tuple(idx1_full)], nl.multiply(is_greater, temp2) + nl.multiply(1 - is_greater, temp1))
                        nl.store(result[tuple(idx2_full)], nl.multiply(is_greater, temp1) + nl.multiply(1 - is_greater, temp2))
    
    return result