from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get tensor shape and handle negative dimension
    shape = a_tensor.shape
    ndim = len(shape)
    
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First copy input tensor to result tensor
    if ndim == 1:
        # For 1D tensor, copy the whole tensor directly
        size = shape[0]
        max_tile_size = nl.tile_size.pmax
        trip_count = math.ceil(size / max_tile_size)
        
        for i in nl.affine_range(trip_count):
            # Generate tensor indices for the current tile
            indices = i * max_tile_size + nl.arange(max_tile_size)
            
            # Load input data with masking to handle boundary
            data = nl.load(a_tensor[indices], mask=(indices < size))
            
            # Store to result
            nl.store(result[indices], data, mask=(indices < size))
    else:
        # For multi-dimensional tensor
        # Reshape the tensor as a collection of vectors along the sorting dimension
        # and copy the data
        
        # Calculate the number of vectors to sort
        num_vectors = 1
        for i in range(ndim):
            if i != dim:
                num_vectors *= shape[i]
        
        vec_size = shape[dim]
        max_tile_size = nl.tile_size.pmax
        
        # Calculate trip counts for tiling
        vec_trip_count = math.ceil(num_vectors / max_tile_size)
        dim_trip_count = math.ceil(vec_size / max_tile_size)
        
        # Copy the tensor to result
        for vec_idx in nl.affine_range(vec_trip_count):
            vec_offset = vec_idx * max_tile_size
            vec_indices = vec_offset + nl.arange(max_tile_size)
            
            for dim_idx in nl.affine_range(dim_trip_count):
                dim_offset = dim_idx * max_tile_size
                dim_indices = dim_offset + nl.arange(max_tile_size)
                
                # Create index tuple based on dimensions
                if dim == 0:
                    # dim is the first dimension
                    src_data = nl.load(a_tensor[dim_indices[:, None], vec_indices[None, :]], 
                                      mask=((dim_indices[:, None] < vec_size) & 
                                            (vec_indices[None, :] < num_vectors)))
                    
                    nl.store(result[dim_indices[:, None], vec_indices[None, :]], src_data,
                            mask=((dim_indices[:, None] < vec_size) & 
                                  (vec_indices[None, :] < num_vectors)))
                else:
                    # dim is not the first dimension
                    src_data = nl.load(a_tensor[vec_indices[:, None], dim_indices[None, :]], 
                                      mask=((vec_indices[:, None] < num_vectors) & 
                                            (dim_indices[None, :] < vec_size)))
                    
                    nl.store(result[vec_indices[:, None], dim_indices[None, :]], src_data,
                            mask=((vec_indices[:, None] < num_vectors) & 
                                  (dim_indices[None, :] < vec_size)))
    
    # Now sort each vector along the specified dimension
    if ndim == 1:
        # For 1D tensor, sort directly
        size = shape[0]
        
        # Bubble sort implementation
        for i in range(size):
            for j in range(0, size - i - 1):
                # Load adjacent elements
                j_idx = nl.arange(1) + j
                j1_idx = nl.arange(1) + j + 1
                
                # Use masking to handle boundary conditions
                curr = nl.load(result[j_idx], mask=(j_idx < size))
                next_val = nl.load(result[j1_idx], mask=(j1_idx < size))
                
                # Compare and swap if needed
                swap_needed = nl.greater(curr, next_val)
                
                # Create swapped values
                new_curr = nl.where(swap_needed, next_val, curr)
                new_next = nl.where(swap_needed, curr, next_val)
                
                # Store back
                nl.store(result[j_idx], new_curr, mask=(j_idx < size))
                nl.store(result[j1_idx], new_next, mask=(j1_idx < size))
    else:
        # For multi-dimensional tensor
        # Sort each vector along the specified dimension
        
        # Calculate the number of vectors to sort
        num_vectors = 1
        for i in range(ndim):
            if i != dim:
                num_vectors *= shape[i]
        
        vec_size = shape[dim]
        max_vec_tile = min(max_tile_size, num_vectors)
        
        # Process vectors in tiles
        for vec_offset in range(0, num_vectors, max_vec_tile):
            actual_vec_tile = min(max_vec_tile, num_vectors - vec_offset)
            
            # Bubble sort each vector
            for i in range(vec_size):
                for j in range(0, vec_size - i - 1):
                    # Create indices for current and next elements
                    j_idx = j
                    j1_idx = j + 1
                    
                    vec_indices = nl.arange(actual_vec_tile) + vec_offset
                    
                    if dim == 0:
                        # dim is the first dimension
                        curr = nl.load(result[j_idx, vec_indices])
                        next_val = nl.load(result[j1_idx, vec_indices])
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(curr, next_val)
                        
                        # Create swapped values
                        new_curr = nl.where(swap_needed, next_val, curr)
                        new_next = nl.where(swap_needed, curr, next_val)
                        
                        # Store back
                        nl.store(result[j_idx, vec_indices], new_curr)
                        nl.store(result[j1_idx, vec_indices], new_next)
                    else:
                        # dim is not the first dimension
                        curr = nl.load(result[vec_indices, j_idx])
                        next_val = nl.load(result[vec_indices, j1_idx])
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(curr, next_val)
                        
                        # Create swapped values
                        new_curr = nl.where(swap_needed, next_val, curr)
                        new_next = nl.where(swap_needed, curr, next_val)
                        
                        # Store back
                        nl.store(result[vec_indices, j_idx], new_curr)
                        nl.store(result[vec_indices, j1_idx], new_next)
    
    return result