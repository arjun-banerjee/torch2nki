from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Handle negative dimensions
    if dim < 0:
        dim = len(a_tensor.shape) + dim
    
    # Get shape of the input tensor
    shape = a_tensor.shape
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First, copy the input tensor to result
    # We need to process the tensor in tiles due to hardware limitations
    if len(shape) == 1:
        # Handle 1D tensor case
        size = shape[0]
        trip_count = math.ceil(size / nl.tile_size.pmax)
        
        # Copy input to result
        for p in nl.affine_range(trip_count):
            # Generate tensor indices for the current tile
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
            
            # Load input data from external memory to on-chip memory
            in_tile = nl.load(a_tensor[i_p], mask=(i_p < size))
            
            # Store to result
            nl.store(result[i_p], value=in_tile, mask=(i_p < size))
        
        # Bubble sort algorithm
        for i in nl.affine_range(size):
            for j in nl.affine_range(size - 1):
                for p in nl.affine_range(trip_count):
                    # Generate tensor indices for the current tile
                    i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
                    
                    # Create mask for valid indices
                    j_mask = (i_p < size - 1) & (i_p == j)
                    
                    # Skip this tile if none of the indices match j
                    if j < p * nl.tile_size.pmax or j >= (p + 1) * nl.tile_size.pmax:
                        continue
                    
                    # Get current and next indices
                    j_idx = nl.full((nl.tile_size.pmax, 1), j, dtype=nl.int32)
                    j_next_idx = nl.full((nl.tile_size.pmax, 1), j+1, dtype=nl.int32)
                    
                    # Load elements to compare
                    elem1 = nl.load(result[j_idx], mask=j_mask)
                    elem2 = nl.load(result[j_next_idx], mask=j_mask)
                    
                    # Compare and swap if needed
                    swap_needed = nl.greater(elem1, elem2)
                    new_elem1 = nl.where(swap_needed, elem2, elem1, mask=j_mask)
                    new_elem2 = nl.where(swap_needed, elem1, elem2, mask=j_mask)
                    
                    # Store back the results
                    nl.store(result[j_idx], new_elem1, mask=j_mask)
                    nl.store(result[j_next_idx], new_elem2, mask=j_mask)
    
    else:
        # Handle multi-dimensional tensors
        # Reshape the tensor to handle sorting along the specified dimension
        
        # Calculate sizes for reshaping
        size_before_dim = 1
        for i in range(dim):
            size_before_dim *= shape[i]
            
        size_of_dim = shape[dim]
        
        size_after_dim = 1
        for i in range(dim + 1, len(shape)):
            size_after_dim *= shape[i]
        
        # Process in tiles
        # For each slice before the dimension to sort
        for b in nl.affine_range(size_before_dim):
            # For each slice after the dimension to sort
            for a in nl.affine_range(size_after_dim):
                # Copy the slice to result
                trip_count = math.ceil(size_of_dim / nl.tile_size.pmax)
                
                # First copy the input to result
                for p in nl.affine_range(trip_count):
                    # Generate tensor indices for the current tile
                    i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
                    
                    # Calculate actual indices based on the dimension
                    idx = []
                    idx_pos = 0
                    for d in range(len(shape)):
                        if d < dim:
                            # Dimensions before dim
                            b_idx = b
                            for d_before in range(dim-1, d, -1):
                                b_idx = b_idx // shape[d_before]
                            b_idx = b_idx % shape[d]
                            idx.append(b_idx)
                        elif d == dim:
                            # The dimension to sort
                            idx.append(i_p)
                            idx_pos = len(idx) - 1
                        else:
                            # Dimensions after dim
                            a_idx = a
                            for d_after in range(len(shape)-1, d, -1):
                                a_idx = a_idx // shape[d_after]
                            a_idx = a_idx % shape[d]
                            idx.append(a_idx)
                    
                    # Load input data from external memory to on-chip memory
                    if len(idx) == 2:
                        in_tile = nl.load(a_tensor[idx[0], idx[1]], mask=(idx[idx_pos] < size_of_dim))
                        nl.store(result[idx[0], idx[1]], value=in_tile, mask=(idx[idx_pos] < size_of_dim))
                    elif len(idx) == 3:
                        in_tile = nl.load(a_tensor[idx[0], idx[1], idx[2]], mask=(idx[idx_pos] < size_of_dim))
                        nl.store(result[idx[0], idx[1], idx[2]], value=in_tile, mask=(idx[idx_pos] < size_of_dim))
                
                # Bubble sort algorithm for this slice
                for i in nl.affine_range(size_of_dim):
                    for j in nl.affine_range(size_of_dim - 1):
                        for p in nl.affine_range(trip_count):
                            # Generate tensor indices for the current tile
                            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
                            
                            # Create mask for valid indices
                            j_mask = (i_p < size_of_dim - 1) & (i_p == j)
                            
                            # Skip this tile if none of the indices match j
                            if j < p * nl.tile_size.pmax or j >= (p + 1) * nl.tile_size.pmax:
                                continue
                            
                            # Calculate indices for current and next elements
                            idx = []
                            idx_next = []
                            for d in range(len(shape)):
                                if d < dim:
                                    # Dimensions before dim
                                    b_idx = b
                                    for d_before in range(dim-1, d, -1):
                                        b_idx = b_idx // shape[d_before]
                                    b_idx = b_idx % shape[d]
                                    idx.append(b_idx)
                                    idx_next.append(b_idx)
                                elif d == dim:
                                    # The dimension to sort
                                    idx.append(j)
                                    idx_next.append(j+1)
                                else:
                                    # Dimensions after dim
                                    a_idx = a
                                    for d_after in range(len(shape)-1, d, -1):
                                        a_idx = a_idx // shape[d_after]
                                    a_idx = a_idx % shape[d]
                                    idx.append(a_idx)
                                    idx_next.append(a_idx)
                            
                            # Load elements to compare
                            if len(idx) == 2:
                                elem1 = nl.load(result[idx[0], idx[1]], mask=j_mask)
                                elem2 = nl.load(result[idx_next[0], idx_next[1]], mask=j_mask)
                                
                                # Compare and swap if needed
                                swap_needed = nl.greater(elem1, elem2)
                                new_elem1 = nl.where(swap_needed, elem2, elem1, mask=j_mask)
                                new_elem2 = nl.where(swap_needed, elem1, elem2, mask=j_mask)
                                
                                # Store back the results
                                nl.store(result[idx[0], idx[1]], new_elem1, mask=j_mask)
                                nl.store(result[idx_next[0], idx_next[1]], new_elem2, mask=j_mask)
                            elif len(idx) == 3:
                                elem1 = nl.load(result[idx[0], idx[1], idx[2]], mask=j_mask)
                                elem2 = nl.load(result[idx_next[0], idx_next[1], idx_next[2]], mask=j_mask)
                                
                                # Compare and swap if needed
                                swap_needed = nl.greater(elem1, elem2)
                                new_elem1 = nl.where(swap_needed, elem2, elem1, mask=j_mask)
                                new_elem2 = nl.where(swap_needed, elem1, elem2, mask=j_mask)
                                
                                # Store back the results
                                nl.store(result[idx[0], idx[1], idx[2]], new_elem1, mask=j_mask)
                                nl.store(result[idx_next[0], idx_next[1], idx_next[2]], new_elem2, mask=j_mask)
    
    return result