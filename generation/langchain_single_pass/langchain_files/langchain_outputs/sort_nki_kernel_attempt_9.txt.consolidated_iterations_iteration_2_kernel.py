from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get tensor shape and handle negative dimensions
    tensor_shape = a_tensor.shape
    ndim = len(tensor_shape)
    
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First copy the input to result
    if ndim == 1:
        # Special case for 1D tensors
        sz = tensor_shape[0]
        trip_count = math.ceil(sz / nl.tile_size.pmax)
        
        for p in nl.affine_range(trip_count):
            # Generate indices for current tile
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            
            # Load data with masking to handle boundary
            in_tile = nl.load(a_tensor[i_p], mask=(i_p < sz))
            
            # Store to result
            nl.store(result[i_p], value=in_tile, mask=(i_p < sz))
            
        # Bubble sort the 1D array
        for i in nl.affine_range(sz):
            for j in nl.affine_range(sz - 1):
                # Load pairs of elements
                idx1 = j
                idx2 = j + 1
                
                val1 = nl.load(result[idx1])
                val2 = nl.load(result[idx2])
                
                # Compare and swap if needed
                is_greater = nl.greater(val1, val2)
                
                # We can't use is_greater directly in an if statement, so use conditional operations
                if_true = val2  # swap: val1 becomes val2
                if_false = val1  # no swap: val1 stays val1
                new_val1 = nl.where(is_greater, if_true, if_false)
                
                if_true = val1  # swap: val2 becomes val1
                if_false = val2  # no swap: val2 stays val2
                new_val2 = nl.where(is_greater, if_true, if_false)
                
                # Store back the potentially swapped values
                nl.store(result[idx1], value=new_val1)
                nl.store(result[idx2], value=new_val2)
    
    else:
        # For multi-dimensional tensors
        # Compute the size of each dimension
        dims = list(tensor_shape)
        
        # Compute the stride for the sort dimension
        sort_dim_size = dims[dim]
        
        # Reshape the problem into a 2D problem where one dimension is the sort dimension
        # and the other dimension contains all other dimensions flattened
        outer_size = 1
        for i in range(ndim):
            if i != dim:
                outer_size *= dims[i]
        
        # Process in tiles to respect hardware limitations
        outer_trip_count = math.ceil(outer_size / nl.tile_size.pmax)
        
        for p in nl.affine_range(outer_trip_count):
            outer_idx = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
            
            # Copy original data to result
            for q in nl.affine_range(sort_dim_size):
                # Construct indices for the current element
                indices = []
                flat_idx = outer_idx
                
                for d in range(ndim):
                    if d == dim:
                        indices.append(q)
                    else:
                        # Calculate proper index for this dimension from flat_idx
                        dim_size = 1
                        for d2 in range(d+1, ndim):
                            if d2 != dim:
                                dim_size *= dims[d2]
                        idx = (flat_idx // dim_size) % dims[d]
                        indices.append(idx)
                
                # Load data with masking
                if ndim == 2:
                    if dim == 0:
                        in_tile = nl.load(a_tensor[q, outer_idx], mask=(outer_idx < outer_size))
                        nl.store(result[q, outer_idx], value=in_tile, mask=(outer_idx < outer_size))
                    else:
                        in_tile = nl.load(a_tensor[outer_idx, q], mask=(outer_idx < outer_size))
                        nl.store(result[outer_idx, q], value=in_tile, mask=(outer_idx < outer_size))
            
            # Perform bubble sort on each slice along sort dimension
            for i in nl.affine_range(sort_dim_size):
                for j in nl.affine_range(sort_dim_size - 1):
                    # Load pairs of elements
                    if ndim == 2:
                        if dim == 0:
                            val1 = nl.load(result[j, outer_idx], mask=(outer_idx < outer_size))
                            val2 = nl.load(result[j+1, outer_idx], mask=(outer_idx < outer_size))
                            
                            # Compare and swap if needed
                            is_greater = nl.greater(val1, val2)
                            new_val1 = nl.where(is_greater, val2, val1)
                            new_val2 = nl.where(is_greater, val1, val2)
                            
                            # Store back
                            nl.store(result[j, outer_idx], value=new_val1, mask=(outer_idx < outer_size))
                            nl.store(result[j+1, outer_idx], value=new_val2, mask=(outer_idx < outer_size))
                        else:
                            val1 = nl.load(result[outer_idx, j], mask=(outer_idx < outer_size))
                            val2 = nl.load(result[outer_idx, j+1], mask=(outer_idx < outer_size))
                            
                            # Compare and swap if needed
                            is_greater = nl.greater(val1, val2)
                            new_val1 = nl.where(is_greater, val2, val1)
                            new_val2 = nl.where(is_greater, val1, val2)
                            
                            # Store back
                            nl.store(result[outer_idx, j], value=new_val1, mask=(outer_idx < outer_size))
                            nl.store(result[outer_idx, j+1], value=new_val2, mask=(outer_idx < outer_size))
    
    return result