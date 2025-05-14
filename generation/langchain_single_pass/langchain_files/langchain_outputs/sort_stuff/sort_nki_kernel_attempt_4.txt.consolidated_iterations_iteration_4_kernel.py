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
    
    # Handle 1D tensor case
    if ndim == 1:
        # Copy input to result
        for p in nl.affine_range(math.ceil(shape[0]/nl.tile_size.pmax)):
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            input_tile = nl.load(a_tensor[i_p], mask=(i_p < shape[0]))
            nl.store(result[i_p], value=input_tile, mask=(i_p < shape[0]))
        
        # Sort the 1D tensor using bubble sort
        n = shape[0]
        for i in range(n):
            for j in range(0, n-i-1):
                # Load elements to compare
                j_idx = nl.full((), j, dtype=nl.int32)
                j_plus_1_idx = nl.full((), j+1, dtype=nl.int32)
                
                a = nl.load(result[j_idx])
                b = nl.load(result[j_plus_1_idx])
                
                # If a > b, swap them
                is_greater = nl.greater(a, b)
                if is_greater[()]:
                    nl.store(result[j_idx], b)
                    nl.store(result[j_plus_1_idx], a)
    
    else:
        # For multi-dimensional tensors, we need to sort along the specified dimension
        
        # Calculate sizes for reshaping
        sort_dim_size = shape[dim]
        
        # For each slice along dimensions other than the sort dimension
        # we need to create an index tuple
        
        # First, copy input to result
        # We'll use a simple loop to copy all elements
        flat_size = 1
        for i in range(ndim):
            flat_size *= shape[i]
            
        for p in nl.affine_range(math.ceil(flat_size/nl.tile_size.pmax)):
            base_idx = p * nl.tile_size.pmax
            indices = base_idx + nl.arange(nl.tile_size.pmax)
            
            # Convert flat indices to multi-dimensional indices
            multi_indices = []
            temp_indices = indices.copy()
            for i in range(ndim-1, -1, -1):
                dim_size = shape[i]
                dim_indices = temp_indices % dim_size
                temp_indices = temp_indices // dim_size
                multi_indices.insert(0, dim_indices)
            
            # Load and store data
            input_data = nl.load(a_tensor[tuple(multi_indices)], mask=(indices < flat_size))
            nl.store(result[tuple(multi_indices)], value=input_data, mask=(indices < flat_size))
        
        # Now sort each slice along the sort dimension
        # Iterate through all possible index combinations except for the sort dimension
        outer_size = 1
        for i in range(ndim):
            if i != dim:
                outer_size *= shape[i]
                
        for p in range(outer_size):
            # Convert flat index p to multi-dimensional indices excluding sort dimension
            indices = []
            temp_p = p
            for i in range(ndim-1, -1, -1):
                if i != dim:
                    dim_size = shape[i]
                    indices.insert(0, temp_p % dim_size)
                    temp_p = temp_p // dim_size
                else:
                    indices.insert(0, 0)  # Placeholder for sort dimension
            
            # Sort this slice using bubble sort
            for i in range(sort_dim_size):
                for j in range(0, sort_dim_size-i-1):
                    # Create indices for elements to compare
                    indices_j = indices.copy()
                    indices_j[dim] = j
                    
                    indices_j_plus_1 = indices.copy()
                    indices_j_plus_1[dim] = j+1
                    
                    # Load elements
                    a = nl.load(result[tuple(indices_j)])
                    b = nl.load(result[tuple(indices_j_plus_1)])
                    
                    # If a > b, swap them
                    is_greater = nl.greater(a, b)
                    if is_greater[()]:
                        nl.store(result[tuple(indices_j)], b)
                        nl.store(result[tuple(indices_j_plus_1)], a)
    
    return result