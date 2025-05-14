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
    
    # Handle 1D tensor case
    if ndim == 1:
        # First copy the input to result
        size = tensor_shape[0]
        
        # Use tiling to handle large tensors
        tile_size = nl.tile_size.pmax
        trip_count = math.ceil(size / tile_size)
        
        # First copy input to result
        for p in nl.affine_range(trip_count):
            i_p = p * tile_size + nl.arange(tile_size)
            src_tile = nl.load(a_tensor[i_p], mask=(i_p < size))
            nl.store(result[i_p], src_tile, mask=(i_p < size))
        
        # Bubble sort implementation
        for i in range(size):
            for j in range(size - i - 1):
                # Load elements to compare
                j_idx = nl.arange(1)
                j_val = nl.load(result[j:j+1])
                j_next_val = nl.load(result[j+1:j+2])
                
                # Compare and swap if needed
                swap_needed = nl.greater(j_val, j_next_val)
                
                # Conditional swap
                if swap_needed.item():
                    nl.store(result[j:j+1], j_next_val)
                    nl.store(result[j+1:j+2], j_val)
        
        return result
    
    # Handle multi-dimensional tensor case
    else:
        # Determine the size of the dimension to sort along
        sort_dim_size = tensor_shape[dim]
        
        # Create a list of all other dimensions to iterate over
        other_dims = []
        total_other_elements = 1
        for i in range(ndim):
            if i != dim:
                other_dims.append(i)
                total_other_elements *= tensor_shape[i]
        
        # Copy input to result first
        for p in nl.affine_range(total_other_elements):
            # Calculate multi-dimensional indices for the current element
            indices = []
            temp_p = p
            for i in range(len(other_dims)-1, -1, -1):
                dim_idx = other_dims[i]
                dim_size = tensor_shape[dim_idx]
                idx = temp_p % dim_size
                temp_p = temp_p // dim_size
                indices.insert(0, idx)
            
            # Insert None at the sort dimension position
            indices.insert(dim, None)
            
            # Create index expressions for each dimension
            index_expr = []
            for i in range(ndim):
                if i == dim:
                    index_expr.append(nl.arange(sort_dim_size))
                else:
                    # Find the position of this dimension in other_dims
                    idx_pos = other_dims.index(i)
                    # Use the corresponding index from the indices list
                    index_expr.append(indices[i])
            
            # Load the slice, sort it, and store it back
            # Load the slice to sort
            slice_to_sort = nl.load(a_tensor[tuple(index_expr)])
            
            # Sort the slice using bubble sort
            sorted_slice = nl.zeros(sort_dim_size, dtype=a_tensor.dtype, buffer=nl.sbuf)
            # Copy slice_to_sort to sorted_slice
            sorted_slice = slice_to_sort
            
            # Bubble sort implementation
            for i in range(sort_dim_size):
                for j in range(sort_dim_size - i - 1):
                    j_idx = nl.arange(1)
                    j_val = sorted_slice[j:j+1]
                    j_next_val = sorted_slice[j+1:j+2]
                    
                    # Compare and swap if needed
                    swap_needed = nl.greater(j_val, j_next_val)
                    
                    # Conditional swap
                    if swap_needed.item():
                        temp = j_val
                        sorted_slice[j:j+1] = j_next_val
                        sorted_slice[j+1:j+2] = temp
            
            # Store the sorted slice back to result
            nl.store(result[tuple(index_expr)], sorted_slice)
        
        return result