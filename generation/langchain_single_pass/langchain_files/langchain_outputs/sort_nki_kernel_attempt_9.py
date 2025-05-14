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
        size = tensor_shape[0]
        
        # Copy input to result first
        i = nl.arange(size)
        nl.store(result[i], nl.load(a_tensor[i]))
        
        # Perform bubble sort
        for i in range(size):
            for j in range(0, size-i-1):
                j_idx = nl.arange(1)
                j_plus_1_idx = nl.arange(1)
                
                # Load current and next element
                curr_val = nl.load(result[j])
                next_val = nl.load(result[j+1])
                
                # Compare and swap if needed
                swap_needed = nl.greater(curr_val, next_val)
                
                # Store the smaller value at position j
                nl.store(result[j], nl.where(swap_needed, next_val, curr_val))
                
                # Store the larger value at position j+1
                nl.store(result[j+1], nl.where(swap_needed, curr_val, next_val))
                
        return result
    
    # Handle multi-dimensional tensor case
    else:
        # Get size of dimension to sort along
        dim_size = tensor_shape[dim]
        
        # Determine shape of the "slices" perpendicular to sort dimension
        prefix_shape = tensor_shape[:dim]
        suffix_shape = tensor_shape[dim+1:]
        
        # Calculate total number of slices
        total_slices = 1
        for d in range(ndim):
            if d != dim:
                total_slices *= tensor_shape[d]
        
        # Calculate max slice size that respects architecture limitations
        max_slice_size = min(nl.tile_size.pmax, total_slices)
        
        # Number of iterations needed to process all slices
        num_iterations = math.ceil(total_slices / max_slice_size)
        
        # First, copy input to result
        for i in range(ndim):
            indices = []
            for d in range(ndim):
                if d == i:
                    indices.append(nl.arange(tensor_shape[d]))
                else:
                    indices.append(nl.arange(1))
            
            idx_tuple = tuple(indices)
            nl.store(result[idx_tuple], nl.load(a_tensor[idx_tuple]))
        
        # Process slices in batches
        for slice_batch in range(num_iterations):
            batch_start = slice_batch * max_slice_size
            batch_end = min(batch_start + max_slice_size, total_slices)
            batch_size = batch_end - batch_start
            
            # Bubble sort each slice
            for i in range(dim_size):
                for j in range(0, dim_size-i-1):
                    # Create indices for the current batch of slices
                    for slice_idx in range(batch_start, batch_end):
                        # Convert flat slice_idx to multi-dimensional indices
                        multi_idx = []
                        temp_idx = slice_idx
                        
                        for d in range(ndim-1, -1, -1):
                            if d == dim:
                                continue
                            
                            if d > dim:
                                idx_d = temp_idx % tensor_shape[d]
                                temp_idx //= tensor_shape[d]
                            else:
                                idx_d = temp_idx % tensor_shape[d]
                                temp_idx //= tensor_shape[d]
                                
                            multi_idx.insert(0, idx_d)
                        
                        # Insert j and j+1 indices for the sort dimension
                        multi_idx.insert(dim, j)
                        curr_indices = tuple(multi_idx)
                        
                        multi_idx[dim] = j + 1
                        next_indices = tuple(multi_idx)
                        
                        # Load current and next elements
                        curr_val = nl.load(result[curr_indices])
                        next_val = nl.load(result[next_indices])
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(curr_val, next_val)
                        
                        # Store the smaller value at position j
                        nl.store(result[curr_indices], nl.where(swap_needed, next_val, curr_val))
                        
                        # Store the larger value at position j+1
                        nl.store(result[next_indices], nl.where(swap_needed, curr_val, next_val))
        
        return result