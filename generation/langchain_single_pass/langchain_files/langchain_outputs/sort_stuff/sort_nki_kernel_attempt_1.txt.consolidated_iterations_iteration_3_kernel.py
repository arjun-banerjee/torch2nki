from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Convert negative dim to positive
    if dim < 0:
        dim = len(a_tensor.shape) + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Handle tensor dimensionality
    if len(a_tensor.shape) == 1:
        # For 1D tensor, sort the entire array
        size = a_tensor.shape[0]
        # Copy input to result first
        for i in nl.affine_range(math.ceil(size / 128)):
            start_idx = i * 128
            length = min(128, size - start_idx)
            
            # Create indices for current tile
            indices = nl.arange(length)
            
            # Load current tile
            tile = nl.load(a_tensor[start_idx + indices], mask=(indices < length))
            
            # Store to result
            nl.store(result[start_idx + indices], value=tile, mask=(indices < length))
        
        # Bubble sort implementation - process in tiles to respect hardware limitations
        for i in nl.affine_range(size - 1):
            # For each bubble pass, we need to compare adjacent elements
            # Process in tiles to handle large arrays
            for j in nl.affine_range(math.ceil((size - i - 1) / 128)):
                start_idx = j * 128
                curr_length = min(128, size - i - 1 - start_idx)
                
                if curr_length <= 0:
                    continue
                    
                # Create indices for current tile
                indices = nl.arange(curr_length)
                
                # Load current and next elements
                curr_elements = nl.load(result[start_idx + indices], mask=(indices < curr_length))
                next_elements = nl.load(result[start_idx + indices + 1], mask=(indices < curr_length))
                
                # Compare and swap if needed
                swap_mask = nl.greater(curr_elements, next_elements)
                
                # Store swapped elements
                nl.store(result[start_idx + indices], 
                         value=nl.where(swap_mask, next_elements, curr_elements),
                         mask=(indices < curr_length))
                nl.store(result[start_idx + indices + 1],
                         value=nl.where(swap_mask, curr_elements, next_elements),
                         mask=(indices < curr_length))
    
    else:
        # For multi-dimensional tensor, sort along specified dimension
        # First copy input to result
        shape = a_tensor.shape
        
        # Calculate how many elements need to be processed
        total_elements = 1
        for i in range(len(shape)):
            total_elements *= shape[i]
            
        # Calculate the size of the dimension to sort
        sort_dim_size = shape[dim]
        
        # Calculate number of slices to process
        slices = total_elements // sort_dim_size
        
        # Copy input to result first
        for i in nl.affine_range(math.ceil(total_elements / 128)):
            start_idx = i * 128
            length = min(128, total_elements - start_idx)
            
            # Flatten tensor for copying
            flat_indices = nl.arange(length)
            
            # Calculate multi-dimensional indices
            multi_indices = []
            remaining_idx = start_idx + flat_indices
            
            # Process each dimension to get proper indices
            for d in range(len(shape)):
                # Calculate stride for this dimension
                stride = 1
                for d2 in range(d+1, len(shape)):
                    stride *= shape[d2]
                
                # Calculate index for this dimension
                idx = remaining_idx // stride
                remaining_idx = remaining_idx % stride
                multi_indices.append(idx)
            
            # Load data from flattened indices
            if len(shape) == 2:
                # Handle 2D case explicitly
                i0 = start_idx // shape[1] + flat_indices // shape[1]
                i1 = start_idx % shape[1] + flat_indices % shape[1]
                tile = nl.load(a_tensor[i0, i1], mask=(flat_indices < length))
                nl.store(result[i0, i1], value=tile, mask=(flat_indices < length))
            elif len(shape) == 3:
                # Handle 3D case explicitly
                i0 = start_idx // (shape[1] * shape[2]) + flat_indices // (shape[1] * shape[2])
                temp = start_idx % (shape[1] * shape[2]) + flat_indices % (shape[1] * shape[2])
                i1 = temp // shape[2]
                i2 = temp % shape[2]
                tile = nl.load(a_tensor[i0, i1, i2], mask=(flat_indices < length))
                nl.store(result[i0, i1, i2], value=tile, mask=(flat_indices < length))
            else:
                # For higher dimensions, process each slice separately
                for s in nl.affine_range(slices):
                    # Calculate indices for this slice
                    slice_indices = []
                    slice_idx = s
                    for d in range(len(shape)):
                        if d == dim:
                            # For sort dimension, we need all indices
                            slice_indices.append(nl.arange(sort_dim_size))
                        else:
                            # For other dimensions, we need specific index
                            stride = 1
                            for d2 in range(d+1, len(shape)):
                                if d2 != dim:
                                    stride *= shape[d2]
                            idx = slice_idx // stride
                            slice_idx = slice_idx % stride
                            slice_indices.append(idx)
                    
                    # Process this slice in smaller chunks if needed
                    for i in nl.affine_range(math.ceil(sort_dim_size / 128)):
                        start = i * 128
                        chunk_size = min(128, sort_dim_size - start)
                        
                        # Get data for this chunk
                        chunk_indices = nl.arange(chunk_size)
                        slice_indices[dim] = start + chunk_indices
                        
                        # Load data
                        chunk_data = nl.load(a_tensor[tuple(slice_indices)], mask=(chunk_indices < chunk_size))
                        
                        # Store to result
                        nl.store(result[tuple(slice_indices)], value=chunk_data, mask=(chunk_indices < chunk_size))
        
        # Now sort each slice along the specified dimension
        for s in nl.affine_range(slices):
            # Calculate indices for this slice
            slice_indices = []
            slice_idx = s
            for d in range(len(shape)):
                if d == dim:
                    # For sort dimension, we need all indices
                    slice_indices.append(nl.arange(sort_dim_size))
                else:
                    # For other dimensions, we need specific index
                    stride = 1
                    for d2 in range(d+1, len(shape)):
                        if d2 != dim:
                            stride *= shape[d2]
                    idx = slice_idx // stride
                    slice_idx = slice_idx % stride
                    slice_indices.append(idx)
            
            # Bubble sort this slice
            for i in nl.affine_range(sort_dim_size - 1):
                for j in nl.affine_range(sort_dim_size - i - 1):
                    # Get current and next elements
                    slice_indices[dim] = j
                    curr = nl.load(result[tuple(slice_indices)])
                    
                    slice_indices[dim] = j + 1
                    next_val = nl.load(result[tuple(slice_indices)])
                    
                    # Compare and swap if needed
                    if nl.greater(curr, next_val):
                        # Swap elements
                        slice_indices[dim] = j
                        nl.store(result[tuple(slice_indices)], value=next_val)
                        
                        slice_indices[dim] = j + 1
                        nl.store(result[tuple(slice_indices)], value=curr)
    
    return result