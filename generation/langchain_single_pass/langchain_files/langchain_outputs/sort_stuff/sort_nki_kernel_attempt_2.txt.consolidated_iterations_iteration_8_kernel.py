from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get shape information
    shape = a_tensor.shape
    ndim = len(shape)
    
    # Handle negative dimension indexing
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result with same shape and dtype as input
    result = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Handle 1D case separately
    if ndim == 1:
        # Copy the input tensor to result
        max_tile_size = nl.tile_size.pmax
        trip_count = math.ceil(shape[0] / max_tile_size)
        
        for i in nl.affine_range(trip_count):
            start_idx = i * max_tile_size
            idx = start_idx + nl.arange(max_tile_size)
            
            # Load data with proper masking for boundary
            data = nl.load(a_tensor[idx], mask=(idx < shape[0]))
            
            # Copy to result
            nl.store(result[idx], data, mask=(idx < shape[0]))
        
        # Sort the 1D array using bubble sort
        for i in nl.affine_range(shape[0]):
            for j in nl.affine_range(shape[0] - 1):
                # Load adjacent elements
                j_idx = nl.arange(1) + j
                j1_idx = j_idx + 1
                
                # Check if we're within bounds
                valid = (j1_idx < shape[0])
                
                # Load elements
                el1 = nl.load(result[j_idx], mask=valid)
                el2 = nl.load(result[j1_idx], mask=valid)
                
                # Compare and swap if needed
                swap_needed = nl.greater(el1, el2)
                
                # Only perform swap if needed and valid
                swap_mask = swap_needed & valid
                if swap_mask.any():
                    nl.store(result[j_idx], el2, mask=swap_mask)
                    nl.store(result[j1_idx], el1, mask=swap_mask)
    
    # For multi-dimensional tensors
    else:
        # Calculate sizes for slicing
        dim_size = shape[dim]
        
        # Create indices for all dimensions
        indices = []
        for i in range(ndim):
            if i == dim:
                # This is the dimension we sort along
                indices.append(slice(None))
            else:
                # For all other dimensions, use full range
                indices.append(slice(0, shape[i]))
        
        # First copy input to result
        for i in nl.affine_range(dim_size):
            # Update the index for the dimension we're sorting
            indices[dim] = i
            
            # Load from input and store to result
            val = nl.load(a_tensor[tuple(indices)])
            nl.store(result[tuple(indices)], val)
        
        # Now perform bubble sort along the specified dimension
        for i in nl.affine_range(dim_size):
            for j in nl.affine_range(dim_size - 1):
                # Get indices for adjacent elements
                indices1 = list(indices)
                indices2 = list(indices)
                indices1[dim] = j
                indices2[dim] = j + 1
                
                # Load adjacent elements along the sort dimension
                el1 = nl.load(result[tuple(indices1)])
                el2 = nl.load(result[tuple(indices2)])
                
                # Compare and swap if needed
                swap_needed = nl.greater(el1, el2)
                
                # Only perform swap if needed
                if swap_needed.any():
                    nl.store(result[tuple(indices1)], el2, mask=swap_needed)
                    nl.store(result[tuple(indices2)], el1, mask=swap_needed)
    
    return result