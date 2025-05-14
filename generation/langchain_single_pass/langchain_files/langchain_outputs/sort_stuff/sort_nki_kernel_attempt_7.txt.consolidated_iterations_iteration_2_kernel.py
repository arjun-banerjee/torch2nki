from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get the shape of the input tensor
    shape = a_tensor.shape
    ndim = len(shape)
    
    # Handle negative dimension index
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First, copy the input tensor to the result tensor
    # We'll work on slices of this tensor
    if ndim == 1:
        # Special case for 1D tensor
        size = shape[0]
        trip_count = math.ceil(size / nl.tile_size.pmax)
        
        for p in nl.affine_range(trip_count):
            # Generate indices for the current tile
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            
            # Load input data
            in_tile = nl.load(a_tensor[i_p], mask=(i_p < size))
            
            # Store to result
            nl.store(result[i_p], value=in_tile, mask=(i_p < size))
    else:
        # Handle multi-dimensional tensors
        # Calculate the number of slices to process
        slice_size = shape[dim]
        
        # Compute the total number of slices across all other dimensions
        num_slices = 1
        for i in range(ndim):
            if i != dim:
                num_slices *= shape[i]
        
        # Calculate tiles needed for slices
        trip_count = math.ceil(num_slices / nl.tile_size.pmax)
        
        for p in nl.affine_range(trip_count):
            # Generate slice indices
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
            
            # Convert flat indices to multi-dimensional indices
            multi_indices = []
            remaining_slices = i_p.copy()
            
            # First copy the data
            for d in range(ndim):
                if d != dim:
                    # Process each non-sort dimension
                    dim_size = shape[d]
                    # We'll load the entire slice along the sort dimension
                    indices_d = remaining_slices % dim_size
                    remaining_slices = remaining_slices // dim_size
                    
                    # Create indices for this dimension
                    if d < dim:
                        multi_indices.append(indices_d)
                    else:
                        multi_indices.append(indices_d)
            
            # Load and store the data
            for i_sort in nl.affine_range(slice_size):
                # Insert sort dimension indices
                sort_indices = []
                for d in range(ndim):
                    if d == dim:
                        sort_indices.append(i_sort)
                    else:
                        idx = multi_indices[0 if d < dim else d-1]
                        sort_indices.append(idx)
                
                # Load the slice data
                in_slice = nl.load(a_tensor[tuple(sort_indices)], mask=(i_p < num_slices))
                
                # Store to result
                nl.store(result[tuple(sort_indices)], value=in_slice, mask=(i_p < num_slices))
    
    # Now perform the bubble sort on each slice along the specified dimension
    if ndim == 1:
        # For 1D tensor, we sort the entire tensor
        size = shape[0]
        
        # Bubble sort implementation
        for i in nl.affine_range(size):
            for j in nl.affine_range(size - 1):
                # Load adjacent elements
                a = nl.load(result[j])
                b = nl.load(result[j + 1])
                
                # Compare and swap if needed
                is_greater = nl.greater(a, b)
                
                # Conditional swap
                temp_a = nl.where(is_greater, b, a)
                temp_b = nl.where(is_greater, a, b)
                
                # Store back
                nl.store(result[j], value=temp_a)
                nl.store(result[j + 1], value=temp_b)
    else:
        # For multi-dimensional tensors, sort each slice along the specified dimension
        slice_size = shape[dim]
        
        # Compute the total number of slices across all other dimensions
        num_slices = 1
        for i in range(ndim):
            if i != dim:
                num_slices *= shape[i]
        
        # Calculate tiles needed for slices
        trip_count = math.ceil(num_slices / nl.tile_size.pmax)
        
        for p in nl.affine_range(trip_count):
            # Generate slice indices
            i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
            
            # Process each slice
            for slice_idx in nl.affine_range(min(nl.tile_size.pmax, num_slices - p * nl.tile_size.pmax)):
                flat_idx = p * nl.tile_size.pmax + slice_idx
                
                # Convert flat index to multi-dimensional indices excluding the sort dimension
                remaining = flat_idx
                indices = [0] * ndim
                for d in range(ndim-1, -1, -1):
                    if d != dim:
                        indices[d] = remaining % shape[d]
                        remaining = remaining // shape[d]
                
                # Bubble sort this slice
                for i in nl.affine_range(slice_size):
                    for j in nl.affine_range(slice_size - 1):
                        # Create indices for adjacent elements
                        idx_a = indices.copy()
                        idx_b = indices.copy()
                        idx_a[dim] = j
                        idx_b[dim] = j + 1
                        
                        # Load adjacent elements
                        a = nl.load(result[tuple(idx_a)])
                        b = nl.load(result[tuple(idx_b)])
                        
                        # Compare and swap if needed
                        is_greater = nl.greater(a, b)
                        
                        # Conditional swap
                        temp_a = nl.where(is_greater, b, a)
                        temp_b = nl.where(is_greater, a, b)
                        
                        # Store back
                        nl.store(result[tuple(idx_a)], value=temp_a)
                        nl.store(result[tuple(idx_b)], value=temp_b)
    
    return result