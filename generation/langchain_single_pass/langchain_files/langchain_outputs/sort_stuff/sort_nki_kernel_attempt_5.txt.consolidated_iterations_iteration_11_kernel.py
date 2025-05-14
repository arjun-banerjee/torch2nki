from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Get shape information
    shape = a_tensor.shape
    ndim = len(shape)
    
    # Handle negative dimension
    if dim < 0:
        dim = ndim + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Handle different dimensions
    if ndim == 1:
        # For 1D tensor, simply sort the entire array
        size = shape[0]
        tile_size = min(128, size)  # Respect hardware limit
        
        # First copy input to result
        for i in nl.affine_range(math.ceil(size / tile_size)):
            start_idx = i * tile_size
            # Create index arrays for loading/storing
            indices = start_idx + nl.arange(tile_size)
            # Load input data
            input_tile = nl.load(a_tensor[indices], mask=(indices < size))
            # Store to result
            nl.store(result[indices], input_tile, mask=(indices < size))
        
        # Now perform bubble sort on the entire array
        for i in nl.affine_range(size):
            for j in nl.affine_range(size - 1):
                # Load adjacent elements for comparison
                idx1 = j
                idx2 = j + 1
                
                # Load values
                val1 = nl.load(result[idx1])
                val2 = nl.load(result[idx2])
                
                # Compare and swap if needed
                swap_needed = nl.greater(val1, val2)
                
                # Perform conditional swap
                if_swapped_val1 = val2
                if_swapped_val2 = val1
                
                new_val1 = nl.where(swap_needed, if_swapped_val1, val1)
                new_val2 = nl.where(swap_needed, if_swapped_val2, val2)
                
                # Store back
                nl.store(result[idx1], new_val1)
                nl.store(result[idx2], new_val2)
    else:
        # For multi-dimensional tensors, sort along the specified dimension
        # Calculate sizes for slicing
        dim_size = shape[dim]
        
        # Calculate the product of dimensions before the sort dimension
        outer_size = 1
        for d in range(dim):
            outer_size *= shape[d]
        
        # Calculate the product of dimensions after the sort dimension
        inner_size = 1
        for d in range(dim + 1, ndim):
            inner_size *= shape[d]
        
        # First copy input to result
        for o in nl.affine_range(outer_size):
            for i in nl.affine_range(inner_size):
                for d in nl.affine_range(dim_size):
                    # Calculate flat index based on dimensions
                    flat_idx = (o * dim_size * inner_size) + (d * inner_size) + i
                    
                    # Load from input tensor
                    val = nl.load(a_tensor.flatten()[flat_idx])
                    
                    # Store to result tensor
                    nl.store(result.flatten()[flat_idx], val)
        
        # Now perform bubble sort on each slice along the specified dimension
        for o in nl.affine_range(outer_size):
            for i in nl.affine_range(inner_size):
                # Sort this specific slice
                for pass_idx in nl.affine_range(dim_size):
                    for d in nl.affine_range(dim_size - 1):
                        # Calculate flat indices for adjacent elements
                        flat_idx1 = (o * dim_size * inner_size) + (d * inner_size) + i
                        flat_idx2 = (o * dim_size * inner_size) + ((d + 1) * inner_size) + i
                        
                        # Load values
                        val1 = nl.load(result.flatten()[flat_idx1])
                        val2 = nl.load(result.flatten()[flat_idx2])
                        
                        # Compare and swap if needed
                        swap_needed = nl.greater(val1, val2)
                        
                        # Perform conditional swap
                        if_swapped_val1 = val2
                        if_swapped_val2 = val1
                        
                        new_val1 = nl.where(swap_needed, if_swapped_val1, val1)
                        new_val2 = nl.where(swap_needed, if_swapped_val2, val2)
                        
                        # Store back
                        nl.store(result.flatten()[flat_idx1], new_val1)
                        nl.store(result.flatten()[flat_idx2], new_val2)
    
    return result