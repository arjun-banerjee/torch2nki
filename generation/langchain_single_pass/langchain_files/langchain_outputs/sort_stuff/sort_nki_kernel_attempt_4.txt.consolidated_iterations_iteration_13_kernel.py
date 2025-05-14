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
    
    # Handle 1D tensor separately
    if ndim == 1:
        size = shape[0]
        # Copy input to result
        for i in nl.affine_range(math.ceil(size / nl.tile_size.pmax)):
            start_idx = i * nl.tile_size.pmax
            indices = start_idx + nl.arange(nl.tile_size.pmax)
            input_tile = nl.load(a_tensor[indices], mask=(indices < size))
            nl.store(result[indices], input_tile, mask=(indices < size))
        
        # Bubble sort
        for i in range(size - 1):
            for j in range(size - i - 1):
                # Create indices for accessing adjacent elements
                idx1 = nl.arange(1)
                idx2 = nl.arange(1)
                
                # Load elements
                a = nl.load(result[j:j+1])
                b = nl.load(result[j+1:j+2])
                
                # Compare and swap if needed
                condition = nl.greater(a, b)
                
                # Store back the smaller value at j
                nl.store(result[j:j+1], nl.where(condition, b, a))
                
                # Store back the larger value at j+1
                nl.store(result[j+1:j+2], nl.where(condition, a, b))
    else:
        # For multi-dimensional tensors, we need to handle sorting along the specified dimension
        # Determine the size of the dimension to sort along
        sort_dim_size = shape[dim]
        
        # We'll reshape our approach based on whether we're sorting along the first dimension or not
        if dim == 0:
            # Sorting along first dimension
            # For each position in the remaining dimensions, sort the slice along dim 0
            
            # Calculate total size of remaining dimensions
            remaining_size = 1
            for i in range(1, ndim):
                remaining_size *= shape[i]
                
            # First copy input to result
            for p in nl.affine_range(math.ceil(shape[0] / nl.tile_size.pmax)):
                start_p = p * nl.tile_size.pmax
                p_indices = start_p + nl.arange(nl.tile_size.pmax)[:, None]
                
                for f in nl.affine_range(math.ceil(remaining_size / nl.tile_size.fmax)):
                    start_f = f * nl.tile_size.fmax
                    f_indices = start_f + nl.arange(nl.tile_size.fmax)[None, :]
                    
                    input_tile = nl.load(a_tensor[p_indices, f_indices], 
                                         mask=((p_indices < shape[0]) & (f_indices < remaining_size)))
                    nl.store(result[p_indices, f_indices], input_tile, 
                             mask=((p_indices < shape[0]) & (f_indices < remaining_size)))
            
            # Now sort each column
            for f in range(remaining_size):
                # Bubble sort for this column
                for i in range(sort_dim_size - 1):
                    for j in range(sort_dim_size - i - 1):
                        # Load adjacent elements
                        a = nl.load(result[j:j+1, f:f+1])
                        b = nl.load(result[j+1:j+2, f:f+1])
                        
                        # Compare and swap if needed
                        condition = nl.greater(a, b)
                        
                        # Store results
                        nl.store(result[j:j+1, f:f+1], nl.where(condition, b, a))
                        nl.store(result[j+1:j+2, f:f+1], nl.where(condition, a, b))
        else:
            # Sorting along a non-first dimension
            # For simplicity, we'll handle the common case of sorting along the last dimension (dim=-1)
            # which is the most frequent use case
            
            # Calculate size of dimensions before the sort dimension
            prefix_size = 1
            for i in range(dim):
                prefix_size *= shape[i]
                
            # First copy input to result
            for p in nl.affine_range(math.ceil(prefix_size / nl.tile_size.pmax)):
                start_p = p * nl.tile_size.pmax
                p_indices = start_p + nl.arange(nl.tile_size.pmax)[:, None]
                
                for f in nl.affine_range(math.ceil(sort_dim_size / nl.tile_size.fmax)):
                    start_f = f * nl.tile_size.fmax
                    f_indices = start_f + nl.arange(nl.tile_size.fmax)[None, :]
                    
                    input_tile = nl.load(a_tensor[p_indices, f_indices], 
                                         mask=((p_indices < prefix_size) & (f_indices < sort_dim_size)))
                    nl.store(result[p_indices, f_indices], input_tile, 
                             mask=((p_indices < prefix_size) & (f_indices < sort_dim_size)))
            
            # Now sort each row
            for p in range(prefix_size):
                # Bubble sort for this row
                for i in range(sort_dim_size - 1):
                    for j in range(sort_dim_size - i - 1):
                        # Load adjacent elements
                        a = nl.load(result[p:p+1, j:j+1])
                        b = nl.load(result[p:p+1, j+1:j+2])
                        
                        # Compare and swap if needed
                        condition = nl.greater(a, b)
                        
                        # Store results
                        nl.store(result[p:p+1, j:j+1], nl.where(condition, b, a))
                        nl.store(result[p:p+1, j+1:j+2], nl.where(condition, a, b))
    
    return result