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
    # Process the tensor in tiles to respect hardware limitations
    
    # Calculate dimensions for tiling
    max_tile_size = 128  # Hardware limitation
    
    # For simplicity, determine which dimensions to use for tiling based on the sorting dimension
    if dim == 0 and ndim > 1:
        # If sorting along first dimension, tile along second dimension
        outer_dim = 1
        inner_dim = 0
    else:
        # Otherwise tile along first dimension
        outer_dim = 0
        inner_dim = dim
    
    # First copy the data to result
    if ndim == 1:
        # Handle 1D tensor case separately
        trip_count = math.ceil(shape[0] / max_tile_size)
        
        for i in nl.affine_range(trip_count):
            start_idx = i * max_tile_size
            # Create indices for current tile
            indices = start_idx + nl.arange(max_tile_size)
            # Load data with masking to handle boundary
            data_tile = nl.load(a_tensor[indices], mask=(indices < shape[0]))
            # Store into result
            nl.store(result[indices], data_tile, mask=(indices < shape[0]))
    
    elif ndim == 2:
        # Handle 2D tensor case
        dim_size_outer = shape[outer_dim]
        dim_size_inner = shape[inner_dim]
        
        trip_count_outer = math.ceil(dim_size_outer / max_tile_size)
        trip_count_inner = math.ceil(dim_size_inner / max_tile_size)
        
        for i in nl.affine_range(trip_count_outer):
            start_outer = i * max_tile_size
            indices_outer = start_outer + nl.arange(max_tile_size)
            
            for j in nl.affine_range(trip_count_inner):
                start_inner = j * max_tile_size
                indices_inner = start_inner + nl.arange(max_tile_size)
                
                if outer_dim == 0 and inner_dim == 1:
                    # Load data with masking to handle boundary
                    data_tile = nl.load(a_tensor[indices_outer[:, None], indices_inner[None, :]], 
                                       mask=((indices_outer[:, None] < dim_size_outer) & 
                                             (indices_inner[None, :] < dim_size_inner)))
                    # Store into result
                    nl.store(result[indices_outer[:, None], indices_inner[None, :]], data_tile, 
                            mask=((indices_outer[:, None] < dim_size_outer) & 
                                  (indices_inner[None, :] < dim_size_inner)))
                else:  # outer_dim == 1 and inner_dim == 0
                    # Load data with masking to handle boundary
                    data_tile = nl.load(a_tensor[indices_inner[:, None], indices_outer[None, :]], 
                                       mask=((indices_inner[:, None] < dim_size_inner) & 
                                             (indices_outer[None, :] < dim_size_outer)))
                    # Store into result
                    nl.store(result[indices_inner[:, None], indices_outer[None, :]], data_tile, 
                            mask=((indices_inner[:, None] < dim_size_inner) & 
                                  (indices_outer[None, :] < dim_size_outer)))
    
    # Now perform bubble sort on each dimension
    if ndim == 1:
        # For 1D tensor, sort the entire array
        size = shape[0]
        # Bubble sort requires size-1 passes
        for _ in nl.static_range(size - 1):
            # Process in tiles
            trip_count = math.ceil((size - 1) / max_tile_size)
            
            for i in nl.affine_range(trip_count):
                start_idx = i * max_tile_size
                # Create indices for current tile, ensuring we don't go out of bounds
                indices = start_idx + nl.arange(max_tile_size)
                indices_next = indices + 1
                
                # Load current elements
                current = nl.load(result[indices], mask=(indices < size - 1))
                # Load next elements
                next_elements = nl.load(result[indices_next], mask=(indices_next < size))
                
                # Compare and swap if needed
                swap_needed = nl.less(next_elements, current)
                new_current = nl.where(swap_needed, next_elements, current)
                new_next = nl.where(swap_needed, current, next_elements)
                
                # Store back the results
                nl.store(result[indices], new_current, mask=(indices < size - 1))
                nl.store(result[indices_next], new_next, mask=(indices_next < size))
    
    elif ndim == 2:
        # For 2D tensor, sort along the specified dimension
        if dim == 0:
            # Sort along the first dimension (columns)
            dim_size = shape[0]
            other_dim = shape[1]
            
            # Process other dimension in tiles
            trip_count_other = math.ceil(other_dim / max_tile_size)
            
            for j in nl.affine_range(trip_count_other):
                start_other = j * max_tile_size
                indices_other = start_other + nl.arange(max_tile_size)
                
                # Bubble sort requires size-1 passes
                for _ in nl.static_range(dim_size - 1):
                    for i in nl.affine_range(dim_size - 1):
                        # Load current and next elements for comparison
                        current = nl.load(result[i, indices_other], 
                                        mask=(indices_other < other_dim))
                        next_elements = nl.load(result[i+1, indices_other], 
                                             mask=(indices_other < other_dim))
                        
                        # Compare and swap if needed
                        swap_needed = nl.less(next_elements, current)
                        new_current = nl.where(swap_needed, next_elements, current)
                        new_next = nl.where(swap_needed, current, next_elements)
                        
                        # Store back the results
                        nl.store(result[i, indices_other], new_current, 
                               mask=(indices_other < other_dim))
                        nl.store(result[i+1, indices_other], new_next, 
                               mask=(indices_other < other_dim))
        
        else:  # dim == 1
            # Sort along the second dimension (rows)
            dim_size = shape[1]
            other_dim = shape[0]
            
            # Process other dimension in tiles
            trip_count_other = math.ceil(other_dim / max_tile_size)
            
            for i in nl.affine_range(trip_count_other):
                start_other = i * max_tile_size
                indices_other = start_other + nl.arange(max_tile_size)
                
                # Bubble sort requires size-1 passes
                for _ in nl.static_range(dim_size - 1):
                    for j in nl.affine_range(dim_size - 1):
                        # Load current and next elements for comparison
                        current = nl.load(result[indices_other, j], 
                                        mask=(indices_other < other_dim))
                        next_elements = nl.load(result[indices_other, j+1], 
                                             mask=(indices_other < other_dim))
                        
                        # Compare and swap if needed
                        swap_needed = nl.less(next_elements, current)
                        new_current = nl.where(swap_needed, next_elements, current)
                        new_next = nl.where(swap_needed, current, next_elements)
                        
                        # Store back the results
                        nl.store(result[indices_other, j], new_current, 
                               mask=(indices_other < other_dim))
                        nl.store(result[indices_other, j+1], new_next, 
                               mask=(indices_other < other_dim))
    
    return result