from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_sort(a_tensor, dim=-1):
    # Handle negative dimension indexing
    if dim < 0:
        dim = len(a_tensor.shape) + dim
    
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # First copy input to result (we'll sort in place)
    for p_offset in nl.affine_range(math.ceil(math.prod(a_tensor.shape) / nl.tile_size.pmax)):
        # Calculate flat indices
        flat_indices = p_offset * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)
        # Load input data with masking to handle boundary
        input_tile = nl.load(a_tensor.reshape((-1,))[flat_indices], 
                             mask=(flat_indices < math.prod(a_tensor.shape)))
        # Store to result
        nl.store(result.reshape((-1,))[flat_indices], value=input_tile, 
                 mask=(flat_indices < math.prod(a_tensor.shape)))
    
    # Get tensor shape info
    tensor_shape = a_tensor.shape
    ndim = len(tensor_shape)
    sort_dim_size = tensor_shape[dim]
    
    # Special case for 1D tensor or when sorting along the last dimension
    if ndim == 1 or dim == ndim - 1:
        # For each slice along non-sort dimensions
        num_slices = math.prod(tensor_shape[:dim]) if dim > 0 else 1
        
        # Process each slice
        for i_slice in nl.affine_range(num_slices):
            # Calculate base offset for this slice
            base_offset = i_slice * sort_dim_size
            
            # Bubble sort implementation
            for i in nl.affine_range(sort_dim_size):
                for j in nl.affine_range(sort_dim_size - 1):
                    # Calculate indices for comparison
                    idx1 = base_offset + j
                    idx2 = base_offset + j + 1
                    
                    # Load elements to compare (one at a time to avoid size limitations)
                    val1 = nl.load(result.reshape((-1,))[idx1])
                    val2 = nl.load(result.reshape((-1,))[idx2])
                    
                    # Compare and swap if necessary
                    condition = nl.greater(val1, val2)
                    
                    # Use where to create conditional swapped values
                    new_val1 = nl.where(condition, val2, val1)
                    new_val2 = nl.where(condition, val1, val2)
                    
                    # Store swapped values back
                    nl.store(result.reshape((-1,))[idx1], value=new_val1)
                    nl.store(result.reshape((-1,))[idx2], value=new_val2)
    else:
        # For sorting along non-last dimension, we need to handle strides
        outer_dims_size = math.prod(tensor_shape[:dim]) if dim > 0 else 1
        inner_dims_size = math.prod(tensor_shape[dim+1:]) if dim < ndim - 1 else 1
        
        # Process each outer slice
        for i_outer in nl.affine_range(outer_dims_size):
            # Process each inner slice
            for i_inner in nl.affine_range(inner_dims_size):
                # Bubble sort implementation
                for i in nl.affine_range(sort_dim_size):
                    for j in nl.affine_range(sort_dim_size - 1):
                        # Calculate indices for comparison
                        idx1 = i_outer * sort_dim_size * inner_dims_size + j * inner_dims_size + i_inner
                        idx2 = i_outer * sort_dim_size * inner_dims_size + (j + 1) * inner_dims_size + i_inner
                        
                        # Load elements to compare
                        val1 = nl.load(result.reshape((-1,))[idx1])
                        val2 = nl.load(result.reshape((-1,))[idx2])
                        
                        # Compare and swap if necessary
                        condition = nl.greater(val1, val2)
                        
                        # Use where to create conditional swapped values
                        new_val1 = nl.where(condition, val2, val1)
                        new_val2 = nl.where(condition, val1, val2)
                        
                        # Store swapped values back
                        nl.store(result.reshape((-1,))[idx1], value=new_val1)
                        nl.store(result.reshape((-1,))[idx2], value=new_val2)
    
    return result