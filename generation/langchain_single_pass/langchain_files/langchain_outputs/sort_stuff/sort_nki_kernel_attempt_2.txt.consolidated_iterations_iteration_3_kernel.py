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
        
    # Initialize result array with same shape and dtype as input
    result = nl.ndarray(shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Calculate sizes for processing
    dim_size = shape[dim]
    
    # Calculate outer size (product of dimensions before the sort dimension)
    outer_size = 1
    for i in range(dim):
        outer_size *= shape[i]
    
    # Calculate inner size (product of dimensions after the sort dimension)
    inner_size = 1
    for i in range(dim + 1, ndim):
        inner_size *= shape[i]
    
    # Process one batch at a time
    max_batch_size = nl.tile_size.pmax
    num_outer_batches = math.ceil(outer_size / max_batch_size)
    
    for outer_batch in nl.affine_range(num_outer_batches):
        # Calculate actual batch size for this iteration
        outer_start = outer_batch * max_batch_size
        curr_outer_size = min(max_batch_size, outer_size - outer_start)
        
        # Process inner batches if necessary
        max_inner_batch = nl.tile_size.pmax
        num_inner_batches = math.ceil(inner_size / max_inner_batch)
        
        for inner_batch in nl.affine_range(num_inner_batches):
            inner_start = inner_batch * max_inner_batch
            curr_inner_size = min(max_inner_batch, inner_size - inner_start)
            
            # Create indices for loading and storing
            i_outer = nl.arange(curr_outer_size)[:, None, None]
            i_dim = nl.arange(dim_size)[None, :, None]
            i_inner = nl.arange(curr_inner_size)[None, None, :]
            
            # Prepare buffer for this batch
            buffer = nl.zeros((curr_outer_size, dim_size, curr_inner_size), 
                             dtype=a_tensor.dtype, buffer=nl.sbuf)
            
            # Load the batch data
            # We need to map our 3D indices to the N-D tensor indices
            # First, copy input to the buffer
            if ndim == 1:
                # For 1D tensors, simply load all elements
                batch_data = nl.load(a_tensor[i_dim[:, :, 0]])
                buffer = batch_data[:, :, None]
            elif dim == 0:
                # If sorting along first dimension
                batch_data = nl.load(a_tensor[i_dim[:, :, 0], 
                                      i_outer[:, 0, :] + outer_start], 
                                    mask=(i_outer[:, 0, :] < curr_outer_size) & 
                                         (i_inner[0, 0, :] < curr_inner_size))
                buffer = batch_data
            elif dim == ndim - 1:
                # If sorting along last dimension
                batch_data = nl.load(a_tensor[i_outer[:, 0, :] + outer_start, 
                                      i_dim[:, :, 0]], 
                                    mask=(i_outer[:, 0, :] < curr_outer_size))
                buffer = batch_data
            else:
                # For middle dimensions, we need to map indices properly
                # This is a simplified case for 3D tensors
                batch_data = nl.load(a_tensor[i_outer[:, 0, :] + outer_start, 
                                      i_dim[:, :, 0], 
                                      i_inner[0, 0, :] + inner_start], 
                                    mask=(i_outer[:, 0, :] < curr_outer_size) & 
                                         (i_inner[0, 0, :] < curr_inner_size))
                buffer = batch_data
            
            # Perform bubble sort on each slice along the dimension
            for i in nl.affine_range(dim_size):
                for j in nl.affine_range(dim_size - 1 - i):
                    # Compare adjacent elements
                    mask = nl.greater(buffer[:, j, :], buffer[:, j+1, :])
                    
                    # Swap if needed
                    temp = nl.zeros_like(buffer[:, j, :])
                    temp = nl.where(mask, buffer[:, j+1, :], buffer[:, j, :])
                    buffer[:, j+1, :] = nl.where(mask, buffer[:, j, :], buffer[:, j+1, :])
                    buffer[:, j, :] = temp
            
            # Store the sorted batch back to result
            if ndim == 1:
                nl.store(result[i_dim[:, :, 0]], buffer[:, :, 0])
            elif dim == 0:
                nl.store(result[i_dim[:, :, 0], 
                                i_outer[:, 0, :] + outer_start], buffer, 
                        mask=(i_outer[:, 0, :] < curr_outer_size) & 
                             (i_inner[0, 0, :] < curr_inner_size))
            elif dim == ndim - 1:
                nl.store(result[i_outer[:, 0, :] + outer_start, 
                                i_dim[:, :, 0]], buffer, 
                        mask=(i_outer[:, 0, :] < curr_outer_size))
            else:
                nl.store(result[i_outer[:, 0, :] + outer_start, 
                                i_dim[:, :, 0], 
                                i_inner[0, 0, :] + inner_start], buffer, 
                        mask=(i_outer[:, 0, :] < curr_outer_size) & 
                             (i_inner[0, 0, :] < curr_inner_size))
    
    return result