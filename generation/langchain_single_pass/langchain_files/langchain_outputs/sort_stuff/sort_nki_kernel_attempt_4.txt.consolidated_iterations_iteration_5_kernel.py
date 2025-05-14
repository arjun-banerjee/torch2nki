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
    
    # Copy input to result since we'll be sorting in-place
    for p in nl.affine_range(math.ceil(shape[0]/nl.tile_size.pmax)):
        i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]
        
        if ndim == 1:
            x_tile = nl.load(a_tensor[i_p], mask=(i_p < shape[0]))
            nl.store(result[i_p], value=x_tile, mask=(i_p < shape[0]))
        else:
            i_f = nl.arange(shape[1])[None, :]
            x_tile = nl.load(a_tensor[i_p, i_f], mask=(i_p < shape[0]))
            nl.store(result[i_p, i_f], value=x_tile, mask=(i_p < shape[0]))
    
    # Sort along the specified dimension
    if ndim == 1 or dim == 0:
        # Sort along first dimension
        size = shape[0]
        for i in nl.affine_range(size):
            for j in nl.affine_range(size - i - 1):
                if ndim == 1:
                    # Load current and next element
                    curr = nl.load(result[j])
                    next_val = nl.load(result[j + 1])
                    
                    # Compare and swap if needed
                    swap = nl.greater(curr, next_val)
                    if swap:
                        nl.store(result[j], next_val)
                        nl.store(result[j + 1], curr)
                else:
                    # For 2D tensors, sort each column independently
                    for k in nl.affine_range(shape[1]):
                        curr = nl.load(result[j, k])
                        next_val = nl.load(result[j + 1, k])
                        
                        swap = nl.greater(curr, next_val)
                        if swap:
                            nl.store(result[j, k], next_val)
                            nl.store(result[j + 1, k], curr)
    elif dim == 1 and ndim >= 2:
        # Sort along second dimension
        for i in nl.affine_range(shape[0]):
            size = shape[1]
            for j in nl.affine_range(size):
                for k in nl.affine_range(size - j - 1):
                    # Load current and next element
                    curr = nl.load(result[i, k])
                    next_val = nl.load(result[i, k + 1])
                    
                    # Compare and swap if needed
                    swap = nl.greater(curr, next_val)
                    if swap:
                        nl.store(result[i, k], next_val)
                        nl.store(result[i, k + 1], curr)
    
    return result