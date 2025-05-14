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
    
    # Get tensor shape
    tensor_shape = a_tensor.shape
    ndims = len(tensor_shape)
    
    # For 1D tensor, sort the entire tensor
    if ndims == 1:
        # Copy input to result
        for p in nl.affine_range(math.ceil(tensor_shape[0]/nl.tile_size.pmax)):
            start_idx = p * nl.tile_size.pmax
            i_p = nl.arange(nl.tile_size.pmax)
            
            # Load input data
            x_tile = nl.load(a_tensor[start_idx + i_p], mask=(start_idx + i_p < tensor_shape[0]))
            
            # Store to result
            nl.store(result[start_idx + i_p], x_tile, mask=(start_idx + i_p < tensor_shape[0]))
        
        # Bubble sort implementation
        n = tensor_shape[0]
        for i in range(n):
            for j in range(0, n-i-1):
                # Load adjacent elements
                elem1 = nl.load(result[j])
                elem2 = nl.load(result[j+1])
                
                # Compare and swap if needed
                cond = nl.greater(elem1, elem2)
                
                # Store the smaller value at j
                nl.store(result[j], nl.where(cond, elem2, elem1))
                
                # Store the larger value at j+1
                nl.store(result[j+1], nl.where(cond, elem1, elem2))
        
        return result
    
    # For multi-dimensional tensors, sort along the specified dimension
    else:
        # Calculate sizes for tiling
        outer_dims = tensor_shape[:dim]
        sort_dim_size = tensor_shape[dim]
        inner_dims = tensor_shape[dim+1:] if dim < ndims - 1 else []
        
        # Product of outer dimensions
        outer_size = 1
        for d in outer_dims:
            outer_size *= d
            
        # Product of inner dimensions
        inner_size = 1
        for d in inner_dims:
            inner_size *= d
        
        # Copy input to result first
        for p_outer in nl.affine_range(math.ceil(outer_size/nl.tile_size.pmax)):
            start_outer = p_outer * nl.tile_size.pmax
            i_p_outer = nl.arange(nl.tile_size.pmax)
            
            for sort_idx in range(sort_dim_size):
                for p_inner in nl.affine_range(math.ceil(inner_size/nl.tile_size.pmax)):
                    start_inner = p_inner * nl.tile_size.pmax
                    i_p_inner = nl.arange(nl.tile_size.pmax)
                    
                    # Calculate multi-dimensional indices
                    outer_indices = []
                    temp_idx = start_outer + i_p_outer[:, None]
                    for d in range(len(outer_dims)):
                        if d < len(outer_dims) - 1:
                            outer_indices.append(temp_idx // math.prod(outer_dims[d+1:]))
                            temp_idx = temp_idx % math.prod(outer_dims[d+1:])
                        else:
                            outer_indices.append(temp_idx)
                    
                    inner_indices = []
                    temp_idx = start_inner + i_p_inner[None, :]
                    for d in range(len(inner_dims)):
                        if d < len(inner_dims) - 1:
                            inner_indices.append(temp_idx // math.prod(inner_dims[d+1:]))
                            temp_idx = temp_idx % math.prod(inner_dims[d+1:])
                        else:
                            inner_indices.append(temp_idx)
                    
                    # Build full index
                    full_idx = []
                    for idx in outer_indices:
                        full_idx.append(idx)
                    full_idx.append(sort_idx)
                    for idx in inner_indices:
                        full_idx.append(idx)
                    
                    # Load and store
                    if len(outer_indices) == 0 and len(inner_indices) == 0:
                        # Simple case: just 1D along sort dimension
                        x_tile = nl.load(a_tensor[sort_idx])
                        nl.store(result[sort_idx], x_tile)
                    elif len(outer_indices) == 0:
                        # Only inner dimensions
                        x_tile = nl.load(a_tensor[sort_idx, start_inner + i_p_inner], 
                                        mask=(start_inner + i_p_inner < inner_size))
                        nl.store(result[sort_idx, start_inner + i_p_inner], x_tile, 
                                mask=(start_inner + i_p_inner < inner_size))
                    elif len(inner_indices) == 0:
                        # Only outer dimensions
                        x_tile = nl.load(a_tensor[start_outer + i_p_outer, sort_idx], 
                                        mask=(start_outer + i_p_outer < outer_size))
                        nl.store(result[start_outer + i_p_outer, sort_idx], x_tile, 
                                mask=(start_outer + i_p_outer < outer_size))
                    else:
                        # Both outer and inner dimensions
                        # For simplicity in this example, handle 2D case with sort_dim=1
                        if dim == 1 and ndims == 2:
                            x_tile = nl.load(a_tensor[start_outer + i_p_outer, sort_idx], 
                                            mask=(start_outer + i_p_outer < outer_size))
                            nl.store(result[start_outer + i_p_outer, sort_idx], x_tile, 
                                    mask=(start_outer + i_p_outer < outer_size))
        
        # Perform bubble sort along the sort dimension
        for outer_idx in range(outer_size):
            outer_indices = []
            temp_idx = outer_idx
            for d in range(len(outer_dims)):
                if d < len(outer_dims) - 1:
                    outer_indices.append(temp_idx // math.prod(outer_dims[d+1:]))
                    temp_idx = temp_idx % math.prod(outer_dims[d+1:])
                else:
                    outer_indices.append(temp_idx)
                    
            for inner_idx in range(inner_size):
                inner_indices = []
                temp_idx = inner_idx
                for d in range(len(inner_dims)):
                    if d < len(inner_dims) - 1:
                        inner_indices.append(temp_idx // math.prod(inner_dims[d+1:]))
                        temp_idx = temp_idx % math.prod(inner_dims[d+1:])
                    else:
                        inner_indices.append(temp_idx)
                
                # Bubble sort this slice
                n = sort_dim_size
                for i in range(n):
                    for j in range(0, n-i-1):
                        # Build indices for adjacent elements
                        idx1 = []
                        for idx in outer_indices:
                            idx1.append(int(idx))
                        idx1.append(j)
                        for idx in inner_indices:
                            idx1.append(int(idx))
                            
                        idx2 = []
                        for idx in outer_indices:
                            idx2.append(int(idx))
                        idx2.append(j+1)
                        for idx in inner_indices:
                            idx2.append(int(idx))
                        
                        # Handle different dimensionality cases
                        if ndims == 2:
                            # 2D case
                            elem1 = nl.load(result[idx1[0], idx1[1]])
                            elem2 = nl.load(result[idx2[0], idx2[1]])
                            
                            # Compare and swap if needed
                            cond = nl.greater(elem1, elem2)
                            
                            # Store the values
                            nl.store(result[idx1[0], idx1[1]], nl.where(cond, elem2, elem1))
                            nl.store(result[idx2[0], idx2[1]], nl.where(cond, elem1, elem2))
                        elif ndims == 3:
                            # 3D case
                            if len(idx1) == 3:
                                elem1 = nl.load(result[idx1[0], idx1[1], idx1[2]])
                                elem2 = nl.load(result[idx2[0], idx2[1], idx2[2]])
                                
                                # Compare and swap if needed
                                cond = nl.greater(elem1, elem2)
                                
                                # Store the values
                                nl.store(result[idx1[0], idx1[1], idx1[2]], nl.where(cond, elem2, elem1))
                                nl.store(result[idx2[0], idx2[1], idx2[2]], nl.where(cond, elem1, elem2))
    
    return result