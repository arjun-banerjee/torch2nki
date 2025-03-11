from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa 
from transpose_2d import tensor_transpose2D_kernel_
import itertools 


# by default, ensure that the P dimension is at axis 0 
@nki.jit
def nki_transpose_nd(a_tensor, axes): 


    


    # At first, the a_tensor sits in s_buf with P = a_tensor.shape[0], F = prod(a_tensor.shape[1:])
    P = a_tensor.shape[0] 
    # current_order[i] = j means that the i-th dimension of our current tensor is the j-th dimension of the original tensor 
    # rev_current_order[i] = j means that the the i-th dimension of our original tensor is in the j-th dimension of our current tensor 
    current_order = list(range(len(a_tensor.shape)))
    p_dim = 0  
    axes = (0, 1, 4, 2, 3)
    out_shape = [0]*len(a_tensor.shape)
    

    # FIRST, ASSUME THAT THE P DIMENSION IS UNCHANGED IN the transposition 
    # Creating indices 
    out_prod = 1.0 
    for i, shape in enumerate(a_tensor.shape):

        out_shape[axes[i]] = shape
        out_prod *= shape 

    out_prod /= a_tensor.shape[0]
    out_tensor = nl.ndarray((a_tensor.shape[p_dim], out_prod), dtype=a_tensor.dtype,
                          buffer=nl.shared_hbm)
    sz_p = a_tensor.shape[0]
    in_tile = nl.load(a_tensor)
    szs = a_tensor.shape
    n_dims = len(axes)

    base = [None]* n_dims
    idxs = [nl.arange(sz) for sz in szs]


    for i, _ in enumerate(szs):
        base[i] = slice(None) 
        idxs[i] = idxs[i][tuple(base)]
        base[i] = None 


    #idxs = [nl.arange(sz_p)[:, None, None], nl.arange(szs[1])[None, :, None], nl.arange(szs[2])[None, None, :]]
    
    prod = szs[1] 
    for sz in szs[2:]: 
        prod *= sz

    in_tile = in_tile.reshape((sz_p, prod))

    n = len(szs)

    # lets say old tensor has shape (a, b, c, d)
    # Then our in_strides should be (i_f4 *(b*c*d) + i_f3*(c*d) + i_f2*d + i_f1)
    # 

    in_strides = [1]*n
    out_strides = [1]*n
    
    i = n-1
    while i > 0: 
        in_strides[i-1] = in_strides[i]*szs[i]
        out_strides[i-1] = out_strides[i]*out_shape[i]
        i -= 1

    #out_strides = [in_strides[ax] for ax in axes]
    print(a_tensor.shape, out_shape)
    print(in_strides, out_strides)

    # id_1*2 + id_2 = id_2*2 + id_1 

    # out_tile[i_p0, i_fn*sz_f1 + i_f(n-1)*sz_f2 + ... + i_f1] = in_tile[i_p0, i_f1*sz_f2 + ... + i_f(n-1)]
    offset_in = [idxs[0], sum([idxs[i]*in_strides[i] for i in range(1, n)])] 
    offset_out = [idxs[0], sum([idxs[axes[i]]*out_strides[i] for i in range(1, n)])]


    out_tile = nl.ndarray(shape=(sz_p, prod), dtype=out_tensor.dtype)
    out_tile[tuple(offset_out)] = nl.copy(in_tile[tuple(offset_in)])
    nl.store(out_tensor, value=out_tile)

    return out_tensor








