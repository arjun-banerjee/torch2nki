import neuronxcc.nki.language as nl
from neuronxcc import nki

@nki.jit
def nki_vector_add(a_tensor, b_tensor):
    """
    Adds two vectors element-wise using NKI.
    
    Args:
        a_tensor: First input vector tensor
        b_tensor: Second input vector tensor
    
    Returns:
        Result tensor containing element-wise sum
    """
    # Check that vectors have same length
    if a_tensor.shape != b_tensor.shape:
        raise ValueError("Vectors must be of the same length")
    
    # Initialize result array in shared HBM
    result = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
    
    # Load input vectors from HBM to on-chip memory
    a_tile = nl.load(a_tensor)
    b_tile = nl.load(b_tensor)
    
    # Perform element-wise addition
    temp = nl.add(a_tile, b_tile)
    
    # Store result back to HBM
    nl.store(result, temp)
    
    return result


import neuronxcc.nki.language as nl
import neuronxcc.nki.isa     as nisa

@nki.jit
def nki_conv2d(X, W, b):
    """
    X: [N, C,    H,      W_in ]
    W: [C_out, C,  R,      S   ]
    b: [C_out]
    Y: [N, C_out, H_out,  W_out]
    """

    
    N, C, H, W_in = X.shape
    C_out, _, R, S = W.shape
    H_out = H - R + 1
    W_out = W_in - S + 1
    Y = nl.ndarray((N, C_out, H_out, W_out), dtype = X.dtype, buffer = nl.hbm)
    # Tile sizes (tune to SBUF capacity)
    C_TILE, H_TILE, W_TILE = 64, 16, 16

    # 1) Loop over batches & spatial tiles
    for n in nl.affine_range(0, N):
      for h0 in nl.affine_range(0, H_out, H_TILE):
        for w0 in nl.affine_range(0, W_out, W_TILE):
          h_tile = min(H_TILE, H_out - h0)
          w_tile = min(W_TILE, W_out - w0)

          # 2) Initialize partial-sum buffer
          psum = nl.zeros([C_out, H_TILE * W_TILE], dtype=X.dtype)

          # 3) Channel + kernel loops for matmul-based convolution
          for c0 in nl.affine_range(0, C, C_TILE):
            for r in range(R):
              for s in range(S):
                # Load a contiguous input sub-tile: [1, C_TILE, h_tile, w_tile]
                X_tile = nl.load(
                  X[n,c0:c0+C_TILE,h0+r:h0+r+H_TILE,w0+s:w0+s+W_TILE]
                )                                                                #   :contentReference[oaicite:11]{index=11}

                # Reshape input patch to [C_TILE, h_tile*w_tile]
                X_mat = X_tile.reshape([C_TILE, h_tile * w_tile])                  #  :contentReference[oaicite:12]{index=12}

                # Load & reshape weight slice to [C_out, C_TILE]
                W_slice = nl.load(
                  W[0:C_out,
                   c0:c0+C_TILE,
                   r:r+1, s:s+1]
                ).reshape([C_out, C_TILE])                                      #      :contentReference[oaicite:13]{index=13}

                # Accumulate matmul: Tensor Engine
                psum = nl.add(
                  psum,
                  nl.matmul(W_slice, X_mat, transpose_x=False)                   #   :contentReference[oaicite:14]{index=14}
                )

          # 4) Add bias & softmax
          bias_tile = nl.load(b[0:C_out]).expand_dims(1)                      # :contentReference[oaicite:15]{index=15}
          out_mat   = nl.add(psum, bias_tile)                                   # :contentReference[oaicite:16]{index=16}
          out_mat   = nl.softmax(out_mat, axis=0)                               # :contentReference[oaicite:17]{index=17}

          # 5) Reshape & store back to HBM
          out_tile = out_mat.reshape([C_out, h_tile, w_tile])                   # :contentReference[oaicite:18]{index=18}
          nl.store(
            out_tile,
            Y[n, 0:C_out, h0:h0+h_tile, w0:w0+w_tile]
          )                                                                  #   :contentReference[oaicite:19]{index=19}
