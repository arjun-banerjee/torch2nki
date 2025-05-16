from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # Extract dimensions
    batch_size, in_channels, input_length = x.shape
    out_channels, in_channels_per_group, kernel_size = weight.shape
    
    # Calculate output length
    output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    
    # Initialize result array
    result = nl.ndarray((batch_size, out_channels, output_length), dtype=x.dtype, buffer=nl.shared_hbm)
    
    # Calculate number of tiles needed to process the input batch dimension
    b_tiles = math.ceil(batch_size / nl.tile_size.pmax)
    
    # Process each output batch in tiles
    for b_idx in nl.affine_range(b_tiles):
        # Calculate current batch tile indices
        b_start = b_idx * nl.tile_size.pmax
        b_end = min(batch_size, (b_idx + 1) * nl.tile_size.pmax)
        b_size = b_end - b_start
        
        # Process each group
        for g in nl.affine_range(groups):
            # Calculate channel ranges for this group
            in_ch_start = g * in_channels_per_group
            in_ch_end = (g + 1) * in_channels_per_group
            out_ch_start = (g * out_channels) // groups
            out_ch_end = ((g + 1) * out_channels) // groups
            oc_size = out_ch_end - out_ch_start
            
            # Process output channels in tiles if needed
            oc_tiles = math.ceil(oc_size / nl.tile_size.pmax)
            for oc_idx in nl.affine_range(oc_tiles):
                oc_start = out_ch_start + oc_idx * nl.tile_size.pmax
                oc_end = min(out_ch_end, out_ch_start + (oc_idx + 1) * nl.tile_size.pmax)
                curr_oc_size = oc_end - oc_start
                
                # Process output length in tiles if needed
                ol_tiles = math.ceil(output_length / nl.tile_size.pmax)
                for ol_idx in nl.affine_range(ol_tiles):
                    ol_start = ol_idx * nl.tile_size.pmax
                    ol_end = min(output_length, (ol_idx + 1) * nl.tile_size.pmax)
                    curr_ol_size = ol_end - ol_start
                    
                    # Create output tile for current batch
                    out_tile = nl.zeros((b_size, curr_oc_size, curr_ol_size), dtype=x.dtype, buffer=nl.sbuf)
                    
                    # For each position in the kernel
                    for k in nl.affine_range(kernel_size):
                        # Calculate input indices with dilation
                        in_pos = ol_start * stride - padding + k * dilation
                        
                        # Skip if outside input bounds
                        if in_pos < 0 or in_pos >= input_length:
                            continue
                        
                        # Process each input position that maps to the current output position
                        for ol_offset in nl.affine_range(curr_ol_size):
                            in_idx = in_pos + ol_offset * stride
                            
                            # Skip invalid positions (outside padding)
                            if in_idx < 0 or in_idx >= input_length:
                                continue
                                
                            # Load input data for current position
                            x_tile = nl.load(x[b_start:b_end, in_ch_start:in_ch_end, in_idx])
                            
                            # Load corresponding weights
                            w_tile = nl.load(weight[oc_start:oc_end, 0:in_channels_per_group, k])
                            
                            # Perform multiplication and accumulate
                            for ic in nl.affine_range(in_channels_per_group):
                                product = nl.multiply(x_tile[:, ic:ic+1, 0], w_tile[:, ic, 0])
                                out_tile[:, :, ol_offset] = nl.add(out_tile[:, :, ol_offset], product)
                    
                    # Add bias if provided
                    if bias is not None:
                        bias_tile = nl.load(bias[oc_start:oc_end])
                        for ol in nl.affine_range(curr_ol_size):
                            out_tile[:, :, ol] = nl.add(out_tile[:, :, ol], bias_tile)
                    
                    # Store result
                    nl.store(result[b_start:b_end, oc_start:oc_end, ol_start:ol_end], out_tile)
    
    return result