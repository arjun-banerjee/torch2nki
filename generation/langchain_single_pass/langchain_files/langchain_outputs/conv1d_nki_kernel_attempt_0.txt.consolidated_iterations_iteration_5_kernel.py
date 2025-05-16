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
    
    # Calculate iterations needed for batch and output channels (respecting pmax=128)
    batch_trips = math.ceil(batch_size / nl.tile_size.pmax)
    out_channel_trips = math.ceil(out_channels / nl.tile_size.pmax)
    
    # Process batch by batch, channel by channel
    for b_idx in nl.affine_range(batch_trips):
        b_start = b_idx * nl.tile_size.pmax
        b_end = min(batch_size, (b_idx + 1) * nl.tile_size.pmax)
        b_size = b_end - b_start
        
        for oc_idx in nl.affine_range(out_channel_trips):
            oc_start = oc_idx * nl.tile_size.pmax
            oc_end = min(out_channels, (oc_idx + 1) * nl.tile_size.pmax)
            oc_size = oc_end - oc_start
            
            # Create indices for batch and output channel dimensions
            i_b = nl.arange(b_size)[:, None, None]
            i_oc = nl.arange(oc_size)[None, :, None]
            i_out = nl.arange(output_length)[None, None, :]
            
            # Initialize accumulator for output
            output_acc = nl.zeros((b_size, oc_size, output_length), dtype=x.dtype, buffer=nl.psum)
            
            # Process groups
            for g in nl.affine_range(groups):
                ic_start_g = g * in_channels_per_group
                ic_end_g = (g + 1) * in_channels_per_group
                oc_start_g = g * (out_channels // groups)
                oc_end_g = (g + 1) * (out_channels // groups)
                
                # Only process if current output channel batch overlaps with this group
                if oc_end <= oc_start_g or oc_start >= oc_end_g:
                    continue
                
                local_oc_start = max(0, oc_start_g - oc_start)
                local_oc_end = min(oc_size, oc_end_g - oc_start)
                
                # Process each input channel
                for ic in nl.affine_range(in_channels_per_group):
                    ic_idx = ic_start_g + ic
                    
                    # For each kernel position
                    for k in nl.affine_range(kernel_size):
                        # Calculate input positions accounting for padding, stride, and dilation
                        in_pos = i_out * stride - padding + k * dilation
                        
                        # Create mask for valid input positions
                        valid_mask = (in_pos >= 0) & (in_pos < input_length)
                        
                        # Only load and compute if there are valid positions
                        if nl.any(valid_mask):
                            # Load input values for valid positions
                            input_vals = nl.zeros((b_size, output_length), dtype=x.dtype, buffer=nl.sbuf)
                            for b in nl.affine_range(b_size):
                                for out_idx in nl.affine_range(output_length):
                                    in_idx = out_idx * stride - padding + k * dilation
                                    if 0 <= in_idx < input_length:
                                        input_vals[b, out_idx] = nl.load(x[b_start + b, ic_idx, in_idx])
                            
                            # Load kernel weights for current kernel position
                            weights = nl.zeros((local_oc_end - local_oc_start, 1), dtype=weight.dtype, buffer=nl.sbuf)
                            for oc in nl.affine_range(local_oc_end - local_oc_start):
                                weights[oc, 0] = nl.load(weight[oc_start + local_oc_start + oc, ic, k])
                            
                            # Multiply input by weights and accumulate
                            for oc in nl.affine_range(local_oc_end - local_oc_start):
                                for b in nl.affine_range(b_size):
                                    for out_idx in nl.affine_range(output_length):
                                        in_idx = out_idx * stride - padding + k * dilation
                                        if 0 <= in_idx < input_length:
                                            output_acc[b, local_oc_start + oc, out_idx] += input_vals[b, out_idx] * weights[oc, 0]
            
            # Add bias if provided
            if bias is not None:
                for oc in nl.affine_range(oc_size):
                    bias_val = nl.load(bias[oc_start + oc])
                    for b in nl.affine_range(b_size):
                        for out_idx in nl.affine_range(output_length):
                            output_acc[b, oc, out_idx] += bias_val
            
            # Store result back to HBM
            i_b_real = b_start + nl.arange(b_size)[:, None, None]
            i_oc_real = oc_start + nl.arange(oc_size)[None, :, None]
            i_out_real = nl.arange(output_length)[None, None, :]
            nl.store(result[i_b_real, i_oc_real, i_out_real], value=output_acc, 
                     mask=(i_b_real < batch_size) & (i_oc_real < out_channels))
    
    return result