from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def nki_conv1d(input_tensor, weight_tensor, bias_tensor=None):
    # Extract dimensions
    batch_size, in_channels, in_length = input_tensor.shape
    out_channels, in_channels_per_group, kernel_size = weight_tensor.shape
    
    # Calculate groups
    groups = in_channels // in_channels_per_group
    
    # Default parameters
    stride = 1
    padding = 0
    dilation = 1
    
    # Calculate output dimensions
    out_length = (in_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Initialize result array
    result = nl.ndarray((batch_size, out_channels, out_length), dtype=input_tensor.dtype, buffer=nl.shared_hbm)
    
    # Calculate maximum partition size for processing
    max_psize = nl.tile_size.pmax
    
    # Process batch dimension in tiles
    for b in nl.affine_range(batch_size):
        # Process output channels in tiles
        for oc_start in nl.affine_range(math.ceil(out_channels / max_psize)):
            oc_size = min(max_psize, out_channels - oc_start * max_psize)
            oc_indices = oc_start * max_psize + nl.arange(max_psize)
            oc_mask = oc_indices < out_channels
            
            # Process output spatial dimension in tiles
            for ol_start in nl.affine_range(math.ceil(out_length / max_psize)):
                ol_size = min(max_psize, out_length - ol_start * max_psize)
                ol_indices = ol_start * max_psize + nl.arange(max_psize)
                ol_mask = ol_indices < out_length
                
                # Initialize output tile
                output_tile = nl.zeros((max_psize, max_psize), dtype=input_tensor.dtype)
                
                # Process groups
                for g in nl.affine_range(groups):
                    ic_start = g * in_channels_per_group
                    ic_end = (g + 1) * in_channels_per_group
                    
                    # Process kernel positions
                    for k in nl.affine_range(kernel_size):
                        # Calculate input indices for this kernel position
                        in_pos = ol_indices * stride + k * dilation - padding
                        
                        # Create masks for valid input positions
                        valid_mask = (in_pos >= 0) & (in_pos < in_length)
                        combined_mask = oc_mask[:, None] & ol_mask[None, :] & valid_mask[None, :]
                        
                        # Process input channels in tiles
                        for ic_tile_start in nl.affine_range(math.ceil(in_channels_per_group / max_psize)):
                            ic_tile_size = min(max_psize, in_channels_per_group - ic_tile_start * max_psize)
                            ic_indices = ic_start + ic_tile_start * max_psize + nl.arange(max_psize)
                            ic_tile_mask = ic_indices < ic_end
                            
                            # Load input values for current kernel position
                            input_indices = nl.arange(max_psize)
                            input_tile = nl.load(
                                input_tensor[b, ic_indices, in_pos],
                                mask=ic_tile_mask[:, None] & valid_mask[None, :]
                            )
                            
                            # Load weights for current output channels and kernel position
                            weight_tile = nl.load(
                                weight_tensor[oc_indices, ic_indices - ic_start, k],
                                mask=oc_mask[:, None] & ic_tile_mask[None, :]
                            )
                            
                            # Multiply input and weights
                            prod = nl.multiply(weight_tile[:, :, None], input_tile[None, :, :])
                            
                            # Sum over input channels
                            output_tile += nl.sum(prod, axis=1)
                
                # Add bias if provided
                if bias_tensor is not None:
                    bias_tile = nl.load(bias_tensor[oc_indices], mask=oc_mask)
                    output_tile = nl.add(output_tile, bias_tile[:, None])
                
                # Store result
                nl.store(
                    result[b, oc_indices, ol_indices],
                    value=output_tile,
                    mask=oc_mask[:, None] & ol_mask[None, :]
                )
    
    return result