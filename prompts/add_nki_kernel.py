Here is the custom kernel for the 'add' operation using AWS Neural Kernel Interface (NKI):

```python
from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_add(a_tensor, b_tensor):
    """
    This function adds two tensors element-wise using AWS Neural Kernel Interface (NKI).

    Parameters:
    a_tensor (numpy.ndarray): The first input tensor.
    b_tensor (numpy.ndarray): The second input tensor.

    Returns:
    numpy.ndarray: The result of adding 'a_tensor' and 'b_tensor' element-wise.
    """
    # Ensure both tensors are of the same shape
    if a_tensor.shape != b_tensor.shape:
        raise ValueError("Tensors must be of the same shape")

    # Initialize a tensor to hold the result
    result = nl.zeros(a_tensor.shape, dtype=a_tensor.dtype)

    # Process the addition
    dummy = nl.add(a_tensor, b_tensor)

    # Store the result
    result = dummy

    return result
```

This function uses the `nl.add` function to add two tensors element-wise. It is vectorized and efficient because it uses the underlying C implementation of NKI, which is much faster than Python's built-in addition operation.