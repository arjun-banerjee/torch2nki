import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import numpy as np


@nki.jit
def print_kernel():
  a = nl.ndarray([4, 4], dtype=nl.float32, buffer=nl.shared_hbm)

  # Create (4, 4) tensor in sbuf
  y = nl.zeros([4, 4], dtype=np.float32)

  # Print tensor y
  nl.device_print("value of y:", y)

  # Directly store tensor y as a single tile
  nl.store(a, value=y)

  return a

np.random.seed(0)

a = nki.simulate_kernel(print_kernel)

assert np.allclose(a, np.zeros([4, 4]))