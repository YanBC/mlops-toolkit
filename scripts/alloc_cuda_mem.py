import time
import pycuda.driver as cuda
import numpy as np
cuda.init()

# Allocate 1000 MiB on GPU #0
_GPU_ID = 0
_NBYTES = 1024**2 * 1000

# Hardware: NVIDIA GeForce RTX 3090
# This next line takes up 251MiB
ctx = cuda.Device(_GPU_ID).make_context()

try:
    h_data = np.zeros(_NBYTES, dtype=np.uint8)

    # Calling mem_alloc does not cause the driver to
    # allocate any gpu memory
    d_data = cuda.mem_alloc(_NBYTES)

    # Driver will allocate memeory on GPU after
    # the first time you actually use that memory
    # allocation
    cuda.memcpy_htod(d_data, h_data)

    time.sleep(9999999)
finally:
    ctx.pop()
