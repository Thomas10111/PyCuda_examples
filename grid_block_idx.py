
import timeit
import numpy
import hashlib
import pathlib

# PyCUDA
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.compiler as compiler
from pycuda.compiler import SourceModule

BLOCK_SIZE = 32  # max 1024

mod = SourceModule("""
// Test compile with "nvcc --cubin -arch sm_61 -m64 -Ie:\\src\\ifdm\\multiscale\\venv\\lib\\site-packages\\pycuda\\cuda kernel.cu"

#include <cstdint>
 
__global__ void update(
    unsigned int count)
{   
    // const int idx = threadIdx.x + blockIdx.x * blockDim.x;   // 1D
    
    const int blockId = blockIdx.x + blockIdx.y * blockDim.x;   // 2D
    const int thread_idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;   // 2D    
    
    if(thread_idx >= count) return;
    
    printf("threadIdx.x: %2d  threadIdx.y: %2d  blockIdx.x: %2d  blockIdx.y: %2d blockDim.x: %2d  blockDim.y: %2d\\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y);
}
""")

last_id = 66

if __name__ == '__main__':
    pycuda.driver.init()

    update_all_individuals = mod.get_function("update")
    grid_x = (last_id + BLOCK_SIZE - 1) // BLOCK_SIZE
    print("grid_x: ", grid_x)

    start_time = timeit.default_timer()
    update_all_individuals(numpy.uint32(last_id),  # unsigned int count
                           block=(BLOCK_SIZE, 1, 1),
                           grid=(grid_x, 1),
                           time_kernel=True
                           )

    print(timeit.default_timer() - start_time)

