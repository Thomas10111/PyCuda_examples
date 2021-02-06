import numpy

# PyCUDA
import pycuda.driver as cuda
import pycuda.autoinit

from pycuda.compiler import SourceModule


mod = SourceModule("""
// Test compile with "nvcc --cubin -arch sm_61 -m64 -Ie:\\src\\ifdm\\multiscale\\venv\\lib\\site-packages\\pycuda\\cuda kernel.cu"

#include <cstdint>
 
__global__ void update(
    unsigned int count)
{   
    // const int idx = threadIdx.x + blockIdx.x * blockDim.x;   // 1D
    
    const int idx_x = blockIdx.x * blockDim.x + threadIdx.x;    // 2D index for x-coord
    const int idx_y = blockIdx.y * blockDim.y + threadIdx.y;    // 2D index for y-coord
    
    const int blockId = blockIdx.x + blockIdx.y * blockDim.x;   // 2D
    const int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x; // global index   
    
    if(idx >= count) return;
    
    printf("%10d  %10d  %10d  %10d %10d  %10d\\n", 
           threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y);
   
    printf("%10d  %10d \\n", idx_x, idx_y);
}
""")

last_id = 31

if __name__ == '__main__':
    pycuda.driver.init()

    print_index = mod.get_function("update")

    print("threadIdx.x | threadIdx.y | blockIdx.x | blockIdx.y | blockDim.x | blockDim.y ")

    print_index(numpy.uint32(last_id),
                block=(3, 2, 1),  # max 36 lines
                grid=(3, 2, 1),
                time_kernel=True
                )


