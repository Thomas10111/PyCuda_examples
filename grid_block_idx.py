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
    
    const int blockId = blockIdx.x + blockIdx.y * blockDim.x;   // 2D
    const int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;    
    
    if(idx >= count) return;
    
    printf("%10d  %10d  %10d  %10d %10d  %10d\\n", 
           threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y);
}
""")

last_id = 31

if __name__ == '__main__':
    pycuda.driver.init()

    update_all_individuals = mod.get_function("update")

    print("threadIdx.x | threadIdx.y | blockIdx.x | blockIdx.y | blockDim.x | blockDim.y ")

    update_all_individuals(numpy.uint32(last_id),
                           block=(3, 2, 1),       # max 36 lines
                           grid=(3, 2, 1),
                           time_kernel=True
                           )


