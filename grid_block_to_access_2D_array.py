import numpy

# PyCUDA
import pycuda.driver as cuda
import pycuda.autoinit

from pycuda.compiler import SourceModule


mod = SourceModule("""
// Test compile with "nvcc --cubin -arch sm_61 -m64 -Ie:\\src\\ifdm\\multiscale\\venv\\lib\\site-packages\\pycuda\\cuda kernel.cu"

#include <cstdint>
 
__global__ void update(int array_2d[10000][8])
{   
    const int idx_x = blockIdx.x * blockDim.x + threadIdx.x;    // 2D index for x-coord
    const int idx_y = blockIdx.y * blockDim.y + threadIdx.y;    // 2D index for y-coord
    
    const int blockId = blockIdx.x + blockIdx.y * blockDim.x;   // 2D
    const int thread_id = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x; //global index
    
    if( idx_x >= 10000 || idx_y >= 8 ) return;
    
    array_2d[idx_x][idx_y] = thread_id;

    printf("%10d  %10d  %10d  %10d %10d  %10d %10d %10d %10d\\n", 
           threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, thread_id, idx_x, idx_y);
}
""")

if __name__ == '__main__':
    pycuda.driver.init()

    _array_2d = pycuda.driver.managed_zeros(shape=(10000, 8), dtype=numpy.int32, mem_flags=cuda.mem_attach_flags.GLOBAL)
    print_index = mod.get_function("update")

    print("threadIdx.x | threadIdx.y | blockIdx.x | blockIdx.y | blockDim.x | blockDim.y | thread_id | idx_x|  idx_y")

    print_index(
                _array_2d,
                block=(128, 8, 1),  # max 36 lines
                grid=(100, 1, 1),
                time_kernel=True
                )
    numpy.savetxt("array_2d.txt", _array_2d, fmt='%4d')


