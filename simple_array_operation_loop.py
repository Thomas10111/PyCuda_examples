import numpy

# PyCUDA
import pycuda.driver as cuda
import pycuda.autoinit
import numpy
import timeit

from pycuda.compiler import SourceModule


mod = SourceModule("""
// Test compile with "nvcc --cubin -arch sm_61 -m64 -Ie:\\src\\ifdm\\multiscale\\venv\\lib\\site-packages\\pycuda\\cuda kernel.cu"

#include <cstdint>
 
__global__ void update(
      unsigned int count,
      float* array_1,
      float* array_2,
      float* result
    )
{   
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;   // 1D
    
    if( idx >= count ) return;
    
    for( int i = 0; i < 100; i++ )
    {
        result[idx] = array_1[idx] * array_2[idx];
    } 
}
""")

if __name__ == '__main__':
    pycuda.driver.init()

    Array_Length = 100000   # 100.000
    BLOCK_SIZE = 1024

    # GPU arrays (unified memory)
    array_gpu_1 = pycuda.driver.managed_zeros(shape=Array_Length, dtype=numpy.float32, mem_flags=cuda.mem_attach_flags.GLOBAL)
    array_gpu_2 = pycuda.driver.managed_zeros(shape=Array_Length, dtype=numpy.float32, mem_flags=cuda.mem_attach_flags.GLOBAL)
    result_gpu = pycuda.driver.managed_zeros(shape=Array_Length, dtype=numpy.float32, mem_flags=cuda.mem_attach_flags.GLOBAL)

    array_gpu_1[:] = numpy.random.rand(Array_Length)
    array_gpu_2[:] = numpy.random.rand(Array_Length)
    result_gpu[:] = -1.0

    gpu_fn = mod.get_function("update")
    grid_x = (Array_Length + BLOCK_SIZE - 1) // BLOCK_SIZE

    start_time = timeit.default_timer()
    gpu_fn(numpy.uint32(Array_Length),
           array_gpu_1,
           array_gpu_2,
           result_gpu,
           block=(BLOCK_SIZE, 1, 1),
           grid=(grid_x, 1, 1),
           time_kernel=True
           )
    print("duration gpu: ", timeit.default_timer() - start_time)
    print("result: ", result_gpu)


    # numpy arrays
    array_numpy_1 = numpy.random.rand(Array_Length)
    array_numpy_2 = numpy.random.rand(Array_Length)
    result_numpy = numpy.ones(Array_Length) * -1.0

    start_time = timeit.default_timer()
    for _ in range(100):
        result_numpy = numpy.multiply(array_numpy_1, array_numpy_2)
    print("duration numpy: ", timeit.default_timer() - start_time)

    print("result: ", result_numpy)

