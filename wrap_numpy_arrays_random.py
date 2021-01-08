
import timeit
import random
import numpy
import hashlib
import pathlib

# PyCUDA
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.compiler as compiler
from pycuda.compiler import SourceModule

MAX_INDIVIDUALS = (1 << 5)
BLOCK_SIZE = 1024

mod = """
// Test compile with "nvcc --cubin -arch sm_61 -m64 -Ie:\\src\\ifdm\\multiscale\\venv\\lib\\site-packages\\pycuda\\cuda kernel.cu"

#include <cstdint>
#include <curand_kernel.h>

extern "C" {
    #define NSTATES (1<<14)
    __device__ curandState_t* states[NSTATES];
    
    __global__ void initkernel(int seed)
    {   
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
        if (idx < NSTATES) 
        {
            curandState_t* s = new curandState_t;
            if (s != 0) {
                curand_init(seed, idx, 0, s);
            }
    
            states[idx] = s;
        }
    }
    
    __global__ void update_individuals(
        unsigned int count,
        float* rand)
    {
        int idx = threadIdx.x + blockIdx.x*blockDim.x;
        
        if(idx > count) return;
        
        curandState_t s = *states[idx];
        rand[idx] = curand_uniform(&s);
        
        states[idx] = &s;   //Otherwise the same random numbers are generated, at next call
    }
}"""


def load_cuda():
    md5 = hashlib.md5()
    md5.update(mod.encode("utf-8"))
    filename = md5.hexdigest() + ".cubin"
    path = pathlib.Path(__file__).resolve().parent / filename
    if not path.exists():
        try:
            cubin = compiler.compile(mod, no_extern_c=True)
            with open(str(path), "wb") as handle:
                handle.write(cubin)
        except cuda.CompileError as ce:
            print(f"{ce}")

    return cuda.module_from_file(str(path))


_cuda_module = load_cuda()
_update_individuals_fn = _cuda_module.get_function("update_individuals")
_initkernel_fn = _cuda_module.get_function("initkernel")

_rand = pycuda.driver.managed_zeros(shape=MAX_INDIVIDUALS, dtype=numpy.float32, mem_flags=cuda.mem_attach_flags.GLOBAL)

_next_id = 20
grid_x = (_next_id + BLOCK_SIZE - 1) // BLOCK_SIZE

pycuda.driver.init()
_initkernel_fn( numpy.uint32(1234), block=(BLOCK_SIZE, 1, 1), grid=(grid_x, 1), time_kernel=True )

_update_individuals_fn( numpy.uint32(_next_id), _rand, block=(BLOCK_SIZE, 1, 1), grid=(grid_x, 1), time_kernel=True )
print("rand: ", _rand)

_update_individuals_fn( numpy.uint32(_next_id), _rand, block=(BLOCK_SIZE, 1, 1), grid=(grid_x, 1), time_kernel=True )
print("rand: ", _rand)



