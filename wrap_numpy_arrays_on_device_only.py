# Sample source code from the Tutorial Introduction in the documentation.

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from timeit import default_timer
import random
import numpy
import hashlib
import pathlib

# PyCUDA
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.compiler as compiler



NumberOfIndividuals = 10
MAX_INDIVIDUALS = (1 << 5)
BLOCK_SIZE = 1024


mod = """
/*
** copyright banner
*/

// Test compile with "nvcc --cubin -arch sm_61 -m64 -Ie:\\src\\ifdm\\multiscale\\venv\\lib\\site-packages\\pycuda\\cuda kernel.cu"

#include <cstdint>
//#include <curand_kernel.h>

extern "C" {
    #define NSTATES (1<<14)
    __device__ float age_device[NSTATES];
    __device__ unsigned int alive_device[NSTATES];
    __device__ float death_age_device[NSTATES];
    
    __global__ void init(
        float* age,
        unsigned int* alive,
        float* death_age)
    {        
        int idx = threadIdx.x + blockIdx.x*blockDim.x;
        
        if(idx > NSTATES) return;
           
        age_device[idx] = age[idx];
        alive_device[idx] = alive[idx];
        death_age_device[idx] = death_age[idx];
    }
    
    __global__ void update_individuals(
        unsigned int count,
        float dt)
    {
        int idx = threadIdx.x + blockIdx.x*blockDim.x;
        
        if(idx > count) return;
        
        age_device[idx] += dt;
        
        if( age_device[idx] > death_age_device[idx] )
        {
            alive_device[idx] = 0;
        }
    }
    
    __global__ void read(
        float* age,
        unsigned int* alive,
        float* death_age)
    {
        int idx = threadIdx.x + blockIdx.x*blockDim.x;
        
        if(idx > NSTATES) return;
        
        age[idx] = age_device[idx];
        alive[idx] = alive_device[idx];
        death_age[idx] = death_age_device[idx];
    }
}"""

def load_cuda_code_individual():
    global _update_individuals_fn
    global _init_gpu_fn
    global _to_host_fn

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

    _cuda_module = cuda.module_from_file(str(path))

    _update_individuals_fn = _cuda_module.get_function("update_individuals")
    _init_gpu_fn = _cuda_module.get_function("init")
    _to_host_fn = _cuda_module.get_function("read")


load_cuda_code_individual()

class Individual_Cuda_Arrays:
    def __init__(self):
        self._alive = numpy.zeros(MAX_INDIVIDUALS, dtype=numpy.uint32)
        self._age = numpy.zeros(MAX_INDIVIDUALS, dtype=numpy.float32)
        self._death_age = numpy.zeros(MAX_INDIVIDUALS, dtype=numpy.float32)

class Individual:
    _next_id = 0
    Cuda_Arrays = Individual_Cuda_Arrays()

    def __init__(self):
        self._id = Individual.get_next_id()
        self.age = random.randint(10, 20)
        self.death_age = random.randint(60, 80)
        self.is_alive = True

    @staticmethod
    def get_next_id():
        _id = Individual._next_id
        Individual._next_id += 1
        return _id

    @property
    def is_alive(self):
        return Individual.Cuda_Arrays._alive[self._id] != 0

    @is_alive.setter
    def is_alive(self, value):
        Individual.Cuda_Arrays._alive[self._id] = 1 if value > 0 else 0

    @property
    def age(self):
        return Individual.Cuda_Arrays._age[self._id]

    @age.setter
    def age(self, value):
        Individual.Cuda_Arrays._age[self._id] = value

    @property
    def death_age(self):
        return Individual.Cuda_Arrays._death_age[self._id]

    @death_age.setter
    def death_age(self, value):
        Individual.Cuda_Arrays._death_age[self._id] = value

    @classmethod
    def init_gpu(cls):
        grid_x = (cls._next_id + BLOCK_SIZE - 1) // BLOCK_SIZE
        _init_gpu_fn( cuda.In(cls.Cuda_Arrays._age),       # float* age
                                cuda.In(cls.Cuda_Arrays._alive),     # unsigned int* alive
                                cuda.In(cls.Cuda_Arrays._death_age),    # float* death_age
                                block=(BLOCK_SIZE, 1, 1),
                                grid=(grid_x, 1),
                                time_kernel=True
                                )

    @classmethod
    def update_all_individuals(cls, dt):
        grid_x = (cls._next_id + BLOCK_SIZE - 1) // BLOCK_SIZE

        _update_individuals_fn( numpy.uint32(cls._next_id),             # unsigned int count
                                numpy.float32(dt),                      # float dt
                                block=(BLOCK_SIZE, 1, 1),
                                grid=(grid_x, 1),
                                time_kernel=True
                                )

    @classmethod
    def to_host(cls):
        grid_x = (cls._next_id + BLOCK_SIZE - 1) // BLOCK_SIZE
        _to_host_fn(cuda.Out(cls.Cuda_Arrays._age),  # float* age
                               cuda.Out(cls.Cuda_Arrays._alive),  # unsigned int* alive
                               cuda.Out(cls.Cuda_Arrays._death_age),  # float* death_age
                               block=(BLOCK_SIZE, 1, 1),
                               grid=(grid_x, 1),
                               time_kernel=True
                               )


if __name__ == '__main__':
    NumberOfRuns = 1
    duration = 60 * 365
    individuals = [Individual() for i in range(0, NumberOfIndividuals)]

    pycuda.driver.init()

    dt = 1/365
    start = default_timer()
    # -------- time -------------
    Individual.init_gpu()
    for _ in range(NumberOfRuns):
        for _ in range(duration):
            Individual.update_all_individuals(dt)

    Individual.to_host()
    # --------------------------
    print("duration: ", default_timer() - start)

    for i in individuals:
        print("alive: ", i.is_alive)

