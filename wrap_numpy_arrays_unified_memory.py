import faulthandler
faulthandler.enable()   # Shows page faults, generates exception when using dill/pickle (disable in this case)

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

NumberOfIndividuals = 10
MAX_INDIVIDUALS = (1 << 5)
BLOCK_SIZE = 1024

mod = """
// Test compile with "nvcc --cubin -arch sm_61 -m64 -Ie:\\src\\ifdm\\multiscale\\venv\\lib\\site-packages\\pycuda\\cuda kernel.cu"

#include <cstdint>

extern "C" {
    #define NSTATES (1<<14)
    
    __global__ void update_individuals(
        unsigned int count,
        float dt,
        float* age,
        unsigned int* alive,
        float* death_age)
    {
        int idx = threadIdx.x + blockIdx.x*blockDim.x;
        
        if(idx > count) return;
        
        age[idx] += dt;
        
        if( age[idx] > death_age[idx] )
        {
            alive[idx] = 0;
        }
    }
}"""


def load_cuda_code_individual():
    global _update_individuals_fn

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


load_cuda_code_individual()


class Individual_Cuda_Arrays:
    def __init__(self):
        # Unified memory
        self._alive = pycuda.driver.managed_zeros(shape=MAX_INDIVIDUALS, dtype=numpy.uint32, mem_flags=cuda.mem_attach_flags.GLOBAL)
        self._age = pycuda.driver.managed_zeros(shape=MAX_INDIVIDUALS, dtype=numpy.float32, mem_flags=cuda.mem_attach_flags.GLOBAL)
        self._death_age = pycuda.driver.managed_zeros(shape=MAX_INDIVIDUALS, dtype=numpy.float32, mem_flags=cuda.mem_attach_flags.GLOBAL)


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
    def update_all_individuals(cls, dt):
        grid_x = (cls._next_id + BLOCK_SIZE - 1) // BLOCK_SIZE

        _update_individuals_fn(numpy.uint32(cls._next_id),  # unsigned int count
                               numpy.float32(dt),  # float dt
                               cls.Cuda_Arrays._age,  # float* age
                               cls.Cuda_Arrays._alive,  # unsigned int* alive
                               cls.Cuda_Arrays._death_age,  # float* death_age
                               block=(BLOCK_SIZE, 1, 1),
                               grid=(grid_x, 1),
                               time_kernel=True
                               )


if __name__ == '__main__':
    NumberOfRuns = 1
    duration = 60 * 365
    individuals = [Individual() for i in range(0, NumberOfIndividuals)]

    pycuda.driver.init()

    # for _ in range(duration):
    #     Individual.update_all_individuals(1.0)

    dt = 1/365
    result_time = timeit.timeit("[Individual.update_all_individuals(dt) for _ in range(duration)]", globals=globals(), number=NumberOfRuns)
    print("duration: ", result_time)

    for i in individuals:
        print("alive: ", i.is_alive)
