import faulthandler

faulthandler.enable()  # Shows page faults, generates exception when using dill/pickle (disable in this case)

import timeit
import random
import numpy
from numba import float32, vectorize, uint32, void, guvectorize

# PyCUDA
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.compiler as compiler
from pycuda.compiler import SourceModule

NumberOfIndividuals = 1000
MAX_INDIVIDUALS = (1 << 15)
BLOCK_SIZE = 1024

mod = SourceModule("""
// Test compile with "nvcc --cubin -arch sm_61 -m64 -Ie:\\src\\ifdm\\multiscale\\venv\\lib\\site-packages\\pycuda\\cuda kernel.cu"

#include <cstdint>

    #define NSTATES (1<<14)

   __device__ void doFunction(unsigned int* no_of_children)
   {
        *no_of_children += 1;
   } 

    __global__ void update(
        unsigned int count,
        float dt,
        float* age,
        unsigned int* alive,
        unsigned int* no_of_children,
        float* death_age)
    {
        int idx = threadIdx.x + blockIdx.x*blockDim.x;  // calculate array index

        if(idx > count) return;                         

        age[idx] += dt;

        if( age[idx] > death_age[idx] )
        {
            alive[idx] = 0;
        }
        else if( age[idx] > 15 && age[idx] < 50 )
        {
            doFunction(&no_of_children[idx]);
        }
    }
""")



class Individual:
    _next_id = 0
    _alive = pycuda.driver.managed_zeros(shape=MAX_INDIVIDUALS, dtype=numpy.uint32, mem_flags=cuda.mem_attach_flags.GLOBAL)
    _age = pycuda.driver.managed_zeros(shape=MAX_INDIVIDUALS, dtype=numpy.float32, mem_flags=cuda.mem_attach_flags.GLOBAL)
    _death_age = pycuda.driver.managed_zeros(shape=MAX_INDIVIDUALS, dtype=numpy.float32, mem_flags=cuda.mem_attach_flags.GLOBAL)
    _no_of_children = pycuda.driver.managed_zeros(shape=MAX_INDIVIDUALS, dtype=numpy.uint32, mem_flags=cuda.mem_attach_flags.GLOBAL)

    # _alive = numpy.zeros(shape=MAX_INDIVIDUALS, dtype=numpy.uint32)
    # _age = numpy.zeros(shape=MAX_INDIVIDUALS, dtype=numpy.float32)
    # _death_age = numpy.zeros(shape=MAX_INDIVIDUALS, dtype=numpy.float32)
    # _no_of_children = numpy.zeros(shape=MAX_INDIVIDUALS, dtype=numpy.uint32)

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
        return Individual._alive[self._id] != 0

    @is_alive.setter
    def is_alive(self, value):
        Individual._alive[self._id] = 1 if value > 0 else 0

    @property
    def age(self):
        return Individual._age[self._id]

    @age.setter
    def age(self, value):
        Individual._age[self._id] = value

    @property
    def death_age(self):
        return Individual._death_age[self._id]

    @death_age.setter
    def death_age(self, value):
        Individual._death_age[self._id] = value

    @property
    def no_of_children(self):
        return Individual._no_of_children[self._id]

    @no_of_children.setter
    def no_of_children(self, value):
        Individual._no_of_children[self._id] = value

    def doFunction(self):
        Individual._no_of_children[self._id] += 1

    @classmethod
    def update_all_individuals(cls, dt):
        grid_x = (cls._next_id + BLOCK_SIZE - 1) // BLOCK_SIZE

        gpu_fn = mod.get_function("update")
        gpu_fn(numpy.uint32(cls._next_id),  # unsigned int count
               numpy.float32(dt),  # float dt
               cls._age,  # float* age
               cls._alive,  # unsigned int* alive
               cls._no_of_children,
               cls._death_age,  # float* death_age
               block=(BLOCK_SIZE, 1, 1),
               grid=(grid_x, 1),
               time_kernel=True
               )
        pycuda.driver.Context.synchronize()

    @classmethod
    def update_all_individuals_numba(cls, dt):
        numba_fn(numpy.uint32(cls._next_id),  # unsigned int count
               numpy.float32(dt),  # float dt
               cls._age,  # float* age
               cls._alive,  # unsigned int* alive
               cls._no_of_children,
               cls._death_age,  # float* death_age
               )


#@vectorize([float32(float32, float32, float32, uint32)])
@guvectorize([(uint32, float32, float32[:], uint32[:], uint32[:], float32[:])], '(),(),(n),(n),(n)->(n)', nopython=True)
def numba_fn(next_id, dt, age, alive, no_of_children, death_age):

    def doFunction(x):
        return x+1

    #for idx in range(age.shape[0]):
    for idx in range(next_id):
        age[idx] += dt

        if age[idx] > death_age[idx]:
             alive[idx] = 0

        elif age[idx] > 15 and age[idx] < 50:
            doFunction(no_of_children[idx])

        age[idx] = doFunction(987)


if __name__ == '__main__':
    NumberOfRuns = 1
    duration = 1 * 365
    individuals = [Individual() for i in range(0, NumberOfIndividuals)]

    pycuda.driver.init()

    gpu_fn = mod.get_function("update")

    dt = 1 / 365
    start_time = timeit.default_timer()
    for _ in range(duration):
        Individual.update_all_individuals(dt)
    print("duration: ", timeit.default_timer() - start_time)

    #
    # # Python
    # start_time = timeit.default_timer()
    # for _ in range(duration):
    #     for i in individuals:
    #         i.age += dt
    #
    #         if i.age > i.death_age:
    #             i.alive = 0
    #         elif 15 < i.age < 50:
    #             i.doFunction()
    #
    # print("duration python: ", timeit.default_timer() - start_time)

    Individual.update_all_individuals_numba(dt)

    start_time = timeit.default_timer()
    for _ in range(duration):
        Individual.update_all_individuals_numba(dt)

    print("duration numba: ", timeit.default_timer() - start_time)
    print(Individual._alive)
    print(Individual._age)
