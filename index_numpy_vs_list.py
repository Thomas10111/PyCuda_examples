"""
A surprisingly big effect has numpy.float32 on the needed calculation time.
If possible do not use numpy arrays in a loop.
"""

from timeit import default_timer
import numpy

duration = 1*365
NumberOfIndividuals = 1000
length = duration * NumberOfIndividuals
dt = 1/365

my_list = [0] * length
my_numpy = numpy.zeros(length)
my_numpy_32 = numpy.zeros(length, dtype=numpy.float32)


start = default_timer()
for i in range(length):
    my_list[i] += dt
print("duration: ", default_timer() - start)

start = default_timer()
for i in range(length):
    my_numpy[i] += dt
print("duration: ", default_timer() - start)

start = default_timer()
for i in range(length):
    my_numpy_32[i] += dt
print("duration mixing 32 and default type: ", default_timer() - start)

start = default_timer()
for i in range(length):
    my_numpy_32[i] += numpy.float32(dt)
print("duration casting to float32: ", default_timer() - start)

start = default_timer()
my_numpy += dt
print("duration array add default: ", default_timer() - start)

start = default_timer()
my_numpy_32 += dt
print("duration array add float32: ", default_timer() - start)

# Output:
# duration:  0.0469182
# duration:  0.1338455
# duration mixing 32 and default type:  0.8713339
# duration casting to float32:  0.2828889000000001
# duration array add default:  0.0003012999999998378
# duration array add float32:  0.00014270000000005112