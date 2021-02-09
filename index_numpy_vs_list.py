"""
A surprisingly big effect has numpy.float32 on the needed calculation time.
If possible do not use numpy arrays in a loop.
"""

from timeit import default_timer
from datetime import datetime
import numpy

COUNT = 10_000_000

def arrayOfFloat32(count):
    return numpy.zeros(count, dtype=numpy.float32)


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

start = datetime.now()
data = arrayOfFloat32(COUNT)
data += numpy.float32(1)
finish = datetime.now()
print(f"{finish - start} to increment {COUNT} float32 in array.")

start = datetime.now()
data = arrayOfFloat32(COUNT)
data += 1.0
finish = datetime.now()
print(f"{finish - start} to increment {COUNT} float32 in array.")

start = datetime.now()
data = arrayOfFloat32(COUNT)
data += dt
finish = datetime.now()
print(f"{finish - start} to increment {COUNT} float32 in array.")


# duration:  0.0544421
# duration:  0.1371477
# duration mixing 32 and default type:  0.8986641
# duration casting to float32:  0.28694280000000005
# duration array add default:  0.00029930000000000234
# duration array add float32:  0.00015209999999998836
# 0:00:00.021934 to increment 10000000 float32 in array.
# 0:00:00.026749 to increment 10000000 float32 in array.
# 0:00:00.022172 to increment 10000000 float32 in array.