import numpy as np
import itertools as iter
import matplotlib.pyplot as plt
from functionfile import  temp_sweep, internalenergy, specificheat, hamiltonian, neighbors

length = 4 #do not go above 4
stepcount = 10000
latticecount = 200
T = np.linspace(0.2 , 5, 100)
real_values = np.zeros((2, len(T)))

lattice_permutations = np.array(list(iter.product([-1,1],repeat= length**2)))
nb = neighbors(length, length ** 2)
Harray = np.zeros(len(lattice_permutations))

for i in range(len(lattice_permutations)):
    Harray[i] = hamiltonian(lattice_permutations[i], nb)
for t in range(len(T)):
    real_values[:, t] = internalenergy(Harray, T[t], 16), specificheat(Harray, T[t], 16)

internal_energy, heat_capacity, s, crtemp = temp_sweep(length, int(length ** 2), stepcount, latticecount, T)

fig, axs = plt.subplots(1, 2)
axs[0].plot(T, real_values[0], c='#F26430', label = 'true values', linestyle = '--')
axs[0].plot(T, internal_energy,c='#009DDC', label = 'Metropolis Approximation')
axs[0].set(xlabel=r'temperature in $J / k_b$', ylabel='Mean Internal Energy in $J/K$')
axs[0].legend(loc='best')
axs[0].set_title('Mean Internal Energy vs Temperature')
axs[1].plot(T, real_values[1], c='#F26430', label = 'Actual Internal Energy', linestyle = '--')
axs[1].plot(T, heat_capacity,c='#009DDC', label = 'Metropolis Approximation')
axs[1].set(xlabel=r'temperature in $J / k_b$', ylabel='Mean Heat Capacity in $J/K$')
axs[1].legend(loc='best')
axs[1].set_title('Mean Heat Capacity vs Temperature')
plt.show()