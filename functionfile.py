import numpy as np
from numba import njit
@njit
def internalenergy(H, T, sites):
    exponentialfactor = np.exp(-H / T)
    return np.sum(H * exponentialfactor) / (sites * np.sum(exponentialfactor))
@njit
def specificheat(H, T, sites):
    exponentialfactor = np.exp(-H / T)
    Hsquared =  np.sum((H ** 2) * exponentialfactor) / (np.sum(exponentialfactor))
    avgH = np.sum(H * exponentialfactor) / (np.sum(exponentialfactor))
    return (Hsquared - avgH ** 2) / ((T ** 2) * sites)
@njit
def neighbors(sidelength, sites):
    nb = np.zeros((sites, 4), dtype = np.int32)
    for p in range(sites):
        i, j = p // sidelength, p % sidelength
        nb[p, 0] = (p - sidelength) % sites                #upper
        nb[p, 1] = i * sidelength + (j + 1) % sidelength   #right
        nb[p, 2] = (p + sidelength) % sites                #lower
        nb[p, 3] = i * sidelength + (j - 1) % sidelength   #left]
    return(nb)
@njit
def hamiltonian(array, neighbor):
    sum = 0
    for i in range(len(array)):
        sum += array[i] * (array[neighbor[i, 0]] + array[neighbor[i, 1]] \
                           + array[neighbor[i, 2]] + array[neighbor[i, 3]])
    return -1 * sum / 2
@njit
def metroboomin(lattice, neighbor, temperature):
    r = np.random.randint(len(lattice))
    halfdu = lattice[r] * (lattice[neighbor[r, 0]] + lattice[neighbor[r, 1]] \
                           + lattice[neighbor[r, 2]] + lattice[neighbor[r, 3]])
    if halfdu <= 0 or np.random.rand() <= np.exp(-2 * halfdu / temperature):
        lattice[r] = -1 * lattice[r]
    return lattice
@njit
def expectation_for_H(sidelength, sites, stepcount, latticecount, temperature):
    nb = neighbors(sidelength, sites)
    L = np.ones((latticecount, sites))
    H, s = np.zeros((latticecount, stepcount)), np.zeros((latticecount, stepcount))
    for lat in range(latticecount):
        for step in range(stepcount):
            L[lat] = metroboomin(L[lat], nb, temperature)
            H[lat, step], s[lat, step] = hamiltonian(L[lat], nb), np.mean(L[lat])
    return H, s
@njit
def values_for_single_lattice(H, s, sites, temperature):
    internal_energy = np.mean(H) / sites
    heat_capacity = np.var(H) / (sites * temperature**2)
    average_spin = np.mean(s)
    return internal_energy, heat_capacity, average_spin
@njit
def sweep_latti(H, s, sites, latticecount, temperature):
    values_of_all_latti = np.zeros((3, latticecount))
    for lat in range(latticecount):
        values_of_all_latti[:, lat] = values_for_single_lattice(H[lat], s[lat], sites, temperature)
    return np.sum(values_of_all_latti, 1) / latticecount
@njit
def temp_sweep(sidelength, sites, stepcount, latticecount, Temperature):
    mean_values_per_temp = np.zeros((3, len(Temperature)))
    for t in range(len(Temperature)):
        H, s = expectation_for_H(sidelength, sites, stepcount, latticecount, Temperature[t])
        mean_values_per_temp[:, t] = sweep_latti(H, s, sites, latticecount, Temperature[t])
        print("progress", 100.0 * (t + 1) / (len(Temperature)))
    crittemp = Temperature[np.argmax(mean_values_per_temp[1])]
    return mean_values_per_temp[0], mean_values_per_temp[1], mean_values_per_temp[2], crittemp