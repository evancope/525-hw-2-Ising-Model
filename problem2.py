import numpy as np
from functionfile import temp_sweep
import matplotlib.pyplot as plt

 #declaring sample count
stepcount = 2500                     
latticecount = 2000
Temps = np.linspace(0.2, 5, 100)

#running simulation
ie10, hc10, mag10, criticaltemp10 = temp_sweep(10, 100, stepcount, latticecount, Temps)
ie20, hc20, mag20, criticaltemp20 = temp_sweep(20, 100, stepcount, latticecount, Temps)

# graphing and stuff
print('----------------------------------------------------')
print(f'critical temperatuature, n = 10 : {criticaltemp10}')
print(f'critical temperatuature, n = 20 : {criticaltemp20}')
print('----------------------------------------------------')

fig, ax = plt.subplots(2, 2)
ax[0,0].plot(Temps, ie10, c = '#009DDC'); ax[0,0].set_title("Internal Energy vs Temp, n = 10")
ax[0,1].plot(Temps, hc10, c = '#F26430'); ax[0,1].set_title("Heat Capacity vs Temp, n = 10")
ax[1,0].plot(Temps, ie20, c = '#009DDC'); ax[1,0].set_title("Internal Energy vs Temp, n = 20")
ax[1,1].plot(Temps, hc20, c = '#F26430'); ax[1,1].set_title("Heat Capacity vs Temp, n = 20")
plt.show() 

fig, axs = plt.subplots(1, 2)
axs[0].plot(Temps, mag10, c='#009DDC')
axs[0].set(xlabel=r'temperature in $J / k_b$', ylabel=r'$\langle s \rangle$')
axs[0].set_title('Mean Spin vs Temperature for n = 10')
axs[1].plot(Temps, mag20, c='#009DDC')
axs[1].set(xlabel=r'temperature in $J / k_b$')
axs[1].set_title('Mean Spin vs Temperature for n = 20')
plt.show()