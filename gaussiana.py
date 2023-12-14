#!/bin/python3
# Ian Mu;oz Nu;ez - Funcion 'Gaussiana'

import numpy as np
import matplotlib.pyplot as plt

v = np.linspace(-10, 10, 1000) # Variable independiente

A = 1 # Parametro que determina la magnitud de la funcion
B = 0.1 # Parametro que determina el ancho de la campana
phi = A*np.exp(-B*v**2) # Funcion o variable dependiente

plt.figure(1)
plt.grid()

plt.plot(v, phi, 'g-', linewidth=2)

plt.title("Funcion Gaussiana", fontsize=20)
plt.xlabel('v', fontsize=15)
plt.ylabel('$\phi(v)$', fontsize=15)

plt.show()

