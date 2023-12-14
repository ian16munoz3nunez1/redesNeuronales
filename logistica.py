#!/bin/python3
# Ian Mu;oz Nu;ez - Funcion 'Logistica'

import numpy as np
import matplotlib.pyplot as plt

v = np.linspace(-10, 10, 1000) # Variable independiente

a = 1 # Parametro que determina la pendiente de la funcion
phi = 1/(1 + np.exp(-a*v)) # Funcion o variable dependiente

plt.figure(1)
plt.grid()

plt.plot(v, phi, 'g-', linewidth=2)

plt.title("Funcion logistica", fontsize=20)
plt.xlabel('v', fontsize=15)
plt.ylabel('$\phi(v)$', fontsize=15)

plt.show()

