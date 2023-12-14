#!/bin/python3
# Ian Mu;oz Nu;ez - Funcion 'Sinusoidal'

import numpy as np
import matplotlib.pyplot as plt

v = np.linspace(-10, 10, 1000) # Variable independiente

A = 1 # Parametro que determina la magnitud de la funcion
B = 1 # Parametro que determina la frecuencia de la funcion
C = 0 # Parametro que determina la fase de la funcion
phi = A*np.sin(B*v + C) # Funcion o variable dependiente

plt.figure(1)
plt.grid()

plt.plot(v, phi, 'g-', linewidth=2)

plt.title("Funcion sinusoidal", fontsize=20)
plt.xlabel('v', fontsize=15)
plt.ylabel('$\\varphi(v)$', fontsize=15)

plt.show()

